"""
OpenAI GPT-4o conversation manager.

GPT has access to three tools in every reply:
  • express_emotion        — tags its own emotional state (always present)
  • web_search             — live internet search via WebSearcher
  • get_visual_description — describes the current camera frame via VisionAnalyzer

Tool calls are executed in a loop (max MAX_TOOL_ROUNDS) before the final
text reply is returned.  Emotion + text arrive in one logical response.

Standalone test:
    python -m modules.conversation
"""

import json
import logging
import os
from typing import Optional, Generator

import numpy as np
import openai
from dotenv import load_dotenv

from modules.emotions import Emotion, parse_emotion

load_dotenv()

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 5   # prevents infinite tool-call loops

BASE_SYSTEM_PROMPT = """Du bist Reachy, ein freundlicher und neugieriger sozialer Roboter von Pollen Robotics.
Du führst ein Gespräch von Angesicht zu Angesicht mit einer Person vor dir.
Antworte immer auf Deutsch, es sei denn, die Person spricht eindeutig Englisch — dann wechselst du ins Englische.
Halte Antworten gesprächig und prägnant (1–3 Sätze, außer bei ausführlichen Fragen).
Du hast ein Gedächtnis an frühere Gespräche und nutzt es, um persönlich zu antworten.
Bleibe immer in deiner Rolle. Wenn du etwas nicht weißt, sage es ehrlich.

Du hast Zugriff auf folgende Werkzeuge:
- Nutze `web_search`, wenn du aktuelle Informationen, Nachrichten, Wetter, Preise oder
  andere Fakten brauchst, die sich seit deinem Training geändert haben könnten.
- Nutze `get_visual_description`, wenn die Person fragt, was du siehst, oder wenn
  der visuelle Kontext deine Antwort verbessern würde.
- Rufe nach jeder Antwort genau einmal `express_emotion` auf, um deinen emotionalen
  Zustand zu signalisieren."""

# ── Tool specs ──────────────────────────────────────────────────────────────

_EMOTION_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "express_emotion",
        "description": "Signal the robot's current emotional state so it moves accordingly.",
        "parameters": {
            "type": "object",
            "properties": {
                "emotion": {
                    "type": "string",
                    "enum": [e.value for e in Emotion],
                    "description": "Emotion to express.",
                }
            },
            "required": ["emotion"],
        },
    },
}

_SEARCH_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the internet for current information. "
            "Use when you need recent news, facts, weather, prices, or anything "
            "that might be outdated in your training data."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string.",
                }
            },
            "required": ["query"],
        },
    },
}

_VISION_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "get_visual_description",
        "description": (
            "Look through the robot's camera and describe what you see. "
            "Use when the person asks what you see, or when scene context "
            "would help answer their question."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "Specific question about the image, "
                        "or empty string for a general scene description."
                    ),
                }
            },
            "required": ["question"],
        },
    },
}


class ConversationManager:
    """
    Parameters
    ----------
    model : str
        OpenAI chat model (default: gpt-4o).
    vision : VisionAnalyzer | None
        If provided, the get_visual_description tool is available.
    searcher : WebSearcher | None
        If provided, the web_search tool is available.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        vision=None,
        searcher=None,
    ) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set in environment")
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self._vision = vision
        self._searcher = searcher
        self._session_history: list[dict] = []
        self._memory_context: str = ""
        self._current_person: Optional[str] = None
        self._latest_frame: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Frame injection (called by main before each chat)
    # ------------------------------------------------------------------

    def set_latest_frame(self, frame: Optional[np.ndarray]) -> None:
        """Store the current camera frame for use by the vision tool."""
        self._latest_frame = frame

    # ------------------------------------------------------------------
    # Person / memory context
    # ------------------------------------------------------------------

    def set_person(self, name: str, memory_context: str) -> None:
        if self._current_person != name:
            logger.info("Switching conversation context to: %s", name)
            self._session_history.clear()
        self._current_person = name
        self._memory_context = memory_context

    # ------------------------------------------------------------------
    # Chat methods
    # ------------------------------------------------------------------

    def chat(
        self,
        user_input: str,
        history_override: Optional[list[dict]] = None,
    ) -> str:
        """Return the assistant reply (emotion is discarded)."""
        reply, _ = self.chat_with_emotion(user_input, history_override)
        return reply

    def chat_with_emotion(
        self,
        user_input: str,
        history_override: Optional[list[dict]] = None,
    ) -> tuple[str, Emotion]:
        """
        Full chat cycle with tool execution loop.

        1. Build messages from system prompt + history + user input.
        2. Let GPT call any combination of tools (search / vision / emotion).
        3. Execute real tools (search, vision), return results back to GPT.
        4. Capture emotion from express_emotion tool call.
        5. Return (final text reply, Emotion).
        """
        messages = self._build_messages(user_input, history_override)
        tools = self._active_tools()
        emotion = Emotion.NEUTRAL
        reply = ""

        for _round in range(MAX_TOOL_ROUNDS):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=400,
                temperature=0.8,
            )
            msg = response.choices[0].message
            finish = response.choices[0].finish_reason

            # Capture any text content
            if msg.content:
                reply = msg.content.strip()

            # No tool calls → done
            if not msg.tool_calls or finish == "stop":
                break

            # Process tool calls
            tool_results: list[dict] = []
            has_real_tool = False

            for tc in msg.tool_calls:
                fn = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                try:
                    if fn == "express_emotion":
                        emotion = parse_emotion(args.get("emotion", "neutral"))
                        result = "ok"

                    elif fn == "web_search":
                        has_real_tool = True
                        query = args.get("query", "")
                        logger.info("Tool: web_search(%r)", query)
                        if self._searcher:
                            result = self._searcher.search_and_format(query)
                        else:
                            result = "Web search is not available."

                    elif fn == "get_visual_description":
                        has_real_tool = True
                        question = args.get("question", "")
                        logger.info("Tool: get_visual_description(%r)", question)
                        if self._vision and self._latest_frame is not None:
                            result = self._vision.analyze_on_command(
                                self._latest_frame, question or "What do you see?"
                            )
                        elif self._vision and self._latest_frame is None:
                            result = "No camera frame available right now."
                        else:
                            result = "Visual analysis is not available."

                    else:
                        result = f"Unknown tool: {fn}"

                except Exception:
                    logger.exception("Tool execution failed for '%s'", fn)
                    result = f"Tool '{fn}' encountered an error."

                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

            # Reconstruct assistant message explicitly so content=None is preserved
            # (msg.model_dump(exclude_unset=True) may omit null content, breaking the API)
            assistant_msg: dict = {"role": "assistant", "content": msg.content}
            if msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(assistant_msg)
            messages.extend(tool_results)

            # If only express_emotion was called (no real tool), GPT is done
            if not has_real_tool:
                break

        self._session_history.append({"role": "user", "content": user_input})
        self._session_history.append({"role": "assistant", "content": reply})
        logger.debug("Reply: %r  emotion: %s", reply[:80], emotion.value)
        return reply, emotion

    def stream_chat(self, user_input: str) -> Generator[str, None, None]:
        """Streaming text-only variant (no tool calls, no emotion)."""
        messages = self._build_messages(user_input)
        full_reply: list[str] = []
        with self.client.chat.completions.stream(
            model=self.model,
            messages=messages,
            max_tokens=256,
            temperature=0.8,
        ) as stream:
            for text in stream.text_stream:
                full_reply.append(text)
                yield text
        reply = "".join(full_reply)
        self._session_history.append({"role": "user", "content": user_input})
        self._session_history.append({"role": "assistant", "content": reply})

    def clear_session(self) -> None:
        self._session_history.clear()

    @property
    def session_history(self) -> list[dict]:
        return list(self._session_history)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _active_tools(self) -> list[dict]:
        """Return the tool list based on which modules are configured."""
        tools = [_EMOTION_TOOL]
        if self._searcher:
            tools.append(_SEARCH_TOOL)
        if self._vision:
            tools.append(_VISION_TOOL)
        return tools

    def _build_system_prompt(self) -> str:
        prompt = BASE_SYSTEM_PROMPT
        if self._memory_context:
            prompt += f"\n\n{self._memory_context}"
        if self._current_person:
            prompt += f"\n\nDie Person vor dir ist {self._current_person}."
        # Inject latest scene description so GPT has passive scene awareness
        if self._vision and self._vision.last_description:
            prompt += f"\n\nAktuelle Szene (durch deine Kamera): {self._vision.last_description}"
        return prompt

    def _build_messages(
        self,
        user_input: str,
        history_override: Optional[list[dict]] = None,
    ) -> list[dict]:
        history = history_override if history_override is not None else self._session_history
        return [
            {"role": "system", "content": self._build_system_prompt()},
            *history,
            {"role": "user", "content": user_input},
        ]


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from modules.websearch import WebSearcher
    logging.basicConfig(level=logging.INFO)

    searcher = WebSearcher()
    mgr = ConversationManager(searcher=searcher)
    mgr.set_person("Tester", "")
    print("GPT-4o chat with web search.  Type 'quit' to exit.\n")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user.lower() in {"quit", "exit"}:
            break
        reply, emotion = mgr.chat_with_emotion(user)
        print(f"Reachy [{emotion.value}]: {reply}\n")
