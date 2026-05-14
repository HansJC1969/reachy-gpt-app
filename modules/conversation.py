"""
OpenAI GPT-4o conversation manager.
Maintains session history, injects memory context, and streams replies.
GPT also tags its own emotional state via function calling — the caller
receives both the text reply and an Emotion value in one API round-trip.

Standalone test:
    python -m modules.conversation
"""

import json
import os
import logging
from typing import Optional, Generator

import openai
from dotenv import load_dotenv

from modules.emotions import Emotion, parse_emotion

load_dotenv()

logger = logging.getLogger(__name__)

BASE_SYSTEM_PROMPT = """You are Reachy, a friendly and curious social robot made by Pollen Robotics.
You are having a face-to-face conversation with a person standing in front of you.
Keep responses conversational and concise (1-3 sentences unless asked for detail).
You have a memory of past interactions and will use it to personalise the conversation.
Never break character. If you don't know something, say so honestly.

After every reply you MUST call the `express_emotion` function to signal how you feel.
Pick the emotion that best matches your current response:
  neutral, freude, trauer, angst, müde, nachdenken, tanzen, ueberraschung, neugier"""

# OpenAI function spec — GPT uses this to tag its own emotion
_EMOTION_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "express_emotion",
        "description": (
            "Signal the robot's emotional state so it can move accordingly. "
            "Call this after every reply."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "emotion": {
                    "type": "string",
                    "enum": [e.value for e in Emotion],
                    "description": "The emotion Reachy should express.",
                }
            },
            "required": ["emotion"],
        },
    },
}


class ConversationManager:
    def __init__(self, model: str = "gpt-4o") -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set in environment")
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self._session_history: list[dict] = []
        self._memory_context: str = ""
        self._current_person: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_person(self, name: str, memory_context: str) -> None:
        """Switch the active person, clear session history, inject memory."""
        if self._current_person != name:
            logger.info("Switching conversation context to: %s", name)
            self._session_history.clear()
        self._current_person = name
        self._memory_context = memory_context

    def chat(
        self,
        user_input: str,
        history_override: Optional[list[dict]] = None,
    ) -> str:
        """Return the assistant reply (emotion is ignored)."""
        reply, _ = self.chat_with_emotion(user_input, history_override)
        return reply

    def chat_with_emotion(
        self,
        user_input: str,
        history_override: Optional[list[dict]] = None,
    ) -> tuple[str, Emotion]:
        """
        Send *user_input* to GPT-4o.
        Returns (reply_text, Emotion).

        GPT signals its emotion via the `express_emotion` tool call embedded
        in the same API response — no extra round-trip required.
        """
        messages = self._build_messages(user_input, history_override)
        logger.debug("Sending %d messages to %s", len(messages), self.model)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=[_EMOTION_TOOL],
            tool_choice="auto",
            max_tokens=300,
            temperature=0.8,
        )

        msg = response.choices[0].message
        reply = (msg.content or "").strip()
        emotion = Emotion.NEUTRAL

        # Parse emotion from tool call if GPT included one
        if msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.function.name == "express_emotion":
                    try:
                        args = json.loads(tc.function.arguments)
                        emotion = parse_emotion(args.get("emotion", "neutral"))
                    except (json.JSONDecodeError, KeyError):
                        pass

        # If GPT returned only a tool call and no text content, ask for a
        # follow-up message (can happen when tool_choice forces it)
        if not reply and msg.tool_calls:
            follow = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    *messages,
                    {"role": "assistant", "content": None, "tool_calls": [
                        tc.model_dump() for tc in msg.tool_calls
                    ]},
                    {"role": "tool", "tool_call_id": msg.tool_calls[0].id, "content": "ok"},
                ],
                max_tokens=256,
                temperature=0.8,
            )
            reply = (follow.choices[0].message.content or "").strip()

        self._session_history.append({"role": "user", "content": user_input})
        self._session_history.append({"role": "assistant", "content": reply})

        logger.debug("Reply: %r  |  emotion: %s", reply[:60], emotion.value)
        return reply, emotion

    def stream_chat(self, user_input: str) -> Generator[str, None, None]:
        """
        Streaming text-only variant (emotion not extracted).
        Yields text chunks; full reply is appended to session history when done.
        """
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

    def _build_system_prompt(self) -> str:
        prompt = BASE_SYSTEM_PROMPT
        if self._memory_context:
            prompt += f"\n\n{self._memory_context}"
        if self._current_person:
            prompt += f"\n\nThe person in front of you is {self._current_person}."
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
    logging.basicConfig(level=logging.INFO)
    mgr = ConversationManager()
    mgr.set_person("Tester", "")
    print("Type 'quit' to exit.\n")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user.lower() in {"quit", "exit"}:
            break
        reply, emotion = mgr.chat_with_emotion(user)
        print(f"Reachy [{emotion.value}]: {reply}\n")
