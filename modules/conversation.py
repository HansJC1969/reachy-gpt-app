"""
OpenAI GPT-4o conversation manager.
Maintains session history, injects memory context, and streams replies.
Can be used independently: python -m modules.conversation
"""

import os
import logging
from typing import Optional, Generator

import openai
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

BASE_SYSTEM_PROMPT = """You are Reachy, a friendly and curious social robot made by Pollen Robotics.
You are having a face-to-face conversation with a person standing in front of you.
Keep responses conversational and concise (1-3 sentences unless asked for detail).
You have a memory of past interactions and will use it to personalise the conversation.
Never break character. If you don't know something, say so honestly."""


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
        """
        Switch the active person.  Clears session history and injects
        memory context into the system prompt.
        """
        if self._current_person != name:
            logger.info("Switching conversation context to: %s", name)
            self._session_history.clear()
        self._current_person = name
        self._memory_context = memory_context

    def chat(self, user_input: str, history_override: Optional[list[dict]] = None) -> str:
        """
        Send *user_input* to GPT-4o and return the assistant reply.
        Appends both turns to session history.
        *history_override* lets callers supply their own history (e.g. loaded from DB).
        """
        messages = self._build_messages(user_input, history_override)
        logger.debug("Sending %d messages to %s", len(messages), self.model)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=256,
            temperature=0.8,
        )
        reply = response.choices[0].message.content.strip()

        self._session_history.append({"role": "user", "content": user_input})
        self._session_history.append({"role": "assistant", "content": reply})

        return reply

    def stream_chat(self, user_input: str) -> Generator[str, None, str]:
        """
        Streaming variant. Yields text chunks as they arrive.
        Returns the full assembled reply.
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
        return reply

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
    import sys

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
        reply = mgr.chat(user)
        print(f"Reachy: {reply}\n")
