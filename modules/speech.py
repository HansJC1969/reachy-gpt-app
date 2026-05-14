"""
Reachy Speech — Text-to-Speech mit OpenAI TTS.

Primärsprache: Deutsch  |  Sekundärsprache: Englisch
OpenAI TTS erkennt die Sprache automatisch aus dem Text.

Audio-Pipeline:
  OpenAI TTS  →  PCM bytes (24 kHz, 16-bit, mono)
               →  numpy int16 array
               →  sounddevice output  →  Lautsprecher

Queue-basiert: speak() ist nie blockierend. Ein Background-Thread
verarbeitet die Warteschlange. Läuft kein Audiogerät, wird TTS
automatisch deaktiviert (sim_mode=True).

Standalone-Test:
    python -m modules.speech "Hallo, ich bin Reachy!"
    python -m modules.speech "Hello, I am Reachy!" --voice nova
"""

from __future__ import annotations

import logging
import os
import queue
import re
import threading
import time
from typing import Optional

import numpy as np
import openai
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Konfiguration ────────────────────────────────────────────────────────────

SAMPLE_RATE  = 24_000          # OpenAI PCM output: 24 kHz, 16-bit, mono
DEFAULT_VOICE = os.environ.get("REACHY_VOICE", "nova")
DEFAULT_SPEED = float(os.environ.get("REACHY_SPEECH_SPEED", "1.0"))
DEFAULT_MODEL = os.environ.get("REACHY_TTS_MODEL", "tts-1")   # or "tts-1-hd"

# OpenAI TTS voices (both tts-1 and tts-1-hd)
AVAILABLE_VOICES = {"alloy", "ash", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer"}

# ── Language detection ────────────────────────────────────────────────────────

_DE_CHARS  = frozenset("äöüßÄÖÜ")
_DE_WORDS  = frozenset({
    "und", "der", "die", "das", "ist", "ich", "du", "wir", "sie", "es",
    "ein", "eine", "nicht", "aber", "auf", "mit", "von", "für", "auch",
    "als", "bei", "nach", "noch", "dann", "jetzt", "hier", "so", "wie",
    "was", "dass", "haben", "sein", "kann", "wird", "bitte", "danke",
    "hallo", "ja", "nein", "gut", "sehr", "gerne", "natürlich",
})


def detect_language(text: str) -> str:
    """Return 'de' or 'en' based on a simple heuristic (no external library needed)."""
    if any(c in _DE_CHARS for c in text):
        return "de"
    words = set(re.findall(r"\b\w+\b", text.lower()))
    if len(words & _DE_WORDS) >= 2:
        return "de"
    return "en"


# ── Audio device check ────────────────────────────────────────────────────────

def sounddevice_available() -> bool:
    """Return True if sounddevice can find at least one output device."""
    try:
        import sounddevice as sd  # type: ignore
        devices = sd.query_devices()
        return any(d["max_output_channels"] > 0 for d in devices)
    except Exception:
        return False


# ── Speech engine ─────────────────────────────────────────────────────────────

_SENTINEL = object()   # unique sentinel; never put None on the queue directly


class SpeechEngine:
    """
    Queue-based, non-blocking TTS using OpenAI's audio.speech API.

    Parameters
    ----------
    api_key : str | None
        OpenAI API key.  Falls back to OPENAI_API_KEY env var.
    voice : str
        One of the OpenAI TTS voice names (default: "nova").
    speed : float
        Speech speed multiplier 0.25–4.0 (default: 1.0).
    model : str
        "tts-1" (fast, low latency) or "tts-1-hd" (higher quality).
    sim_mode : bool
        When True, synthesize but do NOT play audio (useful for testing).
    """

    def __init__(
        self,
        api_key:  Optional[str] = None,
        voice:    str   = DEFAULT_VOICE,
        speed:    float = DEFAULT_SPEED,
        model:    str   = DEFAULT_MODEL,
        sim_mode: bool  = False,
    ) -> None:
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise EnvironmentError("OPENAI_API_KEY not set — SpeechEngine cannot initialise")
        if voice not in AVAILABLE_VOICES:
            logger.warning("Unknown voice %r — falling back to 'nova'", voice)
            voice = "nova"

        self._client     = openai.OpenAI(api_key=key)
        self._voice      = voice
        self._speed      = max(0.25, min(4.0, speed))
        self._model      = model
        self._sim_mode   = sim_mode
        self._queue: queue.Queue = queue.Queue()
        self._stop_evt   = threading.Event()    # stop current utterance
        self._speaking   = threading.Event()    # set while audio is playing
        self._lock       = threading.Lock()

        self._worker_thread = threading.Thread(
            target=self._worker, name="speech-worker", daemon=True
        )
        self._worker_thread.start()
        logger.info(
            "SpeechEngine ready — voice=%s  speed=%.2f  model=%s  sim=%s",
            self._voice, self._speed, self._model, self._sim_mode,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    def speak(self, text: str, *, interrupt: bool = False) -> None:
        """
        Queue *text* for speech synthesis and playback.

        Parameters
        ----------
        text : str
            Text to speak.  Empty / whitespace-only strings are ignored.
        interrupt : bool
            If True, stop the current utterance immediately and discard
            any queued text before adding this one.
        """
        text = text.strip()
        if not text:
            return

        if interrupt:
            self._flush()

        self._queue.put(text)

    def speak_immediate(self, text: str) -> None:
        """Convenience: interrupt whatever is playing and speak *text* now."""
        self.speak(text, interrupt=True)

    def stop(self) -> None:
        """Stop the current utterance and discard everything queued."""
        self._flush()

    def wait_until_done(self, timeout: float = 30.0) -> None:
        """Block until the speech queue is empty and playback has finished."""
        # queue.join() waits until every queued item has had task_done() called,
        # which happens after _speaking.clear() in the worker's finally block —
        # so queue.join() alone is sufficient to wait for completion.
        self._queue.join()

    def shutdown(self) -> None:
        """Stop speech and terminate the worker thread cleanly."""
        self.stop()
        self._queue.put(_SENTINEL)
        self._worker_thread.join(timeout=2.0)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _flush(self) -> None:
        """Signal the current utterance to stop and drain the queue."""
        self._stop_evt.set()
        while True:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break

    def _worker(self) -> None:
        """Background thread: pull text from queue → synthesize → play."""
        while True:
            text = self._queue.get()
            if text is _SENTINEL:
                self._queue.task_done()
                break

            self._stop_evt.clear()
            lang = detect_language(text)
            logger.debug("Speaking [%s]: %r", lang, text[:80])
            self._speaking.set()
            try:
                audio = self._synthesize(text)
                if audio is not None and not self._stop_evt.is_set():
                    self._play(audio)
            except Exception:
                logger.exception("TTS worker error")
            finally:
                self._speaking.clear()
                self._queue.task_done()

    def _synthesize(self, text: str) -> Optional[np.ndarray]:
        """Call OpenAI TTS and return PCM audio as a numpy int16 array."""
        try:
            response = self._client.audio.speech.create(
                model=self._model,
                voice=self._voice,
                input=text,
                response_format="pcm",   # raw 24 kHz 16-bit mono, no decoder needed
                speed=self._speed,
            )
            pcm_bytes = response.content
            return np.frombuffer(pcm_bytes, dtype=np.int16).copy()
        except openai.APIError as exc:
            logger.error("OpenAI TTS API error: %s", exc)
            return None
        except Exception:
            logger.exception("TTS synthesis failed")
            return None

    def _play(self, audio: np.ndarray) -> None:
        """Play PCM audio via sounddevice, checking stop_evt every 250 ms."""
        if self._sim_mode:
            duration = len(audio) / SAMPLE_RATE
            logger.info("[sim] would play %.1f s of audio", duration)
            # Simulate playback duration with stop-event awareness
            deadline = time.monotonic() + duration
            while time.monotonic() < deadline and not self._stop_evt.is_set():
                time.sleep(0.05)
            return

        try:
            import sounddevice as sd  # type: ignore
        except ImportError:
            logger.warning("sounddevice not installed — audio skipped")
            return

        CHUNK = SAMPLE_RATE // 4    # 250 ms chunks for responsive interrupt
        try:
            for start in range(0, len(audio), CHUNK):
                if self._stop_evt.is_set():
                    sd.stop()
                    return
                chunk = audio[start : start + CHUNK]
                sd.play(chunk.reshape(-1, 1), samplerate=SAMPLE_RATE, dtype="int16")
                sd.wait()
        except Exception:
            logger.exception("Audio playback failed")
            try:
                sd.stop()
            except Exception:
                pass


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

    parser = argparse.ArgumentParser(description="Reachy TTS standalone test")
    parser.add_argument("text", nargs="?", default="Hallo! Ich bin Reachy, dein freundlicher Roboter.")
    parser.add_argument("--voice",  default=DEFAULT_VOICE,
                        choices=sorted(AVAILABLE_VOICES))
    parser.add_argument("--speed",  type=float, default=DEFAULT_SPEED)
    parser.add_argument("--model",  default=DEFAULT_MODEL, choices=["tts-1", "tts-1-hd"])
    parser.add_argument("--sim",    action="store_true", help="Simulate playback (no audio output)")
    parser.add_argument("--detect-lang", action="store_true", help="Only detect language, no TTS")
    args = parser.parse_args()

    if args.detect_lang:
        lang = detect_language(args.text)
        print(f"Detected language: {lang}")
        sys.exit(0)

    has_audio = sounddevice_available()
    if not has_audio and not args.sim:
        print("No audio output device found — running in sim mode.")
        args.sim = True

    engine = SpeechEngine(voice=args.voice, speed=args.speed, model=args.model, sim_mode=args.sim)
    print(f"Speaking [{detect_language(args.text)}]: {args.text!r}")
    engine.speak(args.text)
    engine.wait_until_done()
    engine.shutdown()
    print("Done.")
