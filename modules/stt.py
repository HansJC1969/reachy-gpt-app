"""
Speech-to-text for Reachy Mini using OpenAI Whisper.

Audio capture
-------------
On robot (reachy is not None):
    reachy.media.start_recording() / get_audio_sample() / stop_recording()
    SDK LOCAL backend — GStreamer audio IPC, 16 kHz float32 stereo.

Simulation (reachy is None):
    sounddevice InputStream from the default system microphone.

Both paths share the same energy-based VAD loop:
  1. Discard frames until RMS > SPEECH_THRESHOLD (speech onset).
  2. Record through trailing silence until SILENCE_DURATION seconds of
     quiet have elapsed.
  3. Send WAV bytes to OpenAI Whisper-1, return the transcript.

Standalone test:
    python -m modules.stt
    python -m modules.stt --no-robot   (uses system mic via sounddevice)
"""

from __future__ import annotations

import io
import logging
import os
import threading
import time
import wave
from typing import Optional

import numpy as np
import openai
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Audio parameters ──────────────────────────────────────────────────────────

SAMPLE_RATE   = 16_000          # Hz — matches Reachy Mini SDK output
CHUNK_SAMPLES = 1_600           # 0.1 s per VAD chunk

# Energy threshold for speech onset/offset (RMS in float32 [-1, 1] range)
SPEECH_THRESHOLD = 0.02

# Seconds of consecutive silence before recording ends
SILENCE_DURATION  = 1.2
SILENCE_CHUNKS    = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK_SAMPLES)   # ≈ 12

# Discard recordings shorter than this (spurious noise triggers)
MIN_SPEECH_DURATION = 0.3
MIN_SPEECH_CHUNKS   = int(MIN_SPEECH_DURATION * SAMPLE_RATE / CHUNK_SAMPLES)  # ≈ 3

# Hard timeout waiting for speech
MAX_DURATION = 30.0


class SpeechToText:
    """
    Listen for speech via Reachy Mini SDK or sounddevice, transcribe with
    OpenAI Whisper-1.

    Parameters
    ----------
    reachy : ReachyMini or None
        Live robot object for SDK microphone access.
        Pass None to fall back to sounddevice (simulation / laptop mic).
    api_key : str or None
        OpenAI API key; defaults to OPENAI_API_KEY env var.
    """

    def __init__(self, reachy=None, api_key: Optional[str] = None) -> None:
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise EnvironmentError("OPENAI_API_KEY not set — SpeechToText cannot initialise")
        self._client = openai.OpenAI(api_key=key)
        self._reachy  = reachy

    # ── Public API ────────────────────────────────────────────────────────────

    def listen_and_transcribe(
        self,
        timeout: float = MAX_DURATION,
        stop_event: Optional[threading.Event] = None,
    ) -> Optional[str]:
        """
        Block until speech is detected, record until silence, return transcript.

        Returns the stripped transcript string, or None if timed out,
        stop_event was set, no speech was detected, or transcription failed.
        """
        if self._reachy is not None:
            audio = self._record_sdk(timeout, stop_event)
        else:
            audio = self._record_sounddevice(timeout, stop_event)

        if audio is None:
            return None

        duration = len(audio) / SAMPLE_RATE
        logger.debug("STT: %.1f s captured — sending to Whisper", duration)
        return self._transcribe(audio)

    # ── SDK audio path ────────────────────────────────────────────────────────

    def _record_sdk(
        self,
        timeout: float,
        stop_event: Optional[threading.Event],
    ) -> Optional[np.ndarray]:
        try:
            self._reachy.media.start_recording()
        except Exception:
            logger.exception("media.start_recording() failed")
            return None

        # Brief warmup so the GStreamer pipeline is ready before VAD starts
        time.sleep(0.15)

        try:
            return self._vad_loop(self._sdk_chunk, timeout, stop_event)
        finally:
            try:
                self._reachy.media.stop_recording()
            except Exception:
                logger.warning("media.stop_recording() failed", exc_info=True)

    def _sdk_chunk(self) -> Optional[np.ndarray]:
        """Pull one sample block from the SDK and return mono float32."""
        try:
            data = self._reachy.media.get_audio_sample()
        except Exception:
            logger.debug("get_audio_sample() error", exc_info=True)
            return None
        if data is None or len(data) == 0:
            time.sleep(0.02)
            return None
        # SDK returns (samples, channels) — mix down to mono
        mono = data.mean(axis=1) if data.ndim == 2 else data
        return mono.astype(np.float32)

    # ── sounddevice fallback ──────────────────────────────────────────────────

    def _record_sounddevice(
        self,
        timeout: float,
        stop_event: Optional[threading.Event],
    ) -> Optional[np.ndarray]:
        try:
            import sounddevice as sd  # type: ignore
        except ImportError:
            logger.error("sounddevice not installed — cannot capture microphone input")
            return None

        try:
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=CHUNK_SAMPLES,
            )
            stream.start()
        except Exception:
            logger.exception("Failed to open sounddevice input stream")
            return None

        def chunk_fn() -> Optional[np.ndarray]:
            try:
                data, _ = stream.read(CHUNK_SAMPLES)
                return data[:, 0] if data.ndim == 2 else data.flatten()
            except Exception:
                return None

        try:
            return self._vad_loop(chunk_fn, timeout, stop_event)
        finally:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    # ── Shared VAD loop ───────────────────────────────────────────────────────

    def _vad_loop(
        self,
        chunk_fn,
        timeout: float,
        stop_event: Optional[threading.Event],
    ) -> Optional[np.ndarray]:
        """
        Read audio chunks from *chunk_fn*, apply energy VAD, and return
        a mono float32 array covering the detected utterance, or None.
        """
        chunks:         list[np.ndarray] = []
        speech_chunks   = 0
        silence_chunks  = 0
        speech_started  = False
        deadline        = time.monotonic() + timeout

        while time.monotonic() < deadline:
            if stop_event is not None and stop_event.is_set():
                return None

            chunk = chunk_fn()
            if chunk is None:
                continue

            rms = float(np.sqrt(np.mean(chunk ** 2)))

            if rms >= SPEECH_THRESHOLD:
                if not speech_started:
                    speech_started = True
                    logger.debug("STT: speech onset (RMS=%.4f)", rms)
                    print("  [aufnehmen…]", flush=True)
                silence_chunks = 0
                speech_chunks += 1
                chunks.append(chunk)

            elif speech_started:
                # Append silence so Whisper hears natural sentence endings
                silence_chunks += 1
                chunks.append(chunk)
                if silence_chunks >= SILENCE_CHUNKS:
                    break   # enough trailing silence — utterance complete

            # else: still waiting for speech onset — discard chunk

        if not speech_started or speech_chunks < MIN_SPEECH_CHUNKS:
            logger.debug(
                "STT: discarding (speech_chunks=%d, min=%d)",
                speech_chunks, MIN_SPEECH_CHUNKS,
            )
            return None

        # Keep all speech plus a short tail of silence for natural pacing
        tail = SILENCE_CHUNKS // 2
        keep = max(speech_chunks, len(chunks) - silence_chunks + tail)
        return np.concatenate(chunks[:keep])

    # ── Whisper transcription ─────────────────────────────────────────────────

    def _to_wav_bytes(self, audio: np.ndarray) -> bytes:
        """Encode mono float32 array as a 16-bit 16 kHz WAV byte blob."""
        pcm = (audio * 32767.0).clip(-32768, 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)          # int16 = 2 bytes per sample
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())
        return buf.getvalue()

    def _transcribe(self, audio: np.ndarray) -> Optional[str]:
        """POST WAV to Whisper-1 and return transcript, or None on error."""
        wav = self._to_wav_bytes(audio)
        try:
            result = self._client.audio.transcriptions.create(
                model="whisper-1",
                file=("speech.wav", wav, "audio/wav"),
                response_format="text",
            )
            text = result.strip() if isinstance(result, str) else str(result).strip()
            logger.debug("STT transcript: %r", text[:120])
            return text or None
        except openai.APIError as exc:
            logger.error("Whisper API error: %s", exc)
            return None
        except Exception:
            logger.exception("Whisper transcription failed")
            return None


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.DEBUG, format="%(levelname)-8s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Reachy STT standalone test")
    parser.add_argument("--no-robot", action="store_true", help="Use system mic (sounddevice)")
    parser.add_argument("--loops", type=int, default=3, help="Number of listen/transcribe loops")
    args = parser.parse_args()

    stt = SpeechToText(reachy=None)   # no robot object in standalone mode
    print(f"Speak into the {'system microphone' if args.no_robot else 'Reachy microphone'}.")
    print("Press Ctrl-C to stop.\n")

    stop = threading.Event()
    try:
        for i in range(args.loops):
            print(f"--- Round {i + 1} / {args.loops} --- (waiting for speech…)")
            text = stt.listen_and_transcribe(stop_event=stop)
            if text:
                print(f"Transcript: {text!r}\n")
            else:
                print("(no speech detected or transcription failed)\n")
    except KeyboardInterrupt:
        stop.set()
        print("\nStopped.")
        sys.exit(0)
