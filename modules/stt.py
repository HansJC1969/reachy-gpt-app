"""
Speech-to-text for Reachy Mini using OpenAI Whisper.

Audio capture
-------------
On robot (reachy is not None):
    reachy.media.start_recording() / get_audio_sample() / stop_recording()
    SDK LOCAL backend — GStreamer pipeline sourced from reachymini_audio_src,
    delivering 16 kHz float32 stereo frames via IPC.

    Voice activity detection uses reachy.media.get_DoA() which returns
    (angle, is_speech_detected).  DoA is treated as an OR signal: speech is
    detected when EITHER the DSP flags it OR the RMS exceeds the threshold.
    This prevents the DoA being a hard gate that blocks all speech.

Simulation (reachy is None):
    sounddevice InputStream from the default system microphone.
    Energy RMS VAD only (no DoA available).

Both paths share _vad_loop():
  1. Discard frames until speech is detected (DoA flag or RMS > threshold).
  2. Record through trailing silence until SILENCE_DURATION seconds of
     quiet have elapsed.
  3. Stop recording after MAX_RECORD_DURATION (10 s) to prevent runaway capture.
  4. Discard if speech content < MIN_SPEECH_DURATION (0.5 s).
  5. Send WAV bytes to OpenAI Whisper-1 with language=WHISPER_LANGUAGE
     ("de" by default) to prevent language-guessing misreads.

Startup test:
    Call stt.mic_selftest() after construction to record 3 s of ambient audio
    and print the RMS level — use this to verify the mic is working and to
    tune STT_THRESHOLD in .env.

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
from typing import Callable, Optional

import numpy as np
import openai
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

logger = logging.getLogger(__name__)

# ── Audio parameters ──────────────────────────────────────────────────────────

SAMPLE_RATE   = 16_000          # Hz — matches Reachy Mini SDK output
CHUNK_SAMPLES = 1_600           # 0.1 s per VAD chunk

# RMS energy threshold for speech onset/offset.
# 0.020 works at ~0.5–1 m conversational distance with Reachy's mic array.
# Raise if background noise triggers false detection; lower if speech is missed.
# Override with STT_THRESHOLD in .env (run mic_selftest() to find your level).
SPEECH_THRESHOLD = float(os.environ.get("STT_THRESHOLD", "0.020"))

# Seconds of consecutive silence that ends a recording
SILENCE_DURATION  = 0.8
SILENCE_CHUNKS    = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK_SAMPLES)   # = 8

# Minimum speech content before Whisper is called (discard very short bursts)
MIN_SPEECH_DURATION = 0.5
MIN_SPEECH_CHUNKS   = int(MIN_SPEECH_DURATION * SAMPLE_RATE / CHUNK_SAMPLES)  # = 5

# Maximum recording time after speech onset (prevents runaway capture)
MAX_RECORD_DURATION = 10.0
MAX_RECORD_CHUNKS   = int(MAX_RECORD_DURATION * SAMPLE_RATE / CHUNK_SAMPLES)  # = 100

# Whisper language hint — "de" stops Whisper toggling between German and
# English phonemes, fixing misreadings of German proper nouns.
WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "de")

# Hard timeout waiting for speech onset
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

    def mic_selftest(self, duration: float = 3.0) -> float:
        """
        Record *duration* seconds of audio and print the RMS level.

        Use at startup to verify the microphone is working and to help tune
        STT_THRESHOLD in .env.  Returns the measured RMS (0.0 on failure).
        """
        print(f"\n🎙️  Mikrofon-Selbsttest ({duration:.0f}s) — bitte sprechen …", flush=True)
        chunks: list[np.ndarray] = []

        if self._reachy is not None:
            try:
                self._reachy.media.start_recording()
                # Allow the reachymini_audio_src GStreamer pipeline to stabilise
                time.sleep(0.3)
                deadline = time.monotonic() + duration
                while time.monotonic() < deadline:
                    chunk = self._sdk_chunk()
                    if chunk is not None and len(chunk) > 0:
                        chunks.append(chunk)
            except Exception:
                logger.exception("Mikrofon-Selbsttest fehlgeschlagen")
            finally:
                try:
                    self._reachy.media.stop_recording()
                except Exception:
                    pass
        else:
            try:
                import sounddevice as sd  # type: ignore
                data = sd.rec(
                    int(duration * SAMPLE_RATE),
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="float32",
                )
                sd.wait()
                chunks = [data[:, 0]]
            except Exception:
                logger.exception("Mikrofon-Selbsttest fehlgeschlagen (sounddevice)")

        if not chunks:
            print("  ❌ Kein Audio empfangen — Mikrofon prüfen!", flush=True)
            return 0.0

        audio = np.concatenate(chunks)
        rms   = float(np.sqrt(np.mean(audio ** 2)))
        peak  = float(np.max(np.abs(audio)))

        bar_filled = int(min(rms / 0.1, 1.0) * 20)
        bar = "█" * bar_filled + "░" * (20 - bar_filled)

        if rms >= SPEECH_THRESHOLD:
            status = "✅ OK"
        else:
            status = f"⚠️  SEHR LEISE — STT_THRESHOLD ggf. auf {rms * 0.7:.4f} senken"

        print(f"  RMS: {rms:.4f}  Peak: {peak:.4f}  [{bar}]  {status}", flush=True)
        print(f"  Schwellwert aktuell: STT_THRESHOLD={SPEECH_THRESHOLD:.4f}", flush=True)
        return rms

    def listen_and_transcribe(
        self,
        timeout: float = MAX_DURATION,
        stop_event: Optional[threading.Event] = None,
        speaking_guard=None,
    ) -> Optional[str]:
        """
        Block until speech is detected, record until silence, return transcript.

        Parameters
        ----------
        speaking_guard : SpeechEngine or any object with .is_active property
            When provided, the microphone is muted while the guard reports
            active TTS playback.  After TTS finishes, a 1-second settle delay
            is applied before recording starts, preventing echo.

        Returns the stripped transcript string, or None if timed out,
        stop_event was set, no speech was detected, or transcription failed.
        """
        # ── Echo cancellation gate ────────────────────────────────────────────
        if speaking_guard is not None:
            while getattr(speaking_guard, "is_active", False):
                if stop_event is not None and stop_event.is_set():
                    return None
                time.sleep(0.05)
            # Allow speaker cone and GStreamer buffer to fully drain
            time.sleep(1.0)

        if self._reachy is not None:
            audio = self._record_sdk(timeout, stop_event, speaking_guard)
        else:
            audio = self._record_sounddevice(timeout, stop_event, speaking_guard)

        if audio is None:
            return None

        duration = len(audio) / SAMPLE_RATE
        logger.info("STT: %.1f s aufgenommen — sende an Whisper (language=%s)",
                    duration, WHISPER_LANGUAGE)
        return self._transcribe(audio)

    # ── SDK audio path ────────────────────────────────────────────────────────

    def _record_sdk(
        self,
        timeout: float,
        stop_event: Optional[threading.Event],
        speaking_guard=None,
    ) -> Optional[np.ndarray]:
        try:
            self._reachy.media.start_recording()
        except Exception:
            logger.exception("media.start_recording() failed")
            return None

        # Allow the reachymini_audio_src GStreamer pipeline to stabilise
        time.sleep(0.3)

        try:
            return self._vad_loop(
                self._sdk_chunk,
                timeout,
                stop_event,
                is_speech_fn=self._sdk_is_speech,
                speaking_guard=speaking_guard,
            )
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

    def _sdk_is_speech(self) -> bool:
        """
        Use the SDK's hardware DSP speech detector via get_DoA().
        Falls back to False on any error so recording keeps running.
        """
        try:
            _angle, is_speech = self._reachy.media.get_DoA()
            return bool(is_speech)
        except Exception:
            logger.debug("get_DoA() error", exc_info=True)
            return False

    # ── sounddevice fallback ──────────────────────────────────────────────────

    def _record_sounddevice(
        self,
        timeout: float,
        stop_event: Optional[threading.Event],
        speaking_guard=None,
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
            return self._vad_loop(chunk_fn, timeout, stop_event,
                                  is_speech_fn=None, speaking_guard=speaking_guard)
        finally:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    # ── Shared VAD loop ───────────────────────────────────────────────────────

    def _vad_loop(
        self,
        chunk_fn: Callable[[], Optional[np.ndarray]],
        timeout: float,
        stop_event: Optional[threading.Event],
        is_speech_fn: Optional[Callable[[], bool]] = None,
        speaking_guard=None,
    ) -> Optional[np.ndarray]:
        """
        Read audio chunks from *chunk_fn*, apply VAD, and return a mono
        float32 array covering the detected utterance, or None.

        Speech detection (SDK path):
          DoA flag OR RMS >= threshold — either signal is sufficient.
          Using OR instead of AND prevents a conservative DSP from blocking
          all speech when the mic array can hear the user fine.

        Speech detection (sounddevice path):
          RMS >= threshold only.

        Echo cancellation:
          When speaking_guard.is_active is True, all chunks are discarded and
          VAD state is reset — prevents Reachy's own voice from triggering STT.
        """
        chunks:        list[np.ndarray] = []
        speech_chunks  = 0
        silence_chunks = 0
        speech_started = False
        deadline       = time.monotonic() + timeout

        print("🎤 Ich höre zu...", flush=True)

        while time.monotonic() < deadline:
            if stop_event is not None and stop_event.is_set():
                return None

            # ── Echo gate: discard all audio while Reachy is speaking ────────
            if speaking_guard is not None and getattr(speaking_guard, "is_active", False):
                # Reset VAD state so any partial detection is discarded
                chunks.clear()
                speech_chunks  = 0
                silence_chunks = 0
                speech_started = False
                time.sleep(0.05)
                continue

            chunk = chunk_fn()
            if chunk is None or len(chunk) == 0:
                continue

            rms = float(np.sqrt(np.mean(chunk ** 2)))

            if is_speech_fn is not None:
                # SDK path: DoA OR energy — either signal is enough.
                is_speech = is_speech_fn() or rms >= SPEECH_THRESHOLD
            else:
                is_speech = rms >= SPEECH_THRESHOLD

            if is_speech:
                if not speech_started:
                    speech_started = True
                    logger.info("STT: Sprache erkannt (rms=%.4f, threshold=%.4f)",
                                rms, SPEECH_THRESHOLD)
                    print("✅ Sprache erkannt!", flush=True)
                silence_chunks = 0
                speech_chunks += 1
                chunks.append(chunk)

                # Hard cap: stop recording after MAX_RECORD_DURATION
                if speech_chunks >= MAX_RECORD_CHUNKS:
                    logger.info("STT: max Aufnahmedauer erreicht (%.0fs)", MAX_RECORD_DURATION)
                    print(f"  ⏱️  Max. {MAX_RECORD_DURATION:.0f}s — sende an Whisper…", flush=True)
                    break

            elif speech_started:
                silence_chunks += 1
                chunks.append(chunk)
                if silence_chunks >= SILENCE_CHUNKS:
                    speech_secs = speech_chunks * CHUNK_SAMPLES / SAMPLE_RATE
                    print(f"  🔄 Aufnahme beendet ({speech_secs:.1f}s) — transkribiere…",
                          flush=True)
                    break

            # else: still waiting for speech onset — discard chunk

        if not speech_started or speech_chunks < MIN_SPEECH_CHUNKS:
            logger.debug(
                "STT: verworfen (speech_chunks=%d, min=%d, threshold=%.4f)",
                speech_chunks, MIN_SPEECH_CHUNKS, SPEECH_THRESHOLD,
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
                language=WHISPER_LANGUAGE,
                prompt="Reachy, Hans",
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

    stt = SpeechToText(reachy=None)
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
