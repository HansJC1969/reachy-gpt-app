#!/usr/bin/env python3
"""
Reachy GPT App — main entry point.

Threads:
  • camera_loop       – captures frames, runs Haar face detection (~30 fps)
  • tracking_loop     – moves neck/body to follow face (20 Hz)
  • recognition_loop  – identifies person every N frames
  • idle_loop         – triggers MÜDE emotion when no face is seen for a while
  • conversation_loop – stdin → GPT-4o (with web search + vision) → stdout + TTS

CLI flags:
  --no-robot          Run without a physical Reachy (simulation mode)
  --setup             Initialise the SQLite database and exit
  --add-person "Name" Register a new face and exit
  --camera INDEX      OpenCV device index used only in --no-robot sim mode (default: $CAMERA_INDEX or 0)
  --emotion-test      Play each emotion in sequence and exit
  --no-vision         Disable GPT-4o vision (on-demand "Was siehst du?" tool)
  --no-search         Disable web search
  --no-speech         Disable text-to-speech output
  --speech-sim        TTS synthesis but no audio playback (for testing)
  --text-input        Use keyboard input instead of microphone (development only)
"""

import argparse
import logging
import os
import re
import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# ---------------------------------------------------------------------------
# Local modules
# ---------------------------------------------------------------------------
from modules.conversation import ConversationManager
from modules.emotions import Emotion, EmotionEngine
from modules.face_tracking import FaceTracker, FacePosition
from modules.face_recognition_module import FaceRecognitionModule
from modules.memory import (
    init_db,
    get_or_create_person,
    save_message,
    load_recent_messages,
    build_memory_context,
)
from modules.profiles import list_profiles, load_profile
from modules.speech import SpeechEngine, AVAILABLE_VOICES, DEFAULT_VOICE, detect_language, sounddevice_available
from modules.stt import SpeechToText
from modules.vision import VisionAnalyzer
from modules.websearch import WebSearcher

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RECOGNITION_INTERVAL = 15       # run face recognition every N frames
VISION_INTERVAL      = 30.0    # interval kept for VisionAnalyzer constructor (unused for passive loop)
IDLE_TIMEOUT         = 12.0    # seconds without a face before MÜDE animation
CAMERA_INDEX         = int(os.environ.get("CAMERA_INDEX", "0"))
UNKNOWN_PERSON_NAME  = "Stranger"

# Disable head/body tracking until motors are separately tested and calibrated.
# Set True only after verifying joint limits and movement feel on the physical robot.
TRACKING_ENABLED     = False


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

class SharedState:
    def __init__(self) -> None:
        self.latest_frame: Optional[cv2.Mat] = None
        self.latest_face:  Optional[FacePosition] = None
        self.current_person: Optional[str] = None
        self.last_face_seen: float = time.monotonic()



        self.frame_lock   = threading.Lock()
        self.face_lock    = threading.Lock()
        self.person_lock  = threading.Lock()
        self.stop_event   = threading.Event()


# ---------------------------------------------------------------------------
# Thread: camera capture
# ---------------------------------------------------------------------------

def camera_loop(
    state: SharedState,
    tracker: FaceTracker,
    camera_idx: int,
    reachy,
) -> None:
    """
    When running on the robot (reachy is not None) frames are read via
    reachy.media.get_frame(), which uses the SDK's local IPC backend —
    the correct path when SSHed into Reachy Mini.

    In --no-robot simulation mode (reachy is None) OpenCV VideoCapture is
    used with the device index supplied by --camera.
    """
    if reachy is not None:
        logger.info("Camera thread started (Reachy Mini SDK media.get_frame())")
        while not state.stop_event.is_set():
            try:
                frame = reachy.media.get_frame()
            except Exception:
                logger.warning("media.get_frame() failed; retrying…", exc_info=True)
                time.sleep(0.05)
                continue

            if frame is None:
                time.sleep(0.033)
                continue

            # SDK returns RGB; convert to BGR for OpenCV-based processing
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            with state.frame_lock:
                state.latest_frame = frame.copy()

            face = tracker.detect_face(frame)
            with state.face_lock:
                state.latest_face = face
                if face is not None:
                    state.last_face_seen = time.monotonic()

        logger.info("Camera thread stopped")

    else:
        cap = cv2.VideoCapture(camera_idx)
        if not cap.isOpened():
            logger.error("Cannot open camera device %d", camera_idx)
            state.stop_event.set()
            return

        logger.info("Camera thread started (OpenCV device %d)", camera_idx)
        try:
            while not state.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Camera read failed; retrying…")
                    time.sleep(0.05)
                    continue

                with state.frame_lock:
                    state.latest_frame = frame.copy()

                face = tracker.detect_face(frame)
                with state.face_lock:
                    state.latest_face = face
                    if face is not None:
                        state.last_face_seen = time.monotonic()

                time.sleep(0.033)   # ~30 fps
        finally:
            cap.release()
            logger.info("Camera thread stopped")


# ---------------------------------------------------------------------------
# Thread: face tracking
# ---------------------------------------------------------------------------

def tracking_loop(state: SharedState, tracker: FaceTracker) -> None:
    logger.info("Tracking thread started")
    was_tracking = False
    while not state.stop_event.is_set():
        with state.face_lock:
            face = state.latest_face
        if face is not None:
            tracker.update(face)
            was_tracking = True
        elif was_tracking:
            # Face just lost — center once; don't repeat at 20 Hz
            tracker.center_head()
            was_tracking = False
        time.sleep(0.05)    # 20 Hz
    logger.info("Tracking thread stopped")


# ---------------------------------------------------------------------------
# Thread: face recognition
# ---------------------------------------------------------------------------

def recognition_loop(state: SharedState, recognizer: FaceRecognitionModule) -> None:
    logger.info("Recognition thread started")
    frame_count = 0
    while not state.stop_event.is_set():
        time.sleep(0.1)
        frame_count += 1
        if frame_count % RECOGNITION_INTERVAL != 0:
            continue

        with state.frame_lock:
            frame = state.latest_frame
        if frame is None:
            continue

        try:
            name = recognizer.identify(frame)
        except Exception:
            logger.exception("Recognition error")
            continue

        resolved = name or UNKNOWN_PERSON_NAME
        with state.person_lock:
            if resolved != state.current_person:
                logger.info("Person changed: %s → %s", state.current_person, resolved)
                state.current_person = resolved

    logger.info("Recognition thread stopped")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Thread: idle emotion
# ---------------------------------------------------------------------------

def idle_loop(state: SharedState, emotions: EmotionEngine) -> None:
    logger.info("Idle loop started")
    was_idle = False
    while not state.stop_event.is_set():
        time.sleep(1.0)
        idle_sec = time.monotonic() - state.last_face_seen
        if idle_sec > IDLE_TIMEOUT:
            if not was_idle:
                logger.info("No face for %.0fs — expressing MÜDE", idle_sec)
                emotions.play(Emotion.MÜDE)
                was_idle = True
        else:
            if was_idle:
                emotions.play(Emotion.NEUGIER)
                was_idle = False
    logger.info("Idle loop stopped")


# ---------------------------------------------------------------------------
# Stop-command detection
# ---------------------------------------------------------------------------

_STOP_RE = re.compile(
    r"\b(stop|stopp|halt|quiet|ruhig|schweig(?:en)?|aufhören|aufhör)\b",
    re.IGNORECASE,
)


def _is_stop_command(text: str) -> bool:
    """Return True when the transcript is a stop/quiet command."""
    return bool(_STOP_RE.search(text.strip()))


# ---------------------------------------------------------------------------
# Thread: conversation  stdin → GPT → stdout + emotion
# ---------------------------------------------------------------------------

def conversation_loop(
    state: SharedState,
    convo: ConversationManager,
    emotions: EmotionEngine,
    speech: Optional[SpeechEngine] = None,
    stt: Optional[SpeechToText] = None,
) -> None:
    input_mode = "microphone (Whisper)" if stt is not None else "keyboard (--text-input)"
    logger.info("Conversation thread started — input=%s", input_mode)
    current_person_id:   Optional[int] = None
    current_person_name: Optional[str] = None

    while not state.stop_event.is_set():
        # Refresh context when recognised person changes
        with state.person_lock:
            person_name = state.current_person or UNKNOWN_PERSON_NAME

        if person_name != current_person_name:
            current_person_name = person_name
            try:
                current_person_id = get_or_create_person(person_name)
                memory_ctx = build_memory_context(current_person_id)
            except Exception:
                logger.exception("Failed to load memory context for '%s'", person_name)
                current_person_id = None
                memory_ctx = ""
            convo.set_person(person_name, memory_ctx)
            logger.info("Context refreshed for '%s'", person_name)
            emotions.play(Emotion.NEUGIER)
            greeting = f"Hallo{', ' + current_person_name if current_person_name != UNKNOWN_PERSON_NAME else ''}! Schön, dich zu sehen."
            if speech:
                speech.speak(greeting, interrupt=True)

        if stt is not None:
            # Wait for any ongoing TTS to finish so the mic doesn't capture
            # Reachy's own voice, then give the speaker a moment to settle.
            if speech:
                speech.wait_until_done(timeout=60.0)
                time.sleep(0.4)

            print(f"[{current_person_name}] Sprechen… (Stille zum Beenden)", flush=True)
            try:
                user_input = stt.listen_and_transcribe(
                    timeout=30.0, stop_event=state.stop_event
                )
            except KeyboardInterrupt:
                state.stop_event.set()
                break

            if state.stop_event.is_set():
                break
            if user_input is None:
                logger.debug("STT: no speech detected — listening again")
                continue

            print(f"[{current_person_name}] Du: {user_input}", flush=True)
        else:
            try:
                user_input = input(f"[{current_person_name}] Du: ").strip()
            except EOFError:
                state.stop_event.set()
                break
            except KeyboardInterrupt:
                state.stop_event.set()
                break

        if not user_input:
            continue

        # Stop command — interrupt TTS immediately and listen again
        if _is_stop_command(user_input):
            logger.info("Stop command: %r", user_input)
            if speech:
                speech.stop()
                speech.speak("OK.", interrupt=True)
            emotions.play(Emotion.NEUTRAL)
            continue

        # Sleep command — play MÜDE animation and keep listening
        if user_input.lower() in {"schlafe", "schlaf", "sleep"}:
            logger.info("Sleep command: %r", user_input)
            if speech:
                speech.stop()
            emotions.play(Emotion.MÜDE, block=True)
            continue

        if user_input.lower() in {"quit", "exit", ":q", "tschüss", "auf wiedersehen"}:
            if speech:
                speech.speak("Tschüss! Bis zum nächsten Mal.", interrupt=True)
                speech.wait_until_done(timeout=8.0)
            state.stop_event.set()
            break

        # If user started speaking while Reachy was talking, interrupt TTS
        if speech:
            speech.stop()

        # Pass current frame so vision tool can use it
        with state.frame_lock:
            convo.set_latest_frame(state.latest_frame)

        # Show "thinking" while waiting for GPT
        emotions.play(Emotion.NACHDENKEN)

        history = load_recent_messages(current_person_id, limit=10) if current_person_id is not None else []
        try:
            # Stream sentences to TTS as they arrive — first sentence plays while
            # GPT is still generating the rest, cutting perceived latency.
            for sentence in convo.stream_reply_sentences(user_input, history_override=history):
                if speech:
                    speech.speak(sentence)   # queued, not interrupt — preserves order
                print(f"  ↳ {sentence}", flush=True)
        except Exception:
            logger.exception("GPT error")
            emotions.play(Emotion.ANGST)
            error_msg = "Entschuldigung, ich konnte leider keine Antwort bekommen."
            print(f"Reachy: {error_msg}")
            if speech:
                speech.speak(error_msg, interrupt=True)
            continue

        emotion = convo.last_emotion
        reply   = convo.last_reply
        emotions.play(emotion)
        lang = detect_language(reply)
        print(f"Reachy [{emotion.value}][{lang}]: {reply}\n")

        if current_person_id is not None:
            try:
                save_message(current_person_id, "user", user_input)
                save_message(current_person_id, "assistant", reply)
            except Exception:
                logger.exception("Failed to save messages for person_id=%d", current_person_id)

    logger.info("Conversation thread stopped")


# ---------------------------------------------------------------------------
# Entry point helpers
# ---------------------------------------------------------------------------

def build_reachy():
    try:
        from reachy_mini import ReachyMini  # type: ignore
        logger.info("Connecting to Reachy Mini…")
        # Default media backend: SDK manages the camera via GStreamer IPC.
        # Frames are read with reachy.media.get_frame() in camera_loop.
        # sounddevice (TTS output) uses the speaker independently and does
        # not conflict with the SDK's microphone/camera management.
        reachy = ReachyMini()
        reachy.__enter__()
        logger.info("Connected to Reachy Mini")
        return reachy
    except ImportError:
        logger.error("reachy-mini not installed — cannot connect to robot")
        sys.exit(1)
    except Exception:
        logger.exception("Failed to connect to Reachy Mini")
        sys.exit(1)


def run_emotion_test(reachy) -> None:
    engine = EmotionEngine(reachy=reachy)
    print("Emotion test — playing all animations:\n")
    for emo in Emotion:
        print(f"  ▶ {emo.value}")
        engine.play(emo, block=True)
        time.sleep(0.4)
    print("\nDone.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Reachy GPT conversational robot")
    profiles = list_profiles()
    parser.add_argument("--no-robot",     action="store_true")
    parser.add_argument("--setup",        action="store_true")
    parser.add_argument("--add-person",   metavar="NAME")
    parser.add_argument("--emotion-test", action="store_true")
    parser.add_argument("--camera",       type=int, default=CAMERA_INDEX)
    parser.add_argument("--no-vision",    action="store_true", help="Disable GPT-4o scene analysis")
    parser.add_argument("--no-search",    action="store_true", help="Disable web search")
    parser.add_argument("--no-speech",    action="store_true", help="Disable text-to-speech output")
    parser.add_argument("--speech-sim",   action="store_true", help="TTS synthesis without audio playback")
    parser.add_argument("--text-input",   action="store_true", help="Use keyboard instead of microphone")
    parser.add_argument(
        "--profile",
        default="default",
        choices=profiles or ["default"],
        metavar="PROFILE",
        help=f"Personality profile (available: {', '.join(profiles) or 'default'})",
    )
    parser.add_argument(
        "--voice",
        default=None,
        choices=sorted(AVAILABLE_VOICES),
        help="Override TTS voice. onyx-de (male German, default) and nova-de (female German) "
             "use gpt-4o-mini-tts with native German pronunciation instructions.",
    )
    args = parser.parse_args()

    # ---- one-shot commands -------------------------------------------------
    if args.setup:
        init_db()
        print("Database initialised.")
        return

    if args.add_person:
        init_db()
        FaceRecognitionModule().register_from_camera(args.add_person, camera_index=args.camera)
        return

    reachy = None
    if not args.no_robot:
        reachy = build_reachy()

    if args.emotion_test:
        try:
            run_emotion_test(reachy)
        finally:
            if reachy is not None:
                try:
                    reachy.__exit__(None, None, None)
                except Exception:
                    pass
        return

    # ---- normal run --------------------------------------------------------
    init_db()

    # Load personality profile
    profile_instructions, profile_voice = load_profile(args.profile)
    active_voice = args.voice or profile_voice  # CLI --voice overrides profile
    logger.info(
        "Profile: %s  |  voice: %s",
        args.profile,
        active_voice or "(default)",
    )

    # Optional modules
    vision: Optional[VisionAnalyzer] = None
    if not args.no_vision:
        try:
            vision = VisionAnalyzer(interval=VISION_INTERVAL)
            logger.info("Vision module enabled (on-demand only — triggered by 'Was siehst du?')")
        except Exception:
            logger.warning("Vision module disabled (check OPENAI_API_KEY)")

    searcher: Optional[WebSearcher] = None
    if not args.no_search:
        try:
            s = WebSearcher()
            if s.backend != "none":
                searcher = s
                logger.info("Web search enabled (backend=%s)", s.backend)
            else:
                logger.warning("No search backend available — install duckduckgo-search or set TAVILY_API_KEY")
        except Exception:
            logger.warning("Web search module disabled")

    # Text-to-speech
    speech: Optional[SpeechEngine] = None
    if not args.no_speech:
        if reachy is not None:
            # On-robot: SDK GStreamer backend owns the audio hardware.
            # sounddevice cannot reach the speaker; use push_audio_sample instead.
            sim_mode = args.speech_sim
        else:
            sim_mode = args.speech_sim or not sounddevice_available()
            if sim_mode and not args.speech_sim:
                logger.warning("No audio output device found — TTS in sim mode (synthesis only)")
        try:
            speech = SpeechEngine(
                voice=active_voice or DEFAULT_VOICE,
                sim_mode=sim_mode,
                reachy=reachy,
            )
        except Exception:
            logger.warning("TTS disabled (check OPENAI_API_KEY or sounddevice installation)")

    # Speech-to-text
    stt: Optional[SpeechToText] = None
    if not args.text_input:
        try:
            stt = SpeechToText(reachy=reachy)
            logger.info("STT enabled (%s)", "Reachy mic" if reachy is not None else "system mic")
            stt.mic_selftest()
        except Exception:
            logger.warning("STT disabled (check OPENAI_API_KEY or sounddevice installation)")

    tracker    = FaceTracker(reachy=reachy)
    recognizer = FaceRecognitionModule()
    emotions   = EmotionEngine(reachy=reachy)

    # Motion callables exposed as GPT tools
    def _move_head_fn(direction: str) -> None:
        emotions.move_head(direction)

    def _dance_fn() -> None:
        emotions.play(Emotion.TANZEN)

    convo = ConversationManager(
        vision=vision,
        searcher=searcher,
        move_head_fn=_move_head_fn,
        dance_fn=_dance_fn,
    )
    convo.set_profile(profile_instructions)

    state = SharedState()
    state.current_person = UNKNOWN_PERSON_NAME

    # Startup: play FREUDE animation and speak a greeting simultaneously
    emotions.play(Emotion.FREUDE)
    if speech:
        greeting = "Hallo! Ich bin bereit. Wie kann ich dir helfen?"
        if args.profile and args.profile != "default":
            logger.info("Active personality profile: %s", args.profile)
        speech.speak(greeting)

    # Build thread list.
    # Use None as target sentinel; the loop below skips those entries.
    thread_specs = [
        ("camera",       True,  camera_loop,                               (state, tracker, args.camera, reachy)),
        # tracking_loop is disabled until motors are tested and calibrated
        ("tracking",     True,  tracking_loop if TRACKING_ENABLED else None, (state, tracker)),
        ("recognition",  True,  recognition_loop,                           (state, recognizer)),
        ("idle",         True,  idle_loop,                                  (state, emotions)),
        ("conversation", False, conversation_loop,                          (state, convo, emotions, speech, stt)),
    ]

    threads = []
    for name, is_daemon, target, t_args in thread_specs:
        if target is None:
            continue
        t = threading.Thread(target=target, args=t_args, name=name, daemon=is_daemon)
        threads.append(t)

    logger.info("Starting %d threads…", len(threads))
    for t in threads:
        t.start()

    # Show startup joy — Reachy wakes up happy
    emotions.play(Emotion.FREUDE)

    try:
        # Wait for the conversation thread (the only non-daemon thread)
        next(t for t in threads if t.name == "conversation").join()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        state.stop_event.set()
        emotions.stop()
        if speech:
            speech.shutdown()
        logger.info("Shutting down…")
        if reachy is not None:
            try:
                tracker.center_head()
            except Exception:
                logger.warning("Could not centre head on shutdown", exc_info=True)
            try:
                reachy.__exit__(None, None, None)
            except Exception:
                logger.warning("Error disconnecting from Reachy Mini", exc_info=True)


if __name__ == "__main__":
    main()
