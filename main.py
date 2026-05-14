#!/usr/bin/env python3
"""
Reachy GPT App — main entry point.

Threads:
  • camera_loop       – captures frames, runs Haar face detection (~30 fps)
  • tracking_loop     – moves neck/body to follow face (20 Hz)
  • recognition_loop  – identifies person every N frames
  • vision_loop       – GPT-4o scene description every VISION_INTERVAL seconds
  • idle_loop         – triggers MÜDE emotion when no face is seen for a while
  • conversation_loop – stdin → GPT-4o (with web search + vision) → stdout + TTS

CLI flags:
  --no-robot          Run without a physical Reachy (simulation mode)
  --setup             Initialise the SQLite database and exit
  --add-person "Name" Register a new face and exit
  --camera INDEX      Camera device index (default: $CAMERA_INDEX or 0)
  --emotion-test      Play each emotion in sequence and exit
  --no-vision         Disable automatic scene analysis
  --no-search         Disable web search
  --no-speech         Disable text-to-speech output
  --speech-sim        TTS synthesis but no audio playback (for testing)
"""

import argparse
import logging
import os
import queue
import sys
import threading
import time
from typing import Optional

import cv2
from dotenv import load_dotenv

load_dotenv()

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
from modules.speech import SpeechEngine, detect_language, _sounddevice_available
from modules.vision import VisionAnalyzer
from modules.websearch import WebSearcher

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RECOGNITION_INTERVAL = 15       # run face recognition every N frames
VISION_INTERVAL      = 10.0    # seconds between automatic scene analyses
IDLE_TIMEOUT         = 12.0    # seconds without a face before MÜDE animation
CAMERA_INDEX         = int(os.environ.get("CAMERA_INDEX", "0"))
UNKNOWN_PERSON_NAME  = "Stranger"


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

class SharedState:
    def __init__(self) -> None:
        self.latest_frame: Optional[cv2.Mat] = None
        self.latest_face:  Optional[FacePosition] = None
        self.current_person: Optional[str] = None
        self.last_face_seen: float = time.monotonic()

        # Most recent automatic scene description (set by vision_loop)
        self.latest_scene: Optional[str] = None

        self.frame_lock   = threading.Lock()
        self.face_lock    = threading.Lock()
        self.person_lock  = threading.Lock()
        self.stop_event   = threading.Event()

        self.conversation_queue: queue.Queue = queue.Queue()


# ---------------------------------------------------------------------------
# Thread: camera capture
# ---------------------------------------------------------------------------

def camera_loop(state: SharedState, tracker: FaceTracker, camera_idx: int) -> None:
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        logger.error("Cannot open camera %d", camera_idx)
        state.stop_event.set()
        return

    logger.info("Camera thread started (device %d)", camera_idx)
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

    cap.release()
    logger.info("Camera thread stopped")


# ---------------------------------------------------------------------------
# Thread: face tracking
# ---------------------------------------------------------------------------

def tracking_loop(state: SharedState, tracker: FaceTracker) -> None:
    logger.info("Tracking thread started")
    while not state.stop_event.is_set():
        with state.face_lock:
            face = state.latest_face
        if face is not None:
            tracker.update(face)
        else:
            tracker.center_head()
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
# Thread: GPT-4o vision — automatic scene description
# ---------------------------------------------------------------------------

def vision_loop(state: SharedState, vision: VisionAnalyzer) -> None:
    """
    Every VISION_INTERVAL seconds: grab the latest frame, send it to
    GPT-4o vision, and store the description in state.latest_scene.
    The conversation_loop injects this description into the system prompt
    so Reachy has passive scene awareness at all times.
    """
    logger.info("Vision thread started (interval=%.0fs)", VISION_INTERVAL)
    while not state.stop_event.is_set():
        time.sleep(1.0)     # check every second; VisionAnalyzer controls actual interval

        with state.frame_lock:
            frame = state.latest_frame
        if frame is None:
            continue

        desc = vision.analyze_periodic(frame)
        if desc:
            state.latest_scene = desc
            logger.info("Scene: %s", desc[:80])

    logger.info("Vision thread stopped")


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
# Thread: conversation  stdin → GPT → stdout + emotion
# ---------------------------------------------------------------------------

def conversation_loop(
    state: SharedState,
    convo: ConversationManager,
    emotions: EmotionEngine,
    speech: Optional[SpeechEngine] = None,
) -> None:
    logger.info("Conversation thread started — type to talk (Ctrl-C to quit)")
    current_person_id:   Optional[int] = None
    current_person_name: Optional[str] = None

    while not state.stop_event.is_set():
        # Refresh context when recognised person changes
        with state.person_lock:
            person_name = state.current_person or UNKNOWN_PERSON_NAME

        if person_name != current_person_name:
            current_person_name = person_name
            current_person_id = get_or_create_person(person_name)
            memory_ctx = build_memory_context(current_person_id)
            convo.set_person(person_name, memory_ctx)
            logger.info("Context refreshed for '%s'", person_name)
            if current_person_id is not None:
                emotions.play(Emotion.NEUGIER)
                # Greet the newly recognised person aloud
                greeting = f"Hallo{', ' + current_person_name if current_person_name != UNKNOWN_PERSON_NAME else ''}! Schön, dich zu sehen."
                if speech:
                    speech.speak(greeting, interrupt=True)

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
        if user_input.lower() in {"quit", "exit", ":q", "tschüss", "auf wiedersehen"}:
            if speech:
                speech.speak("Tschüss! Bis zum nächsten Mal.", interrupt=True)
                speech.wait_until_done(timeout=5.0)
            state.stop_event.set()
            break

        # Stop ongoing speech — user is already talking
        if speech:
            speech.stop()

        # Pass current frame so vision tool can use it
        with state.frame_lock:
            convo.set_latest_frame(state.latest_frame)

        # Show "thinking" while waiting for GPT
        emotions.play(Emotion.NACHDENKEN)

        history = load_recent_messages(current_person_id, limit=10)
        try:
            reply, emotion = convo.chat_with_emotion(user_input, history_override=history)
        except Exception:
            logger.exception("GPT error")
            emotions.play(Emotion.ANGST)
            error_msg = "Entschuldigung, ich konnte leider keine Antwort bekommen."
            print(f"Reachy: {error_msg}")
            if speech:
                speech.speak(error_msg, interrupt=True)
            continue

        # Start emotion animation and speech simultaneously
        emotions.play(emotion)
        if speech:
            speech.speak(reply, interrupt=True)

        lang = detect_language(reply)
        print(f"Reachy [{emotion.value}][{lang}]: {reply}\n")

        save_message(current_person_id, "user", user_input)
        save_message(current_person_id, "assistant", reply)

    logger.info("Conversation thread stopped")


# ---------------------------------------------------------------------------
# Entry point helpers
# ---------------------------------------------------------------------------

def build_reachy(ip: str):
    try:
        from reachy2_sdk import ReachySDK  # type: ignore
        logger.info("Connecting to Reachy at %s …", ip)
        reachy = ReachySDK(host=ip)
        logger.info("Connected to Reachy")
        return reachy
    except ImportError:
        logger.error("reachy2-sdk not installed — cannot connect to robot")
        sys.exit(1)
    except Exception:
        logger.exception("Failed to connect to Reachy at %s", ip)
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
    parser.add_argument("--no-robot",     action="store_true")
    parser.add_argument("--setup",        action="store_true")
    parser.add_argument("--add-person",   metavar="NAME")
    parser.add_argument("--emotion-test", action="store_true")
    parser.add_argument("--camera",       type=int, default=CAMERA_INDEX)
    parser.add_argument("--no-vision",    action="store_true", help="Disable GPT-4o scene analysis")
    parser.add_argument("--no-search",    action="store_true", help="Disable web search")
    parser.add_argument("--no-speech",    action="store_true", help="Disable text-to-speech output")
    parser.add_argument("--speech-sim",   action="store_true", help="TTS synthesis without audio playback")
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
        reachy_ip = os.environ.get("REACHY_IP", "")
        if not reachy_ip:
            logger.error("REACHY_IP not set. Use --no-robot or set REACHY_IP in .env")
            sys.exit(1)
        reachy = build_reachy(reachy_ip)

    if args.emotion_test:
        run_emotion_test(reachy)
        return

    # ---- normal run --------------------------------------------------------
    init_db()

    # Optional modules
    vision: Optional[VisionAnalyzer] = None
    if not args.no_vision:
        try:
            vision = VisionAnalyzer(interval=VISION_INTERVAL)
            logger.info("Vision module enabled (interval=%.0fs)", VISION_INTERVAL)
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
        sim_mode = args.speech_sim or not _sounddevice_available()
        if sim_mode and not args.speech_sim:
            logger.warning("No audio output device found — TTS in sim mode (synthesis only)")
        try:
            speech = SpeechEngine(sim_mode=sim_mode)
        except Exception:
            logger.warning("TTS disabled (check OPENAI_API_KEY or sounddevice installation)")

    tracker    = FaceTracker(reachy=reachy)
    recognizer = FaceRecognitionModule()
    emotions   = EmotionEngine(reachy=reachy)
    convo      = ConversationManager(vision=vision, searcher=searcher)

    state = SharedState()
    state.current_person = UNKNOWN_PERSON_NAME

    # Startup: play FREUDE animation and speak a greeting simultaneously
    emotions.play(Emotion.FREUDE)
    if speech:
        speech.speak("Hallo! Ich bin bereit. Wie kann ich dir helfen?")

    # Build thread list — vision_loop only started when vision module is active.
    # Use None as target sentinel; the loop below skips those entries.
    thread_specs = [
        ("camera",       True,  camera_loop,                               (state, tracker, args.camera)),
        ("tracking",     True,  tracking_loop,                              (state, tracker)),
        ("recognition",  True,  recognition_loop,                           (state, recognizer)),
        ("vision",       True,  vision_loop if vision is not None else None, (state, vision)),
        ("idle",         True,  idle_loop,                                  (state, emotions)),
        ("conversation", False, conversation_loop,                          (state, convo, emotions, speech)),
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


if __name__ == "__main__":
    main()
