#!/usr/bin/env python3
"""
Reachy GPT App — main entry point.

Threads:
  • camera_loop       – captures frames, runs Haar face detection
  • tracking_loop     – moves neck/body to follow face (20 Hz)
  • recognition_loop  – identifies person every N frames
  • idle_loop         – triggers MÜDE emotion when no face is seen for a while
  • conversation_loop – text I/O: stdin → GPT-4o → stdout + emotion animation

CLI flags:
  --no-robot          Run without a physical Reachy (simulation mode)
  --setup             Initialise the SQLite database and exit
  --add-person "Name" Register a new face and exit
  --camera INDEX      Camera device index (default: $CAMERA_INDEX or 0)
  --emotion-test      Play each emotion in sequence and exit
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RECOGNITION_INTERVAL = 15       # run face recognition every N frames
IDLE_TIMEOUT = 12.0             # seconds without a face before MÜDE animation
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
UNKNOWN_PERSON_NAME = "Stranger"


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

class SharedState:
    def __init__(self) -> None:
        self.latest_frame: Optional[cv2.Mat] = None
        self.latest_face: Optional[FacePosition] = None
        self.current_person: Optional[str] = None
        self.last_face_seen: float = time.monotonic()

        self.frame_lock  = threading.Lock()
        self.face_lock   = threading.Lock()
        self.person_lock = threading.Lock()
        self.stop_event  = threading.Event()

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
# Thread: face tracking (neck / body movement)
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
# Thread: idle emotion — triggers MÜDE when nobody is around
# ---------------------------------------------------------------------------

def idle_loop(state: SharedState, emotions: EmotionEngine) -> None:
    logger.info("Idle loop started")
    was_idle = False

    while not state.stop_event.is_set():
        time.sleep(1.0)
        idle_seconds = time.monotonic() - state.last_face_seen

        if idle_seconds > IDLE_TIMEOUT:
            if not was_idle:
                logger.info("No face for %.0fs — expressing MÜDE", idle_seconds)
                emotions.play(Emotion.MÜDE)
                was_idle = True
        else:
            if was_idle:
                # Face reappeared
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
) -> None:
    logger.info("Conversation thread started — type to talk (Ctrl-C to quit)")
    current_person_id: Optional[int] = None
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
            logger.info("Conversation context refreshed for '%s'", person_name)
            # Greet with curiosity if it's not the first load
            if current_person_id is not None:
                emotions.play(Emotion.NEUGIER)

        try:
            user_input = input(f"[{current_person_name}] You: ").strip()
        except EOFError:
            state.stop_event.set()
            break
        except KeyboardInterrupt:
            state.stop_event.set()
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", ":q"}:
            state.stop_event.set()
            break

        # Show "thinking" immediately while waiting for GPT
        emotions.play(Emotion.NACHDENKEN)

        history = load_recent_messages(current_person_id, limit=10)
        try:
            reply, emotion = convo.chat_with_emotion(user_input, history_override=history)
        except Exception:
            logger.exception("GPT error")
            emotions.play(Emotion.ANGST)
            print("Reachy: [error — could not get a response]")
            continue

        # Express the emotion GPT chose
        emotions.play(emotion)
        print(f"Reachy [{emotion.value}]: {reply}\n")

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
    """Play every emotion in sequence and exit."""
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
    parser.add_argument("--no-robot",     action="store_true", help="Run without physical Reachy")
    parser.add_argument("--setup",        action="store_true", help="Initialise database and exit")
    parser.add_argument("--add-person",   metavar="NAME",      help="Register a new face and exit")
    parser.add_argument("--emotion-test", action="store_true", help="Play all emotions and exit")
    parser.add_argument("--camera",       type=int, default=CAMERA_INDEX)
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

    tracker    = FaceTracker(reachy=reachy)
    recognizer = FaceRecognitionModule()
    convo      = ConversationManager()
    emotions   = EmotionEngine(reachy=reachy)

    state = SharedState()
    state.current_person = UNKNOWN_PERSON_NAME

    # Startup greeting
    emotions.play(Emotion.FREUDE)

    threads = [
        threading.Thread(
            target=camera_loop,
            args=(state, tracker, args.camera),
            name="camera", daemon=True,
        ),
        threading.Thread(
            target=tracking_loop,
            args=(state, tracker),
            name="tracking", daemon=True,
        ),
        threading.Thread(
            target=recognition_loop,
            args=(state, recognizer),
            name="recognition", daemon=True,
        ),
        threading.Thread(
            target=idle_loop,
            args=(state, emotions),
            name="idle", daemon=True,
        ),
        # conversation is non-daemon: controls app lifetime
        threading.Thread(
            target=conversation_loop,
            args=(state, convo, emotions),
            name="conversation", daemon=False,
        ),
    ]

    logger.info("Starting all threads…")
    for t in threads:
        t.start()

    try:
        threads[-1].join()      # wait for conversation thread
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        state.stop_event.set()
        emotions.stop()
        logger.info("Shutting down…")
        if reachy is not None:
            try:
                tracker.center_head()
            except Exception:
                pass


if __name__ == "__main__":
    main()
