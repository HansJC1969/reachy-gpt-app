#!/usr/bin/env python3
"""
Reachy GPT App — main entry point.

Threads:
  • camera_loop    – captures frames, runs face detection every frame
  • tracking_loop  – reads face position and moves neck/body
  • recognition_loop – runs face recognition every N frames
  • conversation_loop – text I/O via stdin (mic optional)

CLI flags:
  --no-robot       Run without a physical Reachy (simulation mode)
  --setup          Initialise the SQLite database and exit
  --add-person "Name"  Register a new face and exit
  --camera INDEX   Camera device index (default: $CAMERA_INDEX or 0)
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
from modules.face_tracking import FaceTracker, FacePosition
from modules.face_recognition_module import FaceRecognitionModule
from modules.memory import init_db, get_or_create_person, save_message, load_recent_messages, build_memory_context

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RECOGNITION_INTERVAL = 15        # run recognition every N frames
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
UNKNOWN_PERSON_NAME = "Stranger"


# ---------------------------------------------------------------------------
# Shared state (protected by locks where needed)
# ---------------------------------------------------------------------------

class SharedState:
    def __init__(self) -> None:
        self.latest_frame: Optional[cv2.Mat] = None
        self.latest_face: Optional[FacePosition] = None
        self.current_person: Optional[str] = None
        self.frame_lock = threading.Lock()
        self.face_lock = threading.Lock()
        self.person_lock = threading.Lock()
        self.stop_event = threading.Event()

        # Queue of (person_name, user_text) for the conversation thread
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

        time.sleep(0.033)  # ~30 fps cap

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
            # Slowly return to centre when no face visible
            tracker.center_head()

        time.sleep(0.05)  # 20 Hz tracking

    logger.info("Tracking thread stopped")


# ---------------------------------------------------------------------------
# Thread: face recognition (every RECOGNITION_INTERVAL frames)
# ---------------------------------------------------------------------------

def recognition_loop(
    state: SharedState,
    recognizer: FaceRecognitionModule,
) -> None:
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
            previous = state.current_person
            if resolved != previous:
                logger.info("Person changed: %s → %s", previous, resolved)
                state.current_person = resolved

    logger.info("Recognition thread stopped")


# ---------------------------------------------------------------------------
# Thread: conversation (stdin → GPT → stdout)
# ---------------------------------------------------------------------------

def conversation_loop(state: SharedState, convo: ConversationManager) -> None:
    logger.info("Conversation thread started — type to talk (Ctrl-C to quit)")
    current_person_id: Optional[int] = None
    current_person_name: Optional[str] = None

    while not state.stop_event.is_set():
        # Detect person change and refresh context
        with state.person_lock:
            person_name = state.current_person or UNKNOWN_PERSON_NAME

        if person_name != current_person_name:
            current_person_name = person_name
            current_person_id = get_or_create_person(person_name)
            memory_ctx = build_memory_context(current_person_id)
            history = load_recent_messages(current_person_id, limit=10)
            convo.set_person(person_name, memory_ctx)
            logger.info("Conversation context refreshed for '%s'", person_name)

        # Non-blocking input check
        try:
            user_input = input(f"[{current_person_name}] You: ").strip()
        except EOFError:
            # stdin closed (e.g. piped input finished)
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

        # Fetch history from DB to include cross-session context
        history = load_recent_messages(current_person_id, limit=10)

        try:
            reply = convo.chat(user_input, history_override=history)
        except Exception:
            logger.exception("GPT error")
            print("Reachy: [error — could not get a response]")
            continue

        print(f"Reachy: {reply}\n")

        # Persist both turns
        save_message(current_person_id, "user", user_input)
        save_message(current_person_id, "assistant", reply)

    logger.info("Conversation thread stopped")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_reachy(ip: str):
    """Import and connect to the robot, returning the SDK object."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Reachy GPT conversational robot")
    parser.add_argument("--no-robot", action="store_true", help="Run without physical Reachy")
    parser.add_argument("--setup", action="store_true", help="Initialise database and exit")
    parser.add_argument("--add-person", metavar="NAME", help="Register a new face and exit")
    parser.add_argument(
        "--camera",
        type=int,
        default=CAMERA_INDEX,
        help="Camera device index",
    )
    args = parser.parse_args()

    # --- one-shot commands ---
    if args.setup:
        init_db()
        print("Database initialised.")
        return

    if args.add_person:
        init_db()
        recognizer = FaceRecognitionModule()
        recognizer.register_from_camera(args.add_person, camera_index=args.camera)
        return

    # --- normal run ---
    init_db()

    reachy = None
    if not args.no_robot:
        reachy_ip = os.environ.get("REACHY_IP", "")
        if not reachy_ip:
            logger.error("REACHY_IP not set. Use --no-robot or set REACHY_IP in .env")
            sys.exit(1)
        reachy = build_reachy(reachy_ip)

    tracker = FaceTracker(reachy=reachy)
    recognizer = FaceRecognitionModule()
    convo = ConversationManager()

    state = SharedState()
    state.current_person = UNKNOWN_PERSON_NAME

    threads = [
        threading.Thread(target=camera_loop,       args=(state, tracker, args.camera),  name="camera",       daemon=True),
        threading.Thread(target=tracking_loop,     args=(state, tracker),                name="tracking",     daemon=True),
        threading.Thread(target=recognition_loop,  args=(state, recognizer),             name="recognition",  daemon=True),
        threading.Thread(target=conversation_loop, args=(state, convo),                  name="conversation", daemon=False),
    ]

    logger.info("Starting all threads…")
    for t in threads:
        t.start()

    try:
        # Wait for the conversation thread (the only non-daemon thread)
        threads[-1].join()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        state.stop_event.set()
        logger.info("Shutting down…")
        if reachy is not None:
            try:
                tracker.center_head()
            except Exception:
                pass


if __name__ == "__main__":
    main()
