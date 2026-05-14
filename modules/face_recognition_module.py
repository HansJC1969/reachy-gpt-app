"""
Face recognition: load/save encodings with pickle, identify known persons,
register new persons, SQLite integration via modules.memory.

Standalone registration:
    python -m modules.face_recognition_module --add-person "Alice"
"""

import logging
import os
import pickle
import time
from pathlib import Path
from typing import Optional

import cv2
import face_recognition
import numpy as np

from modules.memory import get_or_create_person, init_db

logger = logging.getLogger(__name__)

ENCODINGS_PATH = Path(os.environ.get("ENCODINGS_PATH", "face_encodings.pkl"))
CONFIDENCE_THRESHOLD = float(os.environ.get("FACE_CONFIDENCE_THRESHOLD", "0.5"))

# How many frames to sample when registering a new face
REGISTRATION_SAMPLES = 5


class FaceRecognitionModule:
    def __init__(self) -> None:
        # {name: list[np.ndarray]}
        self._encodings: dict[str, list] = {}
        self._load_encodings()

    # ------------------------------------------------------------------
    # Encoding persistence
    # ------------------------------------------------------------------

    def _load_encodings(self) -> None:
        if ENCODINGS_PATH.exists():
            with open(ENCODINGS_PATH, "rb") as f:
                self._encodings = pickle.load(f)
            logger.info(
                "Loaded encodings for %d person(s): %s",
                len(self._encodings),
                list(self._encodings.keys()),
            )
        else:
            logger.info("No existing encodings file found; starting fresh")

    def _save_encodings(self) -> None:
        with open(ENCODINGS_PATH, "wb") as f:
            pickle.dump(self._encodings, f)
        logger.info("Encodings saved to %s", ENCODINGS_PATH)

    # ------------------------------------------------------------------
    # Identification
    # ------------------------------------------------------------------

    def identify(self, frame: np.ndarray) -> Optional[str]:
        """
        Return the name of the first recognised person in *frame*, or None.
        Uses the distance metric: a match is accepted when distance < threshold.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb, model="hog")
        if not locations:
            return None

        encodings = face_recognition.face_encodings(rgb, locations)
        if not encodings:
            return None

        # Flatten all known encodings into parallel lists
        known_names: list[str] = []
        known_encs: list[np.ndarray] = []
        for name, enc_list in self._encodings.items():
            for enc in enc_list:
                known_names.append(name)
                known_encs.append(enc)

        if not known_encs:
            return None

        # Compare first detected face against all knowns
        probe = encodings[0]
        distances = face_recognition.face_distance(known_encs, probe)
        best_idx = int(np.argmin(distances))
        best_dist = float(distances[best_idx])

        if best_dist < CONFIDENCE_THRESHOLD:
            name = known_names[best_idx]
            logger.debug("Recognised '%s' (distance=%.3f)", name, best_dist)
            return name

        logger.debug("Unknown face (best distance=%.3f)", best_dist)
        return None

    def identify_all(self, frame: np.ndarray) -> list[tuple[str, tuple]]:
        """
        Return [(name, (top, right, bottom, left)), ...] for every face in
        the frame, using 'Unknown' for unrecognised faces.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb, model="hog")
        if not locations:
            return []

        encodings = face_recognition.face_encodings(rgb, locations)

        known_names: list[str] = []
        known_encs: list[np.ndarray] = []
        for name, enc_list in self._encodings.items():
            for enc in enc_list:
                known_names.append(name)
                known_encs.append(enc)

        results = []
        for enc, loc in zip(encodings, locations):
            if known_encs:
                distances = face_recognition.face_distance(known_encs, enc)
                best_idx = int(np.argmin(distances))
                best_dist = float(distances[best_idx])
                name = known_names[best_idx] if best_dist < CONFIDENCE_THRESHOLD else "Unknown"
            else:
                name = "Unknown"
            results.append((name, loc))
        return results

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_person(self, name: str, frame: np.ndarray) -> bool:
        """
        Add face encoding(s) from *frame* for *name*.
        Returns True if at least one encoding was extracted.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encs = face_recognition.face_encodings(rgb)
        if not encs:
            logger.warning("No face found in frame for '%s'", name)
            return False

        if name not in self._encodings:
            self._encodings[name] = []
        self._encodings[name].extend(encs)
        self._save_encodings()
        logger.info("Registered %d encoding(s) for '%s'", len(encs), name)
        return True

    def register_from_camera(self, name: str, camera_index: int = 0) -> bool:
        """
        Interactive registration: capture *REGISTRATION_SAMPLES* frames from
        the camera, extract encodings, save them, and create a DB record.
        """
        init_db()
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            logger.error("Cannot open camera %d", camera_index)
            return False

        print(f"Registering '{name}' — look at the camera.")
        print(f"Will capture {REGISTRATION_SAMPLES} samples. Press 'q' to cancel.")

        collected: list[np.ndarray] = []
        while len(collected) < REGISTRATION_SAMPLES:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            encs = face_recognition.face_encodings(rgb)
            if encs:
                collected.append(encs[0])
                progress = len(collected)
                cv2.putText(
                    frame,
                    f"Captured {progress}/{REGISTRATION_SAMPLES}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                )
                print(f"  Sample {progress}/{REGISTRATION_SAMPLES}")

            cv2.imshow(f"Register '{name}'", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            time.sleep(0.3)

        cap.release()
        cv2.destroyAllWindows()

        if not collected:
            print("No encodings captured. Aborting.")
            return False

        if name not in self._encodings:
            self._encodings[name] = []
        self._encodings[name].extend(collected)
        self._save_encodings()

        person_id = get_or_create_person(name)
        logger.info("DB record for '%s' → person_id=%d", name, person_id)
        print(f"Done! Registered '{name}' with {len(collected)} encoding(s).")
        return True

    def known_names(self) -> list[str]:
        return list(self._encodings.keys())


# ---------------------------------------------------------------------------
# Standalone test / registration
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--add-person", metavar="NAME", help="Register a new person")
    parser.add_argument("--identify", action="store_true", help="Identify from camera")
    parser.add_argument("--camera", type=int, default=int(os.environ.get("CAMERA_INDEX", "0")))
    args = parser.parse_args()

    module = FaceRecognitionModule()

    if args.add_person:
        module.register_from_camera(args.add_person, args.camera)

    elif args.identify:
        cap = cv2.VideoCapture(args.camera)
        print("Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            name = module.identify(frame)
            label = name or "Unknown"
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Identify", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

    else:
        print("Known persons:", module.known_names())
