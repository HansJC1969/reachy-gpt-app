"""
Face recognition using MediaPipe FaceMesh.

Embeddings are 936-dimensional vectors built from the (x, y) coordinates of
all 468 FaceMesh landmarks, centred and L2-normalised to be pose-robust.
Similarity is measured with cosine distance (lower = more similar).

CONFIDENCE_THRESHOLD (default 0.5) is the maximum cosine distance accepted as
a match — lower values are stricter.  Same-person distances are typically
0.02–0.15; different people are typically 0.3–0.8.

Standalone registration:
    python -m modules.face_recognition_module --add-person "Alice"
    python -m modules.face_recognition_module --identify
"""

import logging
import os
import pickle
import time
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from modules.memory import get_or_create_person, init_db

logger = logging.getLogger(__name__)

ENCODINGS_PATH        = Path(os.environ.get("ENCODINGS_PATH", "face_encodings.pkl"))
CONFIDENCE_THRESHOLD  = float(os.environ.get("FACE_CONFIDENCE_THRESHOLD", "0.5"))
REGISTRATION_SAMPLES  = 5

# 468 FaceMesh landmarks × 2 (x, y) = 936-dimensional embedding
_EMBEDDING_DIM = 468 * 2

_mp_face_mesh = mp.solutions.face_mesh


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance in [0, 2].  0 = identical, 2 = opposite."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-8:
        return 2.0
    return float(1.0 - np.dot(a, b) / denom)


class FaceRecognitionModule:
    def __init__(self) -> None:
        # {name: list[np.ndarray]}  — each array is shape (_EMBEDDING_DIM,)
        self._encodings: dict[str, list] = {}
        # static_image_mode=True: every frame processed independently (no temporal carryover)
        self._mesh = _mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=5,
            refine_landmarks=False,
            min_detection_confidence=0.5,
        )
        self._load_encodings()

    # ------------------------------------------------------------------
    # Embedding extraction
    # ------------------------------------------------------------------

    def _extract_embedding(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Return a normalised 936-dim landmark embedding for the first face in
        *frame*, or None if no face is detected.
        """
        if frame is None or frame.size == 0:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        lms = results.multi_face_landmarks[0].landmark
        coords = np.array([[lm.x, lm.y] for lm in lms], dtype=np.float32)  # (468, 2)
        coords -= coords.mean(axis=0)           # centre on face
        norm = np.linalg.norm(coords)
        if norm > 0:
            coords /= norm                      # scale-normalise
        return coords.flatten()                 # (936,)

    def _extract_all_embeddings(
        self, frame: np.ndarray
    ) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
        """
        Return [(embedding, (top, right, bottom, left)), ...] for every face
        detected in *frame*.
        """
        if frame is None or frame.size == 0:
            return []
        fh, fw = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._mesh.process(rgb)
        if not results.multi_face_landmarks:
            return []

        output = []
        for face_lms in results.multi_face_landmarks:
            lms = face_lms.landmark
            coords = np.array([[lm.x, lm.y] for lm in lms], dtype=np.float32)

            # Bounding box from landmark extents (top, right, bottom, left)
            xs, ys = coords[:, 0], coords[:, 1]
            bbox = (
                max(0, int(ys.min() * fh)),   # top
                min(fw, int(xs.max() * fw)),  # right
                min(fh, int(ys.max() * fh)),  # bottom
                max(0, int(xs.min() * fw)),   # left
            )

            coords -= coords.mean(axis=0)
            norm = np.linalg.norm(coords)
            if norm > 0:
                coords /= norm
            output.append((coords.flatten(), bbox))
        return output

    def _best_match(
        self,
        embedding: np.ndarray,
        known_names: list[str],
        known_encs: list[np.ndarray],
    ) -> tuple[Optional[str], float]:
        """Return (name, distance) for the closest known encoding, or (None, 2.0)."""
        if not known_encs:
            return None, 2.0
        distances = np.array([_cosine_distance(embedding, e) for e in known_encs])
        best_idx  = int(np.argmin(distances))
        best_dist = float(distances[best_idx])
        if best_dist < CONFIDENCE_THRESHOLD:
            return known_names[best_idx], best_dist
        return None, best_dist

    # ------------------------------------------------------------------
    # Encoding persistence
    # ------------------------------------------------------------------

    def _load_encodings(self) -> None:
        if not ENCODINGS_PATH.exists():
            logger.info("No existing encodings file found; starting fresh")
            return
        try:
            with open(ENCODINGS_PATH, "rb") as f:
                data = pickle.load(f)

            # Validate embedding dimension — old face_recognition (128-dim dlib)
            # encodings are incompatible and must be discarded.
            compatible = all(
                isinstance(enc, np.ndarray) and enc.shape == (_EMBEDDING_DIM,)
                for enc_list in data.values()
                for enc in enc_list
            )
            if compatible:
                self._encodings = data
                logger.info(
                    "Loaded encodings for %d person(s): %s",
                    len(self._encodings), list(self._encodings.keys()),
                )
            else:
                logger.warning(
                    "Encodings file has incompatible format (old face_recognition/dlib "
                    "encodings?). Resetting — please re-register all persons with --add-person."
                )
                self._encodings = {}
        except Exception:
            logger.exception("Failed to load encodings from %s; starting fresh", ENCODINGS_PATH)
            self._encodings = {}

    def _save_encodings(self) -> None:
        try:
            with open(ENCODINGS_PATH, "wb") as f:
                pickle.dump(self._encodings, f)
            logger.info("Encodings saved to %s", ENCODINGS_PATH)
        except OSError:
            logger.exception("Failed to save encodings to %s", ENCODINGS_PATH)

    # ------------------------------------------------------------------
    # Identification
    # ------------------------------------------------------------------

    def identify(self, frame: np.ndarray) -> Optional[str]:
        """
        Return the name of the recognised person in *frame*, or None.
        A match is accepted when cosine distance < CONFIDENCE_THRESHOLD.
        """
        if frame is None or frame.size == 0:
            return None
        try:
            embedding = self._extract_embedding(frame)
            if embedding is None:
                return None

            known_names: list[str] = []
            known_encs:  list[np.ndarray] = []
            for name, enc_list in self._encodings.items():
                for enc in enc_list:
                    known_names.append(name)
                    known_encs.append(enc)

            name, dist = self._best_match(embedding, known_names, known_encs)
            if name:
                logger.debug("Recognised '%s' (cosine distance=%.3f)", name, dist)
            else:
                logger.debug("Unknown face (best distance=%.3f)", dist)
            return name
        except Exception:
            logger.debug("identify() failed", exc_info=True)
            return None

    def identify_all(self, frame: np.ndarray) -> list[tuple[str, tuple]]:
        """
        Return [(name, (top, right, bottom, left)), ...] for every face in
        the frame, using 'Unknown' for unrecognised faces.
        """
        if frame is None or frame.size == 0:
            return []
        try:
            faces = self._extract_all_embeddings(frame)
            if not faces:
                return []

            known_names: list[str] = []
            known_encs:  list[np.ndarray] = []
            for name, enc_list in self._encodings.items():
                for enc in enc_list:
                    known_names.append(name)
                    known_encs.append(enc)

            results = []
            for embedding, bbox in faces:
                name, _ = self._best_match(embedding, known_names, known_encs)
                results.append((name or "Unknown", bbox))
            return results
        except Exception:
            logger.debug("identify_all() failed", exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_person(self, name: str, frame: np.ndarray) -> bool:
        """
        Add face embedding(s) from *frame* for *name*.
        Returns True if at least one embedding was extracted.
        """
        if frame is None or frame.size == 0:
            logger.warning("register_person('%s'): invalid frame", name)
            return False

        embedding = self._extract_embedding(frame)
        if embedding is None:
            logger.warning("No face found in frame for '%s'", name)
            return False

        if name not in self._encodings:
            self._encodings[name] = []
        self._encodings[name].append(embedding)
        self._save_encodings()
        logger.info("Registered 1 embedding for '%s'", name)
        return True

    def register_from_camera(self, name: str, camera_index: int = 0) -> bool:
        """
        Interactive registration: capture *REGISTRATION_SAMPLES* frames from
        the camera, extract embeddings, save them, and create a DB record.
        """
        init_db()
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            logger.error("Cannot open camera %d", camera_index)
            return False

        print(f"Registering '{name}' — look at the camera.")
        print(f"Will capture {REGISTRATION_SAMPLES} samples. Press 'q' to cancel.")

        collected: list[np.ndarray] = []
        try:
            while len(collected) < REGISTRATION_SAMPLES:
                ret, frame = cap.read()
                if not ret:
                    break

                embedding = self._extract_embedding(frame)
                if embedding is not None:
                    collected.append(embedding)
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
        finally:
            cap.release()
            cv2.destroyAllWindows()

        if not collected:
            print("No embeddings captured. Aborting.")
            return False

        if name not in self._encodings:
            self._encodings[name] = []
        self._encodings[name].extend(collected)
        self._save_encodings()

        person_id = get_or_create_person(name)
        logger.info("DB record for '%s' → person_id=%d", name, person_id)
        print(f"Done! Registered '{name}' with {len(collected)} embedding(s).")
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
        if not cap.isOpened():
            print(f"Cannot open camera {args.camera}")
            raise SystemExit(1)
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
