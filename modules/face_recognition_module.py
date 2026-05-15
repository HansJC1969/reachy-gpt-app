"""
Face recognition using OpenCV Haar cascade detection and normalized face-crop
embeddings.  No external models or downloads required beyond opencv-python.

Each registered face is stored as a 4096-dim L2-normalized vector built from
a histogram-equalized 64×64 grayscale crop of the detected face region.
Identification uses cosine distance; a match is accepted when the distance is
below CONFIDENCE_THRESHOLD (default 0.5 — lower is stricter).

Typical distances:
  same person  : 0.02 – 0.20
  different    : 0.25 – 0.80

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
import numpy as np

from modules.memory import get_or_create_person, init_db

logger = logging.getLogger(__name__)

ENCODINGS_PATH       = Path(os.environ.get("ENCODINGS_PATH", "face_encodings.pkl"))
CONFIDENCE_THRESHOLD = float(os.environ.get("FACE_CONFIDENCE_THRESHOLD", "0.5"))
REGISTRATION_SAMPLES = 5

# Face crop is resized to CROP_SIZE × CROP_SIZE before flattening
_CROP_SIZE     = 64
_EMBEDDING_DIM = _CROP_SIZE * _CROP_SIZE   # 4096

_CASCADE_PATH  = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance in [0, 2].  0 = identical."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-8:
        return 2.0
    return float(1.0 - np.dot(a, b) / denom)


class FaceRecognitionModule:
    def __init__(self) -> None:
        # {name: list[np.ndarray]}  — each array has shape (_EMBEDDING_DIM,)
        self._encodings: dict[str, list] = {}
        self._cascade = cv2.CascadeClassifier(_CASCADE_PATH)
        if self._cascade.empty():
            raise RuntimeError(f"Failed to load cascade from {_CASCADE_PATH}")
        self._load_encodings()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_faces(
        self, gray: np.ndarray
    ) -> list[tuple[int, int, int, int]]:
        """Return list of (x, y, w, h) for all detected faces."""
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )
        return [tuple(f) for f in faces] if len(faces) > 0 else []

    def _crop_to_embedding(
        self, gray: np.ndarray, x: int, y: int, w: int, h: int
    ) -> Optional[np.ndarray]:
        """
        Crop *gray* to the face bbox (with 10 % padding), resize to
        _CROP_SIZE × _CROP_SIZE, histogram-equalize, and return an
        L2-normalized float32 vector of length _EMBEDDING_DIM.
        """
        fh, fw = gray.shape[:2]
        pad_x = max(1, int(w * 0.10))
        pad_y = max(1, int(h * 0.10))
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(fw, x + w + pad_x)
        y2 = min(fh, y + h + pad_y)
        crop = gray[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        crop = cv2.resize(crop, (_CROP_SIZE, _CROP_SIZE))
        crop = cv2.equalizeHist(crop)
        vec  = crop.flatten().astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def _extract_embedding(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Return an embedding for the largest face in *frame*, or None."""
        if frame is None or frame.size == 0:
            return None
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray  = cv2.equalizeHist(gray)
        faces = self._detect_faces(gray)
        if not faces:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return self._crop_to_embedding(gray, x, y, w, h)

    def _extract_all_embeddings(
        self, frame: np.ndarray
    ) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
        """
        Return [(embedding, (top, right, bottom, left)), ...] for every
        detected face in *frame*.
        """
        if frame is None or frame.size == 0:
            return []
        fh, fw = frame.shape[:2]
        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray   = cv2.equalizeHist(gray)
        faces  = self._detect_faces(gray)
        result = []
        for x, y, w, h in faces:
            emb = self._crop_to_embedding(gray, x, y, w, h)
            if emb is None:
                continue
            bbox = (y, min(fw, x + w), min(fh, y + h), x)  # top,right,bottom,left
            result.append((emb, bbox))
        return result

    def _best_match(
        self,
        embedding: np.ndarray,
        known_names: list[str],
        known_encs: list[np.ndarray],
    ) -> tuple[Optional[str], float]:
        """Return (name, distance) of the closest known face, or (None, 2.0)."""
        if not known_encs:
            return None, 2.0
        distances = np.array([_cosine_distance(embedding, e) for e in known_encs])
        idx  = int(np.argmin(distances))
        dist = float(distances[idx])
        if dist < CONFIDENCE_THRESHOLD:
            return known_names[idx], dist
        return None, dist

    def _flat_known(self) -> tuple[list[str], list[np.ndarray]]:
        names: list[str]       = []
        encs:  list[np.ndarray] = []
        for name, enc_list in self._encodings.items():
            for enc in enc_list:
                names.append(name)
                encs.append(enc)
        return names, encs

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_encodings(self) -> None:
        if not ENCODINGS_PATH.exists():
            logger.info("No existing encodings file found; starting fresh")
            return
        try:
            with open(ENCODINGS_PATH, "rb") as f:
                data = pickle.load(f)

            # Validate embedding dimension — old mediapipe (936-dim) or
            # dlib (128-dim) encodings are incompatible and must be discarded.
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
                    "Encodings file has incompatible format (old dlib/mediapipe "
                    "encodings?). Resetting — please re-register all persons "
                    "with --add-person."
                )
                self._encodings = {}
        except Exception:
            logger.exception("Failed to load encodings from %s; starting fresh", ENCODINGS_PATH)
            self._encodings = {}

    def _save_encodings(self) -> None:
        # Write to a temp file first, then atomically rename to avoid
        # corrupting the encodings file if the process crashes mid-write.
        tmp_path = ENCODINGS_PATH.with_suffix(".pkl.tmp")
        try:
            with open(tmp_path, "wb") as f:
                pickle.dump(self._encodings, f)
            tmp_path.replace(ENCODINGS_PATH)
            logger.info("Encodings saved to %s", ENCODINGS_PATH)
        except OSError:
            logger.exception("Failed to save encodings to %s", ENCODINGS_PATH)
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Public API
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
            names, encs = self._flat_known()
            name, dist  = self._best_match(embedding, names, encs)
            if name:
                logger.debug("Recognised '%s' (cosine dist=%.3f)", name, dist)
            else:
                logger.debug("Unknown face (best dist=%.3f)", dist)
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
            names, encs = self._flat_known()
            results = []
            for embedding, bbox in faces:
                name, _ = self._best_match(embedding, names, encs)
                results.append((name or "Unknown", bbox))
            return results
        except Exception:
            logger.debug("identify_all() failed", exc_info=True)
            return []

    def register_person(self, name: str, frame: np.ndarray) -> bool:
        """
        Add a face embedding from *frame* for *name*.
        Returns True if a face was found and stored.
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
