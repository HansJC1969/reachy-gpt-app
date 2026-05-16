"""
Face detection via OpenCV Haar cascade.

Provides secondary offsets (x, y, z, roll, pitch, yaw) in the world frame
for MovementManager to poll at 60Hz. No direct SDK calls — all movement is
handled by MovementManager.

Detection: Haar frontalface cascade (no extra dependencies).
Smoothing: EMA low-pass filter (α=0.20).
Dead zone: 15% of half-frame around centre → no offset.
Scale: face position → yaw/pitch radians with gentle gain.

Can be tested without a robot:
    python -m modules.face_tracking --no-robot
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Detection parameters
_SCALE_FACTOR   = 1.1
_MIN_NEIGHBOURS = 5
MIN_FACE_PX     = 60

# EMA filter — heavy smoothing (lower α = more lag, less jitter)
EMA_ALPHA = 0.20

# Dead zone — no offset when face is within this fraction of half-frame
DEAD_ZONE = 0.15

# Clamp applied input before scaling — keeps head motion safe
POSITION_CLAMP = 0.50

# Scale face normalised position → world-frame radians
# At full clamp (0.50 off-centre): yaw≈14°, pitch≈9°
FACE_TRACK_YAW_SCALE   = 0.50   # rad per unit dx_norm (after clamp)
FACE_TRACK_PITCH_SCALE = 0.35   # rad per unit dy_norm (after clamp)

# Face-loss patience: keep last offsets for this long before zeroing
FACE_LOSS_TIMEOUT = 0.5   # seconds


@dataclass
class FacePosition:
    """Detected face bounding box and normalised offsets."""
    x: int; y: int; w: int; h: int; frame_w: int; frame_h: int

    @property
    def cx(self) -> int:
        return self.x + self.w // 2

    @property
    def cy(self) -> int:
        return self.y + self.h // 2

    @property
    def dx_norm(self) -> float:
        return (self.cx - self.frame_w / 2) / (self.frame_w / 2)

    @property
    def dy_norm(self) -> float:
        return (self.cy - self.frame_h / 2) / (self.frame_h / 2)


class FaceTracker:
    """
    Face detection + conversion to MovementManager secondary offsets.

    Implements the camera_worker interface expected by MovementManager:
        get_face_tracking_offsets() → (x, y, z, roll, pitch, yaw)  world frame (m / rad)
    """

    def __init__(self, reachy=None) -> None:
        self._cascade = cv2.CascadeClassifier(_CASCADE_PATH)
        if self._cascade.empty():
            raise RuntimeError(f"Failed to load cascade from {_CASCADE_PATH}")

        # EMA state
        self._ema_cx: Optional[float] = None
        self._ema_cy: Optional[float] = None

        # Thread-safe offset storage
        self._offsets_lock = threading.Lock()
        self._offsets: Tuple[float, float, float, float, float, float] = (0.0,) * 6
        self._last_face_time: float = 0.0

        # reachy kept only for compatibility (not used for movement)
        self.reachy = reachy

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_face(self, frame: np.ndarray) -> Optional[FacePosition]:
        """Return the largest detected face, or None."""
        if frame is None or frame.size == 0:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=_SCALE_FACTOR,
            minNeighbors=_MIN_NEIGHBOURS,
            minSize=(MIN_FACE_PX, MIN_FACE_PX),
        )
        if len(faces) == 0:
            return None

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        fh, fw = frame.shape[:2]
        return FacePosition(x=x, y=y, w=w, h=h, frame_w=fw, frame_h=fh)

    # ------------------------------------------------------------------
    # Offset update — called from camera_loop at ~30fps
    # ------------------------------------------------------------------

    def update_offsets(self, face: Optional[FacePosition]) -> None:
        """
        Convert face position to (x, y, z, roll, pitch, yaw) world-frame offsets
        and store them for MovementManager to poll.
        """
        if face is None:
            # Hold last offsets briefly to avoid jumpy behaviour on single-frame misses
            if time.monotonic() - self._last_face_time > FACE_LOSS_TIMEOUT:
                self._ema_cx = None
                self._ema_cy = None
                with self._offsets_lock:
                    self._offsets = (0.0,) * 6
            return

        self._last_face_time = time.monotonic()

        # EMA smoothing
        if self._ema_cx is None:
            self._ema_cx = float(face.cx)
            self._ema_cy = float(face.cy)
        else:
            self._ema_cx = EMA_ALPHA * face.cx + (1.0 - EMA_ALPHA) * self._ema_cx
            self._ema_cy = EMA_ALPHA * face.cy + (1.0 - EMA_ALPHA) * self._ema_cy

        # Normalised position (centre = 0, edge = ±1)
        dx = (self._ema_cx - face.frame_w / 2.0) / (face.frame_w / 2.0)
        dy = (self._ema_cy - face.frame_h / 2.0) / (face.frame_h / 2.0)

        # Dead zone
        if abs(dx) < DEAD_ZONE and abs(dy) < DEAD_ZONE:
            with self._offsets_lock:
                self._offsets = (0.0,) * 6
            return

        # Clamp
        dx_c = max(-POSITION_CLAMP, min(POSITION_CLAMP, dx))
        dy_c = max(-POSITION_CLAMP, min(POSITION_CLAMP, dy))

        # World-frame offsets (metres / radians)
        # Positive yaw   = head turns right   (face is to the right, dx > 0)
        # Positive pitch = head tilts up       (face is below centre, dy > 0)
        yaw_rad   =  dx_c * FACE_TRACK_YAW_SCALE
        pitch_rad =  dy_c * FACE_TRACK_PITCH_SCALE

        with self._offsets_lock:
            self._offsets = (0.0, 0.0, 0.0, 0.0, pitch_rad, yaw_rad)

    # ------------------------------------------------------------------
    # MovementManager interface
    # ------------------------------------------------------------------

    def get_face_tracking_offsets(self) -> Tuple[float, float, float, float, float, float]:
        """Return (x, y, z, roll, pitch, yaw) world-frame offsets for MovementManager."""
        with self._offsets_lock:
            return self._offsets

    def center(self) -> None:
        """Reset all offsets to zero (call when robot goes to sleep or tracking pauses)."""
        self._ema_cx = None
        self._ema_cy = None
        with self._offsets_lock:
            self._offsets = (0.0,) * 6

    # Backward-compat alias used in some earlier code
    def center_head(self) -> None:
        self.center()

    # ------------------------------------------------------------------
    # Visualisation helper
    # ------------------------------------------------------------------

    @staticmethod
    def draw_face(frame: np.ndarray, face: FacePosition) -> np.ndarray:
        cv2.rectangle(frame, (face.x, face.y), (face.x + face.w, face.y + face.h), (0, 255, 0), 2)
        cv2.circle(frame, (face.cx, face.cy), 4, (0, 0, 255), -1)
        cv2.putText(frame, f"dx={face.dx_norm:.2f}  dy={face.dy_norm:.2f}",
                    (face.x, face.y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return frame


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-robot", action="store_true")
    parser.add_argument("--camera", type=int, default=int(os.environ.get("CAMERA_INDEX", "0")))
    args = parser.parse_args()

    tracker = FaceTracker(reachy=None)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Cannot open camera {args.camera}")
        raise SystemExit(1)

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        face = tracker.detect_face(frame)
        tracker.update_offsets(face)
        offsets = tracker.get_face_tracking_offsets()
        if face:
            FaceTracker.draw_face(frame, face)
            pitch_deg = math.degrees(offsets[4])
            yaw_deg   = math.degrees(offsets[5])
            print(f"\r  pitch={pitch_deg:+.1f}°  yaw={yaw_deg:+.1f}°    ", end="", flush=True)
        cv2.imshow("FaceTracker test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
