"""
Real-time face tracking using OpenCV Haar cascades.
Moves Reachy's neck (yaw/pitch) to centre the detected face.
Rotates the body if the face is >40% off-centre horizontally.

Can be tested without a robot:
    python -m modules.face_tracking --no-robot
"""

import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Path to OpenCV's bundled frontalface cascade
_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Neck joint limits (degrees)
YAW_MIN, YAW_MAX = -45.0, 45.0
PITCH_MIN, PITCH_MAX = -30.0, 20.0

# Proportional gain for smooth tracking
KP_YAW = 0.08
KP_PITCH = 0.06

# Body-rotation triggers when face is this fraction off horizontal centre
BODY_ROTATION_THRESHOLD = 0.40

# Minimum face detection scale and neighbours
SCALE_FACTOR = 1.1
MIN_NEIGHBOURS = 5
MIN_FACE_PX = 60  # ignore faces smaller than this


@dataclass
class FacePosition:
    """Detected face bounding box and normalised offsets."""
    x: int
    y: int
    w: int
    h: int
    frame_w: int
    frame_h: int

    @property
    def cx(self) -> int:
        return self.x + self.w // 2

    @property
    def cy(self) -> int:
        return self.y + self.h // 2

    @property
    def dx_norm(self) -> float:
        """Horizontal offset normalised to [-1, +1]."""
        return (self.cx - self.frame_w / 2) / (self.frame_w / 2)

    @property
    def dy_norm(self) -> float:
        """Vertical offset normalised to [-1, +1] (positive = face below centre)."""
        return (self.cy - self.frame_h / 2) / (self.frame_h / 2)


class FaceTracker:
    def __init__(self, reachy=None) -> None:
        """
        Parameters
        ----------
        reachy : reachy2_sdk.ReachySDK or None
            Pass None to run in simulation mode (no robot).
        """
        self.reachy = reachy
        self._cascade = cv2.CascadeClassifier(_CASCADE_PATH)
        if self._cascade.empty():
            raise RuntimeError(f"Failed to load cascade from {_CASCADE_PATH}")

        # Current neck state (degrees)
        self._yaw: float = 0.0
        self._pitch: float = 0.0

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
            scaleFactor=SCALE_FACTOR,
            minNeighbors=MIN_NEIGHBOURS,
            minSize=(MIN_FACE_PX, MIN_FACE_PX),
        )

        if len(faces) == 0:
            return None

        # Pick the largest face (most likely the primary person)
        largest = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest
        fh, fw = frame.shape[:2]
        return FacePosition(x=x, y=y, w=w, h=h, frame_w=fw, frame_h=fh)

    # ------------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------------

    def update(self, face: FacePosition) -> None:
        """
        Given a detected face position, compute new neck angles and send
        them to the robot (or log them in no-robot mode).
        """
        # Proportional control: nudge toward the target
        self._yaw = float(np.clip(
            self._yaw + KP_YAW * face.dx_norm * (YAW_MAX - YAW_MIN) / 2,
            YAW_MIN, YAW_MAX,
        ))
        # Negative sign: face below centre → tilt head down (positive pitch on most robots)
        self._pitch = float(np.clip(
            self._pitch - KP_PITCH * face.dy_norm * (abs(PITCH_MIN) + PITCH_MAX) / 2,
            PITCH_MIN, PITCH_MAX,
        ))

        logger.debug("Neck target → yaw=%.1f°  pitch=%.1f°", self._yaw, self._pitch)

        if self.reachy is not None:
            self._move_neck(self._yaw, self._pitch)
        else:
            logger.debug("[sim] neck yaw=%.2f pitch=%.2f", self._yaw, self._pitch)

        # Body rotation when face is far off-centre
        if abs(face.dx_norm) > BODY_ROTATION_THRESHOLD:
            self._rotate_body(face.dx_norm)

    def center_head(self) -> None:
        """Return neck to neutral position."""
        self._yaw = 0.0
        self._pitch = 0.0
        if self.reachy is not None:
            self._move_neck(0.0, 0.0)

    # ------------------------------------------------------------------
    # Robot helpers
    # ------------------------------------------------------------------

    def _move_neck(self, yaw: float, pitch: float) -> None:
        try:
            head = self.reachy.head
            head.neck.yaw.goal_position = yaw
            head.neck.pitch.goal_position = pitch
        except Exception:
            logger.exception("Failed to move neck")

    def _rotate_body(self, dx_norm: float) -> None:
        """Rotate the mobile base slightly toward the face."""
        if self.reachy is None:
            logger.debug("[sim] body rotate dx_norm=%.2f", dx_norm)
            return
        try:
            # Positive dx_norm → face is to the right → rotate right (positive)
            speed = 0.10  # m/s tangential speed
            direction = math.copysign(speed, dx_norm)
            self.reachy.mobile_base.set_speed(vx=0.0, vy=0.0, vtheta=direction)
            time.sleep(0.15)
            self.reachy.mobile_base.set_speed(vx=0.0, vy=0.0, vtheta=0.0)
        except Exception:
            logger.exception("Failed to rotate body")

    # ------------------------------------------------------------------
    # Visualisation helper
    # ------------------------------------------------------------------

    @staticmethod
    def draw_face(frame: np.ndarray, face: FacePosition) -> np.ndarray:
        """Draw bounding box and cross-hair on frame (for debugging)."""
        cv2.rectangle(frame, (face.x, face.y), (face.x + face.w, face.y + face.h), (0, 255, 0), 2)
        cv2.circle(frame, (face.cx, face.cy), 4, (0, 0, 255), -1)
        label = f"dx={face.dx_norm:.2f}  dy={face.dy_norm:.2f}"
        cv2.putText(frame, label, (face.x, face.y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
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
        if face:
            tracker.update(face)
            FaceTracker.draw_face(frame, face)
        cv2.imshow("FaceTracker test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
