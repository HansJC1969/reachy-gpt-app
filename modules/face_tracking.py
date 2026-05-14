"""
Real-time face tracking using MediaPipe Face Detection.
Moves Reachy Mini's head to follow the detected face using look_at_image(),
which delegates inverse kinematics to the SDK.
Rotates the body yaw when the face is >40% off-centre horizontally.

SDK: reachy_mini (ReachyMini)
  reachy.look_at_image(u, v, duration=0)    — instant pixel-based head pointing
  reachy.set_target_body_yaw(rad)           — absolute body yaw command
  reachy.goto_target(head=…, body_yaw=0.0) — smooth motion; used for center_head()

Can be tested without a robot:
    python -m modules.face_tracking --no-robot
"""

import logging
import math
import os
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

_mp_face_detection = mp.solutions.face_detection

# Body yaw limits — from Reachy Mini safety spec
BODY_YAW_MIN = math.radians(-160.0)
BODY_YAW_MAX = math.radians(160.0)

# How much to increment body yaw per tracking call when face is off-centre
BODY_ROTATION_STEP = math.radians(5.0)

# Body rotation triggers when face is this fraction off horizontal centre
BODY_ROTATION_THRESHOLD = 0.40

# Ignore detections whose bounding box is smaller than this (pixels)
MIN_FACE_PX = 60


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
        reachy : reachy_mini.ReachyMini or None
            Pass None to run in simulation mode (no robot).
        """
        self.reachy = reachy
        # model_selection=0: optimised for short range (< 2 m) — ideal for Reachy
        self._detector = _mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5,
        )

        # Accumulated body yaw in radians; set_target_body_yaw takes an absolute angle
        self._body_yaw: float = 0.0

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_face(self, frame: np.ndarray) -> Optional[FacePosition]:
        """Return the most-confident detected face, or None."""
        if frame is None or frame.size == 0:
            return None
        try:
            fh, fw = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._detector.process(rgb)
            if not results.detections:
                return None

            # Pick the detection with the highest confidence score
            best = max(results.detections, key=lambda d: d.score[0])
            bb = best.location_data.relative_bounding_box

            x = int(bb.xmin * fw)
            y = int(bb.ymin * fh)
            w = int(bb.width  * fw)
            h = int(bb.height * fh)

            # Clamp to frame bounds
            x = max(0, min(x, fw - 1))
            y = max(0, min(y, fh - 1))
            w = max(0, min(w, fw - x))
            h = max(0, min(h, fh - y))

            if w < MIN_FACE_PX or h < MIN_FACE_PX:
                return None

            return FacePosition(x=x, y=y, w=w, h=h, frame_w=fw, frame_h=fh)
        except Exception:
            logger.debug("detect_face() failed", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------------

    def update(self, face: FacePosition) -> None:
        """Point the head at *face* and rotate the body if it is too far off-centre."""
        if self.reachy is not None:
            self._look_at_face(face)
        else:
            logger.debug("[sim] face centre pixel (%d, %d)", face.cx, face.cy)

        if abs(face.dx_norm) > BODY_ROTATION_THRESHOLD:
            self._rotate_body(face.dx_norm)

    def center_head(self) -> None:
        """Return the head and body smoothly to the neutral (forward) position."""
        self._body_yaw = 0.0
        if self.reachy is not None:
            try:
                from reachy_mini.utils import create_head_pose
                self.reachy.goto_target(
                    head=create_head_pose(yaw=0, pitch=0, degrees=True),
                    body_yaw=0.0,
                    duration=0.5,
                )
            except Exception:
                logger.warning("Could not centre head", exc_info=True)

    # ------------------------------------------------------------------
    # Robot helpers
    # ------------------------------------------------------------------

    def _look_at_face(self, face: FacePosition) -> None:
        """Use SDK inverse kinematics to point the head at the face pixel."""
        try:
            self.reachy.look_at_image(face.cx, face.cy, duration=0)
        except Exception:
            logger.exception("Failed to point head at face")

    def _rotate_body(self, dx_norm: float) -> None:
        """Increment the absolute body yaw toward the face by BODY_ROTATION_STEP."""
        self._body_yaw = float(np.clip(
            self._body_yaw + math.copysign(BODY_ROTATION_STEP, dx_norm),
            BODY_YAW_MIN,
            BODY_YAW_MAX,
        ))
        if self.reachy is None:
            logger.debug(
                "[sim] body_yaw=%.2f rad (%.1f°)",
                self._body_yaw, math.degrees(self._body_yaw),
            )
            return
        try:
            self.reachy.set_target_body_yaw(self._body_yaw)
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
