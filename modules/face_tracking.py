"""
Real-time face tracking using OpenCV Haar cascade.
Moves Reachy Mini's head to follow the detected face using look_at_image(),
which delegates inverse kinematics to the SDK.
Rotates the body yaw when the face is >50% off-centre horizontally.

Safe-speed tracking design (all limits from AGENTS.md)
-------------------------------------------------------
• EMA low-pass filter (α=0.20) — heavy smoothing to suppress jitter.
• Dead zone 20%: no command when face is within 20% of centre in both axes.
• Move threshold 8%: only command when smoothed position changed ≥8% of frame.
• Cooldown 1.0 s between head commands so each motion finishes before the next.
• look_at_image(duration=1.0) → 1-second smooth interpolation.
• Position clamp 20%: face coords clamped to ±20% of frame half-width/height
  before IK — limits effective head angle to ~20% of physical limit
  (20% × 40° ≈ ±8° effective pitch/roll, 20% × 180° ≈ ±36° yaw).
• Body rotation: 3° step, 2 s cooldown, triggers at >50% horizontal offset.

SDK: reachy_mini (ReachyMini)
  reachy.look_at_image(u, v, duration)      — smooth pixel-based head pointing
  reachy.goto_target(body_yaw=rad, duration) — smooth body yaw
  reachy.goto_target(head=…, body_yaw=0.0)  — used for center_head()

Joint limits (from AGENTS.md — SDK clamps automatically):
  Head pitch / roll : ±40°
  Head yaw          : ±180°
  Body yaw          : ±160°
  Head-body delta   : max 65°

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

# OpenCV's bundled frontalface cascade — no download required
_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# ---------------------------------------------------------------------------
# Body rotation parameters
# ---------------------------------------------------------------------------

# Physical limits from AGENTS.md; SDK clamps automatically
BODY_YAW_MIN = math.radians(-160.0)
BODY_YAW_MAX = math.radians( 160.0)

# Increment absolute body yaw per rotation command (safe: 3° per step)
BODY_ROTATION_STEP = math.radians(3.0)

# Body rotation triggers when smoothed face is >50% off horizontal centre
BODY_ROTATION_THRESHOLD = 0.50

# Minimum seconds between body-rotation commands
BODY_ROTATION_COOLDOWN = 2.0

# ---------------------------------------------------------------------------
# Head tracking parameters  (safe limits: 20% of physical maximum)
# ---------------------------------------------------------------------------

# Haar detection minimum face size (px)
_SCALE_FACTOR   = 1.1
_MIN_NEIGHBOURS = 5
MIN_FACE_PX     = 60

# EMA low-pass filter weight (lower α → smoother, more lag; 0.20 = heavy smoothing)
EMA_ALPHA = 0.20

# Dead zone 20%: face inside this box around centre → no head movement
DEAD_ZONE = 0.20          # fraction of half-frame width/height

# Minimum change (normalised) from last commanded position to trigger a move
MOVE_THRESHOLD = 0.08     # 8% of frame

# Clamp face position to ±20% of half-frame before IK.
# Limits effective head angle to ~20% of physical maximum:
#   pitch/roll: 20% × 40° ≈ ±8°   yaw: 20% × 180° ≈ ±36°
POSITION_CLAMP = 0.20

# 1-second smooth interpolation per SDK motion command
LOOK_DURATION = 1.0

# Minimum gap between successive head commands (must be ≥ LOOK_DURATION)
LOOK_COOLDOWN = 1.0


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
        self._cascade = cv2.CascadeClassifier(_CASCADE_PATH)
        if self._cascade.empty():
            raise RuntimeError(f"Failed to load cascade from {_CASCADE_PATH}")

        # Accumulated body yaw (absolute, radians)
        self._body_yaw: float = 0.0

        # EMA state — reset to None when face is lost
        self._ema_cx: Optional[float] = None
        self._ema_cy: Optional[float] = None

        # Last normalised position actually commanded to the head
        self._last_cmd_dx: float = 0.0
        self._last_cmd_dy: float = 0.0

        # Timestamps for cooldowns
        self._last_look_time:  float = 0.0
        self._last_body_time:  float = 0.0

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

        largest = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest
        fh, fw = frame.shape[:2]
        return FacePosition(x=x, y=y, w=w, h=h, frame_w=fw, frame_h=fh)

    # ------------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------------

    def update(self, face: FacePosition) -> None:
        """
        Apply EMA filter, dead zone, move threshold and cooldown, then
        command the head and body only when a meaningful move is warranted.
        """
        # 1. Update EMA low-pass filter
        if self._ema_cx is None:
            self._ema_cx = float(face.cx)
            self._ema_cy = float(face.cy)
        else:
            self._ema_cx = EMA_ALPHA * face.cx + (1.0 - EMA_ALPHA) * self._ema_cx
            self._ema_cy = EMA_ALPHA * face.cy + (1.0 - EMA_ALPHA) * self._ema_cy

        # 2. Normalised offsets from smoothed position
        dx = (self._ema_cx - face.frame_w / 2.0) / (face.frame_w / 2.0)
        dy = (self._ema_cy - face.frame_h / 2.0) / (face.frame_h / 2.0)

        # 3. Head movement: dead zone → threshold → cooldown → clamp → command
        if abs(dx) > DEAD_ZONE or abs(dy) > DEAD_ZONE:
            delta_dx = abs(dx - self._last_cmd_dx)
            delta_dy = abs(dy - self._last_cmd_dy)
            now = time.monotonic()

            if (delta_dx >= MOVE_THRESHOLD or delta_dy >= MOVE_THRESHOLD) and \
               (now - self._last_look_time >= LOOK_COOLDOWN):

                # Clamp to 80% of frame half-extent before IK
                dx_c = max(-POSITION_CLAMP, min(POSITION_CLAMP, dx))
                dy_c = max(-POSITION_CLAMP, min(POSITION_CLAMP, dy))
                target_cx = int((dx_c + 1.0) * face.frame_w / 2.0)
                target_cy = int((dy_c + 1.0) * face.frame_h / 2.0)

                if self.reachy is not None:
                    self._look_at_face(target_cx, target_cy)
                else:
                    logger.debug(
                        "[sim] look → norm=(%.2f, %.2f)  pixel=(%d, %d)",
                        dx_c, dy_c, target_cx, target_cy,
                    )

                self._last_cmd_dx  = dx
                self._last_cmd_dy  = dy
                self._last_look_time = now

        # 4. Body rotation when face is far off horizontal centre
        now = time.monotonic()
        if abs(dx) > BODY_ROTATION_THRESHOLD and \
           (now - self._last_body_time >= BODY_ROTATION_COOLDOWN):
            self._rotate_body(dx)
            self._last_body_time = now

    def center_head(self) -> None:
        """Return the head and body smoothly to the neutral position and reset state."""
        self._body_yaw     = 0.0
        self._ema_cx       = None
        self._ema_cy       = None
        self._last_cmd_dx  = 0.0
        self._last_cmd_dy  = 0.0
        self._last_look_time  = 0.0
        self._last_body_time  = 0.0

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

    def _look_at_face(self, cx: int, cy: int) -> None:
        """Smooth pixel-based head pointing via SDK IK."""
        try:
            self.reachy.look_at_image(cx, cy, duration=LOOK_DURATION)
        except Exception:
            logger.exception("Failed to point head at face")

    def _rotate_body(self, dx_norm: float) -> None:
        """Increment absolute body yaw toward the face by BODY_ROTATION_STEP."""
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
            self.reachy.goto_target(body_yaw=self._body_yaw, duration=LOOK_DURATION)
        except Exception:
            logger.exception("Failed to rotate body")

    # ------------------------------------------------------------------
    # Visualisation helper
    # ------------------------------------------------------------------

    @staticmethod
    def draw_face(frame: np.ndarray, face: FacePosition) -> np.ndarray:
        """Draw bounding box and cross-hair on frame (for debugging)."""
        cv2.rectangle(
            frame, (face.x, face.y), (face.x + face.w, face.y + face.h),
            (0, 255, 0), 2,
        )
        cv2.circle(frame, (face.cx, face.cy), 4, (0, 0, 255), -1)
        label = f"dx={face.dx_norm:.2f}  dy={face.dy_norm:.2f}"
        cv2.putText(
            frame, label, (face.x, face.y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
        )
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
