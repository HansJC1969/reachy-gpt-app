"""Dance and emotion moves for the movement queue system.

Adapted from pollen-robotics/reachy_mini_conversation_app.
"""

from __future__ import annotations
import logging
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from reachy_mini.motion.move import Move
from reachy_mini.motion.recorded_move import RecordedMoves
from reachy_mini_dances_library.dance_move import DanceMove


logger = logging.getLogger(__name__)


class DanceQueueMove(Move):  # type: ignore
    """Wrapper for dance moves to work with the movement queue system."""

    def __init__(self, move_name: str):
        self.dance_move = DanceMove(move_name)
        self.move_name = move_name

    @property
    def duration(self) -> float:
        return float(self.dance_move.duration)

    def evaluate(self, t: float) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        try:
            head_pose, antennas, body_yaw = self.dance_move.evaluate(t)
            if isinstance(antennas, tuple):
                antennas = np.array([antennas[0], antennas[1]])
            return (head_pose, antennas, body_yaw)
        except Exception as e:
            logger.error("Error evaluating dance move '%s' at t=%s: %s", self.move_name, t, e)
            from reachy_mini.utils import create_head_pose
            return (create_head_pose(0, 0, 0, 0, 0, 0, degrees=True), np.array([0.0, 0.0], dtype=np.float64), 0.0)


class EmotionQueueMove(Move):  # type: ignore
    """Wrapper for emotion moves to work with the movement queue system."""

    def __init__(self, emotion_name: str, recorded_moves: RecordedMoves):
        self.emotion_move = recorded_moves.get(emotion_name)
        self.emotion_name = emotion_name

    @property
    def duration(self) -> float:
        return float(self.emotion_move.duration)

    def evaluate(self, t: float) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        try:
            head_pose, antennas, body_yaw = self.emotion_move.evaluate(t)
            if isinstance(antennas, tuple):
                antennas = np.array([antennas[0], antennas[1]])
            return (head_pose, antennas, body_yaw)
        except Exception as e:
            logger.error("Error evaluating emotion '%s' at t=%s: %s", self.emotion_name, t, e)
            from reachy_mini.utils import create_head_pose
            return (create_head_pose(0, 0, 0, 0, 0, 0, degrees=True), np.array([0.0, 0.0], dtype=np.float64), 0.0)


class GotoQueueMove(Move):  # type: ignore
    """Linear interpolation move for goto-style head positioning."""

    def __init__(
        self,
        target_head_pose: NDArray[np.float32],
        start_head_pose: NDArray[np.float32] | None = None,
        target_antennas: Tuple[float, float] = (0, 0),
        start_antennas: Tuple[float, float] | None = None,
        target_body_yaw: float = 0,
        start_body_yaw: float | None = None,
        duration: float = 1.0,
    ):
        self._duration = duration
        self.target_head_pose = target_head_pose
        self.start_head_pose = start_head_pose
        self.target_antennas = target_antennas
        self.start_antennas = start_antennas or (0, 0)
        self.target_body_yaw = target_body_yaw
        self.start_body_yaw = start_body_yaw or 0

    @property
    def duration(self) -> float:
        return self._duration

    def evaluate(self, t: float) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        try:
            from reachy_mini.utils import create_head_pose
            from reachy_mini.utils.interpolation import linear_pose_interpolation

            t_clamped = max(0.0, min(1.0, t / self.duration))
            start_pose = self.start_head_pose if self.start_head_pose is not None else create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
            head_pose = linear_pose_interpolation(start_pose, self.target_head_pose, t_clamped)
            antennas = np.array([
                self.start_antennas[0] + (self.target_antennas[0] - self.start_antennas[0]) * t_clamped,
                self.start_antennas[1] + (self.target_antennas[1] - self.start_antennas[1]) * t_clamped,
            ], dtype=np.float64)
            body_yaw = self.start_body_yaw + (self.target_body_yaw - self.start_body_yaw) * t_clamped
            return (head_pose, antennas, body_yaw)
        except Exception as e:
            logger.error("Error evaluating goto move at t=%s: %s", t, e)
            return (self.target_head_pose.astype(np.float64), np.array(list(self.target_antennas), dtype=np.float64), self.target_body_yaw)
