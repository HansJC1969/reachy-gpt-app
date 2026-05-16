"""Movement system with sequential primary moves and additive secondary moves.

Adapted from pollen-robotics/reachy_mini_conversation_app.

Design overview
- Primary moves (emotions, dances, goto, breathing) are mutually exclusive and run
  sequentially.
- Secondary moves (speech sway, face tracking) are additive offsets applied on top
  of the current primary pose.
- There is a single control point to the robot: ReachyMini.set_target.
- The control loop runs near 60 Hz and is phase-aligned via a monotonic clock.
- Idle behaviour starts an infinite BreathingMove after a short inactivity delay
  unless listening is active.

Threading model
- A dedicated worker thread owns all real-time state and issues set_target commands.
- Other threads communicate via a command queue (enqueue moves, mark activity,
  toggle listening).
- Secondary offset producers set pending values guarded by locks; the worker
  snaps them atomically.

Units and frames
- Secondary offsets are metres for x/y/z and radians for roll/pitch/yaw in the
  world frame.
- Antennas and body_yaw are in radians.
- Head pose composition uses compose_world_offset(primary_head, secondary_head).

Safety
- Listening freezes antennas, then blends them back on unfreeze.
- Interpolations and blends are used to avoid jumps at all times.
- set_target errors are rate-limited in logs.
"""

from __future__ import annotations
import time
import logging
import threading
from queue import Empty, Queue
from typing import Any, Dict, Tuple
from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
from reachy_mini.motion.move import Move
from reachy_mini.utils.interpolation import (
    compose_world_offset,
    linear_pose_interpolation,
)


logger = logging.getLogger(__name__)

CONTROL_LOOP_FREQUENCY_HZ = 60.0

FullBodyPose = Tuple[NDArray[np.float32], Tuple[float, float], float]


class BreathingMove(Move):  # type: ignore
    """Breathing move with interpolation to neutral and then continuous breathing patterns."""

    def __init__(
        self,
        interpolation_start_pose: NDArray[np.float32],
        interpolation_start_antennas: Tuple[float, float],
        interpolation_duration: float = 1.0,
    ):
        self.interpolation_start_pose = interpolation_start_pose
        self.interpolation_start_antennas = np.array(interpolation_start_antennas)
        self.interpolation_duration = interpolation_duration

        self.neutral_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
        self.neutral_antennas = np.array([-0.1745, 0.1745])  # ~10° offset to reduce shaking

        self.breathing_z_amplitude = 0.005  # 5mm gentle breathing
        self.breathing_frequency = 0.1      # Hz (6 breaths per minute)
        self.antenna_sway_amplitude = np.deg2rad(15)  # 15 degrees
        self.antenna_frequency = 0.5        # Hz

    @property
    def duration(self) -> float:
        return float("inf")

    def evaluate(self, t: float) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        if t < self.interpolation_duration:
            alpha = t / self.interpolation_duration
            head_pose = linear_pose_interpolation(self.interpolation_start_pose, self.neutral_head_pose, alpha)
            antennas = ((1 - alpha) * self.interpolation_start_antennas + alpha * self.neutral_antennas).astype(np.float64)
        else:
            bt = t - self.interpolation_duration
            z = self.breathing_z_amplitude * np.sin(2 * np.pi * self.breathing_frequency * bt)
            head_pose = create_head_pose(x=0, y=0, z=z, roll=0, pitch=0, yaw=0, degrees=True, mm=False)
            sway = self.antenna_sway_amplitude * np.sin(2 * np.pi * self.antenna_frequency * bt)
            antennas = np.array([sway, -sway], dtype=np.float64)

        return (head_pose, antennas, 0.0)


def combine_full_body(primary_pose: FullBodyPose, secondary_pose: FullBodyPose) -> FullBodyPose:
    """Fuse primary and secondary poses: head via compose_world_offset, antennas/body_yaw via addition."""
    primary_head, primary_antennas, primary_body_yaw = primary_pose
    secondary_head, secondary_antennas, secondary_body_yaw = secondary_pose
    combined_head = compose_world_offset(primary_head, secondary_head, reorthonormalize=False)
    combined_antennas = (
        primary_antennas[0] + secondary_antennas[0],
        primary_antennas[1] + secondary_antennas[1],
    )
    combined_body_yaw = primary_body_yaw + secondary_body_yaw
    return (combined_head, combined_antennas, combined_body_yaw)


def clone_full_body_pose(pose: FullBodyPose) -> FullBodyPose:
    """Deep copy of pose tuple."""
    head, antennas, body_yaw = pose
    return (head.copy(), (float(antennas[0]), float(antennas[1])), float(body_yaw))


@dataclass
class MovementState:
    current_move: Move | None = None
    move_start_time: float | None = None
    last_activity_time: float = 0.0
    speech_offsets: Tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    face_tracking_offsets: Tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    last_primary_pose: FullBodyPose | None = None

    def update_activity(self) -> None:
        self.last_activity_time = time.monotonic()


@dataclass
class LoopFrequencyStats:
    mean: float = 0.0
    m2: float = 0.0
    min_freq: float = float("inf")
    count: int = 0
    last_freq: float = 0.0
    potential_freq: float = 0.0

    def reset(self) -> None:
        self.mean = 0.0
        self.m2 = 0.0
        self.min_freq = float("inf")
        self.count = 0


class MovementManager:
    """Coordinate sequential primary moves, additive secondary offsets, and robot output at 60Hz."""

    def __init__(self, current_robot: ReachyMini, camera_worker: Any = None):
        self.current_robot = current_robot
        self.camera_worker = camera_worker
        self._now = time.monotonic
        self.state = MovementState()
        self.state.last_activity_time = self._now()
        neutral_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
        self.state.last_primary_pose = (neutral_pose, (0.0, 0.0), 0.0)
        self.move_queue: deque[Move] = deque()
        self.idle_inactivity_delay = 0.3
        self.target_frequency = CONTROL_LOOP_FREQUENCY_HZ
        self.target_period = 1.0 / self.target_frequency
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._is_listening = False
        neutral_clone = clone_full_body_pose(self.state.last_primary_pose)
        self._last_commanded_pose: FullBodyPose = neutral_clone
        self._listening_antennas: Tuple[float, float] = neutral_clone[1]
        self._antenna_unfreeze_blend = 1.0
        self._antenna_blend_duration = 0.4
        self._last_listening_blend_time = self._now()
        self._breathing_active = False
        self._listening_debounce_s = 0.15
        self._last_listening_toggle_time = self._now()
        self._last_set_target_err = 0.0
        self._set_target_err_interval = 1.0
        self._set_target_err_suppressed = 0
        self._cached_secondary_offsets: tuple = ()
        self._cached_secondary_pose: FullBodyPose = (np.eye(4, dtype=np.float32), (0.0, 0.0), 0.0)
        self._command_queue: Queue[Tuple[str, Any]] = Queue()
        self._speech_offsets_lock = threading.Lock()
        self._pending_speech_offsets: Tuple[float, float, float, float, float, float] = (0.0,) * 6
        self._speech_offsets_dirty = False
        self._face_offsets_lock = threading.Lock()
        self._pending_face_offsets: Tuple[float, float, float, float, float, float] = (0.0,) * 6
        self._face_offsets_dirty = False
        self._shared_state_lock = threading.Lock()
        self._shared_last_activity_time = self.state.last_activity_time
        self._shared_is_listening = False
        self._status_lock = threading.Lock()
        self._freq_stats = LoopFrequencyStats()
        self._freq_snapshot = LoopFrequencyStats()

    # ── Public API ────────────────────────────────────────────────────────────

    def queue_move(self, move: Move) -> None:
        self._command_queue.put(("queue_move", move))

    def clear_move_queue(self) -> None:
        self._command_queue.put(("clear_queue", None))

    def set_speech_offsets(self, offsets: Tuple[float, float, float, float, float, float]) -> None:
        with self._speech_offsets_lock:
            self._pending_speech_offsets = offsets
            self._speech_offsets_dirty = True

    def set_moving_state(self, duration: float) -> None:
        self._command_queue.put(("set_moving_state", duration))

    def is_idle(self) -> bool:
        with self._shared_state_lock:
            last_activity = self._shared_last_activity_time
            listening = self._shared_is_listening
        if listening:
            return False
        return self._now() - last_activity >= self.idle_inactivity_delay

    def set_listening(self, listening: bool) -> None:
        with self._shared_state_lock:
            if self._shared_is_listening == listening:
                return
        self._command_queue.put(("set_listening", listening))

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            logger.warning("MovementManager already running; start() ignored")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.working_loop, daemon=True, name="movement-manager")
        self._thread.start()
        logger.info("MovementManager started at %.0fHz", self.target_frequency)

    def stop(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            return
        logger.info("Stopping MovementManager…")
        self.clear_move_queue()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        try:
            neutral = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
            self.current_robot.goto_target(head=neutral, antennas=[-0.1745, 0.1745], duration=1.0, body_yaw=0.0)
        except Exception as e:
            logger.debug("Reset to neutral failed: %s", e)

    def get_status(self) -> Dict[str, Any]:
        with self._status_lock:
            pose = clone_full_body_pose(self._last_commanded_pose)
            freq = self._freq_snapshot
        return {
            "queue_size": len(self.move_queue),
            "is_listening": self._is_listening,
            "breathing_active": self._breathing_active,
            "last_commanded_pose": {
                "head": pose[0].tolist(),
                "antennas": pose[1],
                "body_yaw": pose[2],
            },
            "loop_frequency": {
                "last": freq.last_freq,
                "mean": freq.mean,
                "min": freq.min_freq,
                "potential": freq.potential_freq,
                "samples": freq.count,
            },
        }

    # ── Internal: signal processing ───────────────────────────────────────────

    def _poll_signals(self, current_time: float) -> None:
        self._apply_pending_offsets()
        while True:
            try:
                command, payload = self._command_queue.get_nowait()
            except Empty:
                break
            self._handle_command(command, payload, current_time)

    def _apply_pending_offsets(self) -> None:
        with self._speech_offsets_lock:
            if self._speech_offsets_dirty:
                self.state.speech_offsets = self._pending_speech_offsets
                self._speech_offsets_dirty = False
                self.state.update_activity()

        with self._face_offsets_lock:
            if self._face_offsets_dirty:
                self.state.face_tracking_offsets = self._pending_face_offsets
                self._face_offsets_dirty = False
                self.state.update_activity()

    def _handle_command(self, command: str, payload: Any, current_time: float) -> None:
        if command == "queue_move":
            if isinstance(payload, Move):
                self.move_queue.append(payload)
                self.state.update_activity()
                try:
                    dur = float(payload.duration)
                    logger.debug("Queued move duration=%.2fs, queue=%d", dur, len(self.move_queue))
                except Exception:
                    pass
            else:
                logger.warning("queue_move: invalid payload %s", payload)
        elif command == "clear_queue":
            self.move_queue.clear()
            self.state.current_move = None
            self.state.move_start_time = None
            self._breathing_active = False
            logger.info("Move queue cleared")
        elif command == "set_moving_state":
            try:
                float(payload)
            except (TypeError, ValueError):
                logger.warning("set_moving_state: invalid payload %s", payload)
                return
            self.state.update_activity()
        elif command == "mark_activity":
            self.state.update_activity()
        elif command == "set_listening":
            desired = bool(payload)
            now = self._now()
            if now - self._last_listening_toggle_time < self._listening_debounce_s:
                return
            self._last_listening_toggle_time = now
            if self._is_listening == desired:
                return
            self._is_listening = desired
            self._last_listening_blend_time = now
            if desired:
                self._listening_antennas = (
                    float(self._last_commanded_pose[1][0]),
                    float(self._last_commanded_pose[1][1]),
                )
                self._antenna_unfreeze_blend = 0.0
            else:
                self._antenna_unfreeze_blend = 0.0
            self.state.update_activity()
        else:
            logger.warning("Unknown command: %s", command)

    # ── Internal: primary motion ──────────────────────────────────────────────

    def _update_primary_motion(self, current_time: float) -> None:
        self._manage_move_queue(current_time)
        self._manage_breathing(current_time)

    def _manage_move_queue(self, current_time: float) -> None:
        if self.state.current_move is None or (
            self.state.move_start_time is not None
            and current_time - self.state.move_start_time >= self.state.current_move.duration
        ):
            self.state.current_move = None
            self.state.move_start_time = None
            if self.move_queue:
                self.state.current_move = self.move_queue.popleft()
                self.state.move_start_time = current_time
                self._breathing_active = isinstance(self.state.current_move, BreathingMove)

    def _manage_breathing(self, current_time: float) -> None:
        if (
            self.state.current_move is None
            and not self.move_queue
            and not self._is_listening
            and not self._breathing_active
        ):
            idle_for = current_time - self.state.last_activity_time
            if idle_for >= self.idle_inactivity_delay:
                try:
                    _, current_antennas = self.current_robot.get_current_joint_positions()
                    current_head_pose = self.current_robot.get_current_head_pose()
                    self._breathing_active = True
                    self.state.update_activity()
                    self.move_queue.append(BreathingMove(
                        interpolation_start_pose=current_head_pose,
                        interpolation_start_antennas=current_antennas,
                        interpolation_duration=1.0,
                    ))
                except Exception as e:
                    self._breathing_active = False
                    logger.debug("Failed to start breathing: %s", e)

        if isinstance(self.state.current_move, BreathingMove) and self.move_queue:
            self.state.current_move = None
            self.state.move_start_time = None
            self._breathing_active = False

        if self.state.current_move is not None and not isinstance(self.state.current_move, BreathingMove):
            self._breathing_active = False

    # ── Internal: pose composition ────────────────────────────────────────────

    def _get_primary_pose(self, current_time: float) -> FullBodyPose:
        if self.state.current_move is not None and self.state.move_start_time is not None:
            t = current_time - self.state.move_start_time
            head, antennas, body_yaw = self.state.current_move.evaluate(t)
            if head is None:
                head = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
            if antennas is None:
                antennas = np.array([-0.1745, 0.1745])
            if body_yaw is None:
                body_yaw = 0.0
            pose: FullBodyPose = (head.copy(), (float(antennas[0]), float(antennas[1])), float(body_yaw))
            self.state.last_primary_pose = clone_full_body_pose(pose)
            return pose
        if self.state.last_primary_pose is not None:
            return clone_full_body_pose(self.state.last_primary_pose)
        neutral = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
        pose = (neutral, (0.0, 0.0), 0.0)
        self.state.last_primary_pose = clone_full_body_pose(pose)
        return pose

    def _get_secondary_pose(self) -> FullBodyPose:
        offsets = (
            self.state.speech_offsets[0] + self.state.face_tracking_offsets[0],
            self.state.speech_offsets[1] + self.state.face_tracking_offsets[1],
            self.state.speech_offsets[2] + self.state.face_tracking_offsets[2],
            self.state.speech_offsets[3] + self.state.face_tracking_offsets[3],
            self.state.speech_offsets[4] + self.state.face_tracking_offsets[4],
            self.state.speech_offsets[5] + self.state.face_tracking_offsets[5],
        )
        if offsets == self._cached_secondary_offsets:
            return self._cached_secondary_pose
        secondary_head = create_head_pose(
            x=offsets[0], y=offsets[1], z=offsets[2],
            roll=offsets[3], pitch=offsets[4], yaw=offsets[5],
            degrees=False, mm=False,
        )
        self._cached_secondary_offsets = offsets
        self._cached_secondary_pose = (secondary_head, (0.0, 0.0), 0.0)
        return self._cached_secondary_pose

    def _compose_full_body_pose(self, current_time: float) -> FullBodyPose:
        return combine_full_body(self._get_primary_pose(current_time), self._get_secondary_pose())

    def _update_face_tracking(self, current_time: float) -> None:
        if self.camera_worker is not None:
            try:
                offsets = self.camera_worker.get_face_tracking_offsets()
                self.state.face_tracking_offsets = offsets
            except Exception:
                self.state.face_tracking_offsets = (0.0,) * 6
        else:
            self.state.face_tracking_offsets = (0.0,) * 6

    # ── Internal: antenna blend ───────────────────────────────────────────────

    def _calculate_blended_antennas(self, target: Tuple[float, float]) -> Tuple[float, float]:
        now = self._now()
        if self._is_listening:
            self._antenna_unfreeze_blend = 0.0
            self._last_listening_blend_time = now
            return self._listening_antennas

        dt = max(0.0, now - self._last_listening_blend_time)
        self._last_listening_blend_time = now
        blend = min(1.0, self._antenna_unfreeze_blend + (dt / self._antenna_blend_duration if self._antenna_blend_duration > 0 else 1.0))
        self._antenna_unfreeze_blend = blend
        antennas = (
            self._listening_antennas[0] * (1.0 - blend) + target[0] * blend,
            self._listening_antennas[1] * (1.0 - blend) + target[1] * blend,
        )
        if blend >= 1.0:
            self._listening_antennas = (float(target[0]), float(target[1]))
        return antennas

    # ── Internal: robot output ────────────────────────────────────────────────

    def _issue_control_command(self, head: NDArray[np.float32], antennas: Tuple[float, float], body_yaw: float) -> None:
        try:
            self.current_robot.set_target(head=head, antennas=antennas, body_yaw=body_yaw)
            with self._status_lock:
                self._last_commanded_pose = clone_full_body_pose((head, antennas, body_yaw))
        except Exception as e:
            now = self._now()
            if now - self._last_set_target_err >= self._set_target_err_interval:
                suppressed = self._set_target_err_suppressed
                self._set_target_err_suppressed = 0
                msg = f"set_target failed: {e}"
                if suppressed:
                    msg += f" ({suppressed} repeats suppressed)"
                logger.error(msg)
                self._last_set_target_err = now
            else:
                self._set_target_err_suppressed += 1

    # ── Internal: telemetry ───────────────────────────────────────────────────

    def _update_frequency_stats(self, loop_start: float, prev_loop_start: float, stats: LoopFrequencyStats) -> LoopFrequencyStats:
        period = loop_start - prev_loop_start
        if period > 0:
            stats.last_freq = 1.0 / period
            stats.count += 1
            delta = stats.last_freq - stats.mean
            stats.mean += delta / stats.count
            stats.m2 += delta * (stats.last_freq - stats.mean)
            stats.min_freq = min(stats.min_freq, stats.last_freq)
        return stats

    def _schedule_next_tick(self, loop_start: float, stats: LoopFrequencyStats) -> Tuple[float, LoopFrequencyStats]:
        elapsed = self._now() - loop_start
        stats.potential_freq = 1.0 / elapsed if elapsed > 0 else float("inf")
        sleep_time = max(0.0, self.target_period - elapsed)
        return sleep_time, stats

    def _record_frequency_snapshot(self, stats: LoopFrequencyStats) -> None:
        with self._status_lock:
            self._freq_snapshot = LoopFrequencyStats(
                mean=stats.mean, m2=stats.m2, min_freq=stats.min_freq,
                count=stats.count, last_freq=stats.last_freq, potential_freq=stats.potential_freq,
            )

    def _maybe_log_frequency(self, loop_count: int, interval: int, stats: LoopFrequencyStats) -> None:
        if loop_count % interval != 0 or stats.count == 0:
            return
        variance = stats.m2 / stats.count if stats.count > 0 else 0.0
        logger.debug(
            "MovementManager freq — avg=%.1fHz min=%.1fHz last=%.1fHz potential=%.1fHz (target %.0fHz)",
            stats.mean, stats.min_freq if stats.min_freq != float("inf") else 0.0,
            stats.last_freq, stats.potential_freq, self.target_frequency,
        )
        stats.reset()

    def _publish_shared_state(self) -> None:
        with self._shared_state_lock:
            self._shared_last_activity_time = self.state.last_activity_time
            self._shared_is_listening = self._is_listening

    # ── Main loop ─────────────────────────────────────────────────────────────

    def working_loop(self) -> None:
        loop_count = 0
        print_interval = max(1, int(self.target_frequency * 10))  # log every 10 s
        prev_loop_start = self._now()
        stats = self._freq_stats

        while not self._stop_event.is_set():
            loop_start = self._now()
            loop_count += 1

            if loop_count > 1:
                stats = self._update_frequency_stats(loop_start, prev_loop_start, stats)
            prev_loop_start = loop_start

            self._poll_signals(loop_start)
            self._update_primary_motion(loop_start)
            self._update_face_tracking(loop_start)

            head, antennas, body_yaw = self._compose_full_body_pose(loop_start)
            antennas_cmd = self._calculate_blended_antennas(antennas)
            self._issue_control_command(head, antennas_cmd, body_yaw)

            sleep_time, stats = self._schedule_next_tick(loop_start, stats)
            self._publish_shared_state()
            self._record_frequency_snapshot(stats)
            self._maybe_log_frequency(loop_count, print_interval, stats)

            if sleep_time > 0:
                time.sleep(sleep_time)
