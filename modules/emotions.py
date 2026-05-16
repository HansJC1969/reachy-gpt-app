"""
Emotion and movement control for Reachy Mini.

Uses pollen-robotics MovementManager + RecordedMoves (SDK pre-recorded animations)
when a robot is connected. Falls back to legacy keyframe animations in sim mode.

Supported emotions (map to RecordedMoves names discovered at runtime):
    NEUTRAL, FREUDE, TRAUER, ANGST, MÜDE, NACHDENKEN,
    TANZEN, ÜBERRASCHUNG, NEUGIER
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from modules.moves import MovementManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Emotion catalogue
# ---------------------------------------------------------------------------

class Emotion(Enum):
    NEUTRAL       = "neutral"
    FREUDE        = "freude"
    TRAUER        = "trauer"
    ANGST         = "angst"
    MÜDE          = "müde"
    NACHDENKEN    = "nachdenken"
    TANZEN        = "tanzen"
    ÜBERRASCHUNG  = "ueberraschung"
    NEUGIER       = "neugier"


EMOTION_LABELS: dict[str, Emotion] = {e.value: e for e in Emotion}
EMOTION_LABELS.update({
    "joy": Emotion.FREUDE, "happy": Emotion.FREUDE, "happiness": Emotion.FREUDE,
    "sad": Emotion.TRAUER, "sadness": Emotion.TRAUER,
    "fear": Emotion.ANGST, "scared": Emotion.ANGST,
    "tired": Emotion.MÜDE, "sleepy": Emotion.MÜDE,
    "thinking": Emotion.NACHDENKEN, "ponder": Emotion.NACHDENKEN,
    "dance": Emotion.TANZEN, "dancing": Emotion.TANZEN,
    "surprise": Emotion.ÜBERRASCHUNG, "surprised": Emotion.ÜBERRASCHUNG,
    "curious": Emotion.NEUGIER, "curiosity": Emotion.NEUGIER,
})

# Candidate RecordedMoves names per emotion (tried in order; first match wins)
_RECORDED_CANDIDATES: dict[Emotion, list[str]] = {
    Emotion.FREUDE:       ["happy", "joy", "excited", "freude"],
    Emotion.TRAUER:       ["sad", "sadness", "unhappy", "trauer"],
    Emotion.ANGST:        ["scared", "fear", "anxious", "angst"],
    Emotion.MÜDE:         ["tired", "sleepy", "yawn", "müde"],
    Emotion.NACHDENKEN:   ["thinking", "ponder", "curious", "nachdenken"],
    Emotion.ÜBERRASCHUNG: ["surprised", "surprise", "shock", "ueberraschung"],
    Emotion.NEUGIER:      ["curious", "interested", "wonder", "neugier"],
    Emotion.NEUTRAL:      ["neutral"],
    Emotion.TANZEN:       ["happy", "excited"],   # fallback if no dance library
}


def parse_emotion(label: str) -> Emotion:
    """Parse a string to an Emotion, falling back to NEUTRAL."""
    return EMOTION_LABELS.get(label.lower().strip(), Emotion.NEUTRAL)


# ---------------------------------------------------------------------------
# Legacy keyframe data (used in simulation / fallback mode)
# ---------------------------------------------------------------------------

@dataclass
class Keyframe:
    t: float; pitch: float; yaw: float; roll: float; l_ant: float; r_ant: float
    body_yaw: float = 0.0


@dataclass
class Animation:
    keyframes: list[Keyframe]
    loop: bool = False
    loop_count: int = 0


def _coslerp(a: float, b: float, t: float) -> float:
    t2 = (1.0 - math.cos(t * math.pi)) / 2.0
    return a + (b - a) * t2


def _interpolate(kfs: list[Keyframe], t: float) -> Keyframe:
    if t <= kfs[0].t:
        return kfs[0]
    if t >= kfs[-1].t:
        return kfs[-1]
    for i in range(len(kfs) - 1):
        k0, k1 = kfs[i], kfs[i + 1]
        if k0.t <= t <= k1.t:
            span = k1.t - k0.t
            a = (t - k0.t) / span if span > 0 else 1.0
            return Keyframe(
                t=t,
                pitch=_coslerp(k0.pitch, k1.pitch, a),
                yaw=_coslerp(k0.yaw, k1.yaw, a),
                roll=_coslerp(k0.roll, k1.roll, a),
                l_ant=_coslerp(k0.l_ant, k1.l_ant, a),
                r_ant=_coslerp(k0.r_ant, k1.r_ant, a),
                body_yaw=_coslerp(k0.body_yaw, k1.body_yaw, a),
            )
    return kfs[-1]


NEUTRAL_POSE = Keyframe(t=0.0, pitch=0, yaw=0, roll=0, l_ant=0, r_ant=0)
SLEEP_POSE   = Keyframe(t=0.0, pitch=-18, yaw=0, roll=20, l_ant=-38, r_ant=-38)

_DROOP_TO_SLEEP = Animation(keyframes=[
    Keyframe(t=0.0, pitch=0,   yaw=0, roll=0,  l_ant=0,   r_ant=0),
    Keyframe(t=2.0, pitch=-8,  yaw=0, roll=10, l_ant=-15, r_ant=-15),
    Keyframe(t=4.5, pitch=-16, yaw=0, roll=18, l_ant=-32, r_ant=-32),
    Keyframe(t=6.0, pitch=-18, yaw=0, roll=20, l_ant=-38, r_ant=-38),
])

_LEGACY_ANIMATIONS: dict[Emotion, Animation] = {
    Emotion.NEUTRAL: Animation(keyframes=[
        Keyframe(t=0.0, pitch=0, yaw=0, roll=0, l_ant=0, r_ant=0),
        Keyframe(t=0.6, pitch=0, yaw=0, roll=0, l_ant=0, r_ant=0),
    ]),
    Emotion.FREUDE: Animation(keyframes=[
        Keyframe(t=0.0, pitch=0,  yaw=0,  roll=0,  l_ant=0,  r_ant=0),
        Keyframe(t=0.2, pitch=12, yaw=8,  roll=5,  l_ant=50, r_ant=50),
        Keyframe(t=0.4, pitch=-3, yaw=-5, roll=-3, l_ant=20, r_ant=20),
        Keyframe(t=0.6, pitch=12, yaw=8,  roll=5,  l_ant=55, r_ant=40),
        Keyframe(t=0.8, pitch=-3, yaw=-5, roll=-3, l_ant=25, r_ant=25),
        Keyframe(t=1.0, pitch=10, yaw=5,  roll=3,  l_ant=45, r_ant=55),
        Keyframe(t=1.5, pitch=5,  yaw=0,  roll=0,  l_ant=35, r_ant=35),
        Keyframe(t=2.0, pitch=0,  yaw=0,  roll=0,  l_ant=0,  r_ant=0),
    ]),
    Emotion.TRAUER: Animation(keyframes=[
        Keyframe(t=0.0, pitch=0,   yaw=0,  roll=0,  l_ant=0,   r_ant=0),
        Keyframe(t=2.5, pitch=-14, yaw=-4, roll=10, l_ant=-40, r_ant=-40),
        Keyframe(t=5.5, pitch=-14, yaw=-2, roll=8,  l_ant=-40, r_ant=-40),
        Keyframe(t=7.0, pitch=-16, yaw=-4, roll=10, l_ant=-45, r_ant=-45),
    ]),
    Emotion.ANGST: Animation(keyframes=[
        Keyframe(t=0.0,  pitch=-3, yaw=0,   roll=0, l_ant=-15, r_ant=-15),
        Keyframe(t=0.10, pitch=-4, yaw=-12, roll=0, l_ant=-25, r_ant=-30),
        Keyframe(t=0.20, pitch=-4, yaw=12,  roll=0, l_ant=-30, r_ant=-25),
        Keyframe(t=0.30, pitch=-4, yaw=-10, roll=0, l_ant=-28, r_ant=-32),
        Keyframe(t=0.40, pitch=-4, yaw=10,  roll=0, l_ant=-32, r_ant=-28),
        Keyframe(t=0.85, pitch=-3, yaw=0,   roll=0, l_ant=-20, r_ant=-20),
        Keyframe(t=2.2,  pitch=0,  yaw=0,   roll=0, l_ant=0,   r_ant=0),
    ]),
    Emotion.MÜDE: Animation(keyframes=[
        Keyframe(t=0.0, pitch=0,   yaw=0, roll=0,  l_ant=0,   r_ant=0),
        Keyframe(t=3.0, pitch=-10, yaw=0, roll=14, l_ant=-20, r_ant=-20),
        Keyframe(t=6.5, pitch=-18, yaw=0, roll=20, l_ant=-38, r_ant=-38),
        Keyframe(t=7.0, pitch=-4,  yaw=0, roll=4,  l_ant=8,   r_ant=8),
        Keyframe(t=9.5, pitch=-18, yaw=0, roll=20, l_ant=-38, r_ant=-38),
    ]),
    Emotion.NACHDENKEN: Animation(keyframes=[
        Keyframe(t=0.0, pitch=0, yaw=0,  roll=0,  l_ant=0,  r_ant=0),
        Keyframe(t=0.7, pitch=6, yaw=6,  roll=14, l_ant=45, r_ant=5),
        Keyframe(t=2.6, pitch=5, yaw=-4, roll=10, l_ant=45, r_ant=15),
        Keyframe(t=4.5, pitch=6, yaw=6,  roll=14, l_ant=55, r_ant=5),
        Keyframe(t=5.5, pitch=0, yaw=0,  roll=0,  l_ant=0,  r_ant=0),
    ]),
    Emotion.TANZEN: Animation(loop=True, loop_count=4, keyframes=[
        Keyframe(t=0.0,  pitch=2, yaw=0,   roll=0,   l_ant=10,  r_ant=10,  body_yaw=0.0),
        Keyframe(t=0.75, pitch=5, yaw=20,  roll=12,  l_ant=45,  r_ant=-10, body_yaw=8.0),
        Keyframe(t=1.5,  pitch=2, yaw=0,   roll=0,   l_ant=10,  r_ant=10,  body_yaw=0.0),
        Keyframe(t=2.25, pitch=5, yaw=-20, roll=-12, l_ant=-10, r_ant=45,  body_yaw=-8.0),
        Keyframe(t=3.0,  pitch=2, yaw=0,   roll=0,   l_ant=10,  r_ant=10,  body_yaw=0.0),
    ]),
    Emotion.ÜBERRASCHUNG: Animation(keyframes=[
        Keyframe(t=0.0,  pitch=0,  yaw=0, roll=0, l_ant=0,  r_ant=0),
        Keyframe(t=0.08, pitch=18, yaw=0, roll=0, l_ant=60, r_ant=60),
        Keyframe(t=0.25, pitch=20, yaw=0, roll=0, l_ant=65, r_ant=65),
        Keyframe(t=1.2,  pitch=8,  yaw=0, roll=0, l_ant=25, r_ant=25),
        Keyframe(t=2.0,  pitch=0,  yaw=0, roll=0, l_ant=0,  r_ant=0),
    ]),
    Emotion.NEUGIER: Animation(keyframes=[
        Keyframe(t=0.0, pitch=0,  yaw=0,  roll=0,  l_ant=0,  r_ant=0),
        Keyframe(t=0.6, pitch=10, yaw=12, roll=10, l_ant=25, r_ant=25),
        Keyframe(t=2.3, pitch=10, yaw=-6, roll=-5, l_ant=20, r_ant=30),
        Keyframe(t=4.0, pitch=10, yaw=12, roll=10, l_ant=25, r_ant=25),
        Keyframe(t=5.0, pitch=0,  yaw=0,  roll=0,  l_ant=0,  r_ant=0),
    ]),
}

_HEAD_MOVE_ANIMATIONS: dict[str, Animation] = {
    "left":  Animation(keyframes=[Keyframe(t=0.0, pitch=0, yaw=-30, roll=0, l_ant=0, r_ant=0), Keyframe(t=1.2, pitch=0, yaw=-30, roll=0, l_ant=0, r_ant=0), Keyframe(t=2.0, pitch=0, yaw=0, roll=0, l_ant=0, r_ant=0)]),
    "right": Animation(keyframes=[Keyframe(t=0.0, pitch=0, yaw=30,  roll=0, l_ant=0, r_ant=0), Keyframe(t=1.2, pitch=0, yaw=30,  roll=0, l_ant=0, r_ant=0), Keyframe(t=2.0, pitch=0, yaw=0, roll=0, l_ant=0, r_ant=0)]),
    "up":    Animation(keyframes=[Keyframe(t=0.0, pitch=15, yaw=0, roll=0, l_ant=15, r_ant=15), Keyframe(t=1.2, pitch=15, yaw=0, roll=0, l_ant=15, r_ant=15), Keyframe(t=2.0, pitch=0, yaw=0, roll=0, l_ant=0, r_ant=0)]),
    "down":  Animation(keyframes=[Keyframe(t=0.0, pitch=-12, yaw=0, roll=0, l_ant=-10, r_ant=-10), Keyframe(t=1.2, pitch=-12, yaw=0, roll=0, l_ant=-10, r_ant=-10), Keyframe(t=2.0, pitch=0, yaw=0, roll=0, l_ant=0, r_ant=0)]),
    "front": Animation(keyframes=[Keyframe(t=0.0, pitch=0, yaw=0, roll=0, l_ant=0, r_ant=0), Keyframe(t=0.6, pitch=0, yaw=0, roll=0, l_ant=0, r_ant=0)]),
}

_TICK = 0.04   # legacy keyframe tick rate (25 Hz)


# ---------------------------------------------------------------------------
# EmotionEngine
# ---------------------------------------------------------------------------

class EmotionEngine:
    """
    Play emotions via MovementManager + RecordedMoves (robot mode)
    or legacy keyframe animations (simulation / fallback).
    """

    def __init__(self, reachy=None, movement_manager: Optional["MovementManager"] = None) -> None:
        self.reachy = reachy
        self.movement_manager = movement_manager

        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._current_emotion = Emotion.NEUTRAL

        # Try to load SDK animation libraries (only available on the robot)
        self._recorded_moves = None
        self._available_emotions: list[str] = []
        self._dance_available = False
        self._available_dances: list[str] = []

        if reachy is not None:
            self._try_load_recorded_moves()
            self._try_load_dance_library()

    def _try_load_recorded_moves(self) -> None:
        try:
            from reachy_mini.motion.recorded_move import RecordedMoves
            self._recorded_moves = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
            self._available_emotions = list(self._recorded_moves.list_moves())
            logger.info("RecordedMoves loaded — available: %s", self._available_emotions)
        except Exception as e:
            logger.warning("RecordedMoves unavailable (%s) — using keyframe fallback", e)

    def _try_load_dance_library(self) -> None:
        try:
            from reachy_mini_dances_library.collection.dance import AVAILABLE_MOVES
            self._available_dances = list(AVAILABLE_MOVES.keys())
            self._dance_available = bool(self._available_dances)
            if self._dance_available:
                logger.info("Dance library loaded — available: %s", self._available_dances)
        except Exception as e:
            logger.warning("Dance library unavailable (%s)", e)

    def _map_to_recorded(self, emotion: Emotion) -> Optional[str]:
        """Return the first matching RecordedMoves name for this emotion, or None."""
        for candidate in _RECORDED_CANDIDATES.get(emotion, []):
            if candidate in self._available_emotions:
                return candidate
        if emotion.value in self._available_emotions:
            return emotion.value
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def current_emotion(self) -> Emotion:
        with self._lock:
            return self._current_emotion

    def _set_emotion(self, emotion: Emotion) -> None:
        with self._lock:
            self._current_emotion = emotion

    def play(self, emotion: Emotion, *, block: bool = False) -> None:
        """Play an emotion animation. Non-blocking unless block=True."""
        self._set_emotion(emotion)
        logger.info("[emotion] %s", emotion.value)

        if self.movement_manager is not None and self.reachy is not None:
            self._play_via_manager(emotion, block=block)
        else:
            self._play_legacy(emotion, block=block)

    def _play_via_manager(self, emotion: Emotion, *, block: bool) -> None:
        """Queue the emotion via MovementManager using SDK pre-recorded animations."""
        from modules.dance_emotion_moves import EmotionQueueMove, DanceQueueMove

        if emotion == Emotion.TANZEN and self._dance_available:
            import random
            from reachy_mini_dances_library.collection.dance import AVAILABLE_MOVES
            name = random.choice(self._available_dances)
            move = DanceQueueMove(name)
            self.movement_manager.queue_move(move)
            if block:
                time.sleep(move.duration)
            return

        if self._recorded_moves is not None:
            name = self._map_to_recorded(emotion)
            if name is not None:
                move = EmotionQueueMove(name, self._recorded_moves)
                self.movement_manager.queue_move(move)
                if block:
                    dur = move.duration
                    if dur < float("inf"):
                        time.sleep(dur)
                return

        # Fallback: no RecordedMoves for this emotion → use legacy keyframes
        # Legacy keyframes call set_target directly; pause movement_manager?
        # We skip the direct SDK calls in sim-less legacy mode for safety.
        logger.debug("[emotion] no RecordedMoves for %s — skipping (manager mode)", emotion.value)

    def _play_legacy(self, emotion: Emotion, *, block: bool) -> None:
        """Play via old keyframe animations (sim mode or fallback)."""
        self._cancel()
        self._stop_evt.clear()
        anim = _LEGACY_ANIMATIONS.get(emotion, _LEGACY_ANIMATIONS[Emotion.NEUTRAL])
        self._thread = threading.Thread(
            target=self._run_legacy, args=(anim,),
            name=f"emotion-{emotion.value}", daemon=True,
        )
        self._thread.start()
        if block:
            self._thread.join()

    def stop(self) -> None:
        """Stop current animation and return to neutral."""
        self._cancel()
        self._apply_legacy(NEUTRAL_POSE)
        self._set_emotion(Emotion.NEUTRAL)

    def sleep_mode(self) -> None:
        """Droop head/antennas to sleep pose over ~6 s, then hold."""
        self._set_emotion(Emotion.MÜDE)
        self._cancel()
        self._stop_evt.clear()
        logger.info("[emotion] entering sleep mode")

        if self.movement_manager is not None and self.reachy is not None:
            self._sleep_via_manager()
        else:
            self._sleep_legacy()

    def _sleep_via_manager(self) -> None:
        """Queue droop + hold moves via MovementManager."""
        try:
            from reachy_mini.utils import create_head_pose
            from modules.dance_emotion_moves import GotoQueueMove

            sleep_head = create_head_pose(pitch=-18, yaw=0, roll=20, degrees=True)
            sleep_antennas_rad = math.radians(-38)
            droop = GotoQueueMove(
                target_head_pose=sleep_head,
                target_antennas=(sleep_antennas_rad, sleep_antennas_rad),
                target_body_yaw=0.0,
                duration=6.0,
            )
            self.movement_manager.queue_move(droop)
            # Block so caller knows the droop has finished before starting the wake loop
            time.sleep(6.5)
        except Exception as e:
            logger.warning("sleep_via_manager failed: %s", e)
            self._sleep_legacy()

    def _sleep_legacy(self) -> None:
        """Keyframe droop to sleep, then hold in background thread."""
        t_start = time.monotonic()
        duration = _DROOP_TO_SLEEP.keyframes[-1].t
        while not self._stop_evt.is_set():
            elapsed = time.monotonic() - t_start
            if elapsed >= duration:
                break
            self._apply_legacy(_interpolate(_DROOP_TO_SLEEP.keyframes, elapsed))
            time.sleep(_TICK)
        if self._stop_evt.is_set():
            return
        self._apply_legacy(SLEEP_POSE)
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._hold_static_legacy, args=(SLEEP_POSE,),
            daemon=True, name="emotion-sleep-hold",
        )
        self._thread.start()

    def idle(self) -> None:
        """Gently return to neutral when no face is detected."""
        if self.current_emotion != Emotion.NEUTRAL:
            self.play(Emotion.NEUTRAL)

    def move_head(self, direction: str) -> None:
        """Move head to a named direction then return to neutral."""
        if self.movement_manager is not None and self.reachy is not None:
            self._move_head_via_manager(direction)
        else:
            anim = _HEAD_MOVE_ANIMATIONS.get(direction.lower())
            if anim is None:
                logger.warning("move_head: unknown direction %r", direction)
                return
            self._cancel()
            self._stop_evt.clear()
            self._thread = threading.Thread(
                target=self._run_legacy, args=(anim,),
                name=f"head-{direction}", daemon=True,
            )
            self._thread.start()

    def _move_head_via_manager(self, direction: str) -> None:
        try:
            from reachy_mini.utils import create_head_pose
            from modules.dance_emotion_moves import GotoQueueMove

            _DELTAS = {
                "left":  (0, 0, 0, 0, 0,  40),
                "right": (0, 0, 0, 0, 0, -40),
                "up":    (0, 0, 0, 0, -30, 0),
                "down":  (0, 0, 0, 0,  30, 0),
                "front": (0, 0, 0, 0,   0, 0),
            }
            deltas = _DELTAS.get(direction.lower(), (0, 0, 0, 0, 0, 0))
            target = create_head_pose(*deltas, degrees=True)
            current_head = self.reachy.get_current_head_pose()
            _, current_ants = self.reachy.get_current_joint_positions()
            hold = GotoQueueMove(target_head_pose=target, start_head_pose=current_head,
                                 target_antennas=(0, 0), start_antennas=(current_ants[0], current_ants[1]),
                                 duration=0.8)
            back = GotoQueueMove(target_head_pose=create_head_pose(0, 0, 0, 0, 0, 0, degrees=True),
                                 target_antennas=(0, 0), duration=0.8)
            self.movement_manager.queue_move(hold)
            self.movement_manager.queue_move(back)
        except Exception as e:
            logger.warning("move_head_via_manager failed: %s", e)

    # ------------------------------------------------------------------
    # Legacy keyframe helpers
    # ------------------------------------------------------------------

    def _cancel(self) -> None:
        self._stop_evt.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)

    def _hold_static_legacy(self, pose: Keyframe) -> None:
        while not self._stop_evt.is_set():
            self._apply_legacy(pose)
            time.sleep(_TICK * 5)

    def _run_legacy(self, anim: Animation) -> None:
        uses_body = any(abs(kf.body_yaw) >= 0.1 for kf in anim.keyframes)
        max_repeats = anim.loop_count if (anim.loop and anim.loop_count > 0) else (999 if anim.loop else 1)
        duration = anim.keyframes[-1].t
        repeats = 0

        while repeats < max_repeats and not self._stop_evt.is_set():
            t_start = time.monotonic()
            while not self._stop_evt.is_set():
                elapsed = time.monotonic() - t_start
                if elapsed >= duration:
                    break
                self._apply_legacy(_interpolate(anim.keyframes, elapsed), send_body=uses_body)
                time.sleep(_TICK)
            if not self._stop_evt.is_set():
                self._apply_legacy(anim.keyframes[-1], send_body=uses_body)
            repeats += 1

        if not self._stop_evt.is_set():
            if uses_body:
                self._reset_body_yaw_legacy()
            self._glide_to_neutral_legacy()
            self._set_emotion(Emotion.NEUTRAL)

    def _glide_to_neutral_legacy(self, duration: float = 0.8) -> None:
        glide = Animation(keyframes=[NEUTRAL_POSE, Keyframe(t=duration, pitch=0, yaw=0, roll=0, l_ant=0, r_ant=0)])
        t_start = time.monotonic()
        while not self._stop_evt.is_set():
            elapsed = time.monotonic() - t_start
            if elapsed >= duration:
                break
            self._apply_legacy(_interpolate(glide.keyframes, elapsed))
            time.sleep(_TICK)
        self._apply_legacy(NEUTRAL_POSE)

    def _apply_legacy(self, pose: Keyframe, *, send_body: bool = False) -> None:
        if self.reachy is None:
            logger.debug("[sim] pitch=%+.1f yaw=%+.1f roll=%+.1f l=%+.1f r=%+.1f",
                         pose.pitch, pose.yaw, pose.roll, pose.l_ant, pose.r_ant)
            return
        try:
            from reachy_mini.utils import create_head_pose
            head_pose = create_head_pose(pitch=pose.pitch, yaw=pose.yaw, roll=pose.roll, degrees=True)
            antennas = np.deg2rad([pose.r_ant, pose.l_ant])
            self.reachy.set_target(head=head_pose, antennas=antennas)
        except Exception:
            logger.debug("Head/antenna command failed", exc_info=True)
        if send_body:
            try:
                self.reachy.set_target_body_yaw(math.radians(pose.body_yaw))
            except AttributeError:
                try:
                    self.reachy.goto_target(body_yaw=math.radians(pose.body_yaw), duration=_TICK * 2)
                except Exception:
                    logger.debug("Body yaw command failed", exc_info=True)
            except Exception:
                logger.debug("Body yaw command failed", exc_info=True)

    def _reset_body_yaw_legacy(self, duration: float = 0.8) -> None:
        if self.reachy is None:
            return
        try:
            self.reachy.goto_target(body_yaw=0.0, duration=duration)
        except Exception:
            logger.debug("Body yaw reset failed", exc_info=True)


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)-8s %(message)s")
    parser = argparse.ArgumentParser(description="Emotion animation demo (no robot)")
    parser.add_argument("--demo", action="store_true", default=True)
    parser.add_argument("--emotion", default=None)
    args = parser.parse_args()

    engine = EmotionEngine(reachy=None)

    if args.emotion:
        emo = parse_emotion(args.emotion)
        print(f"Playing: {emo.value}")
        engine.play(emo, block=True)
    else:
        for emo in Emotion:
            print(f"\n▶  {emo.value}")
            engine.play(emo, block=True)
            time.sleep(0.5)

    print("\nDemo complete.")
