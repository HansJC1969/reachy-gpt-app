"""
Emotional expression for Reachy Mini.

Drives neck (yaw / pitch / roll) and both antennas through keyframe animations.
Each emotion plays in a background daemon thread, so the caller never blocks
unless block=True is passed to EmotionEngine.play().

Standalone demo (no robot):
    python -m modules.emotions --demo
    python -m modules.emotions --demo --emotion tanzen
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Emotion catalogue
# ---------------------------------------------------------------------------

class Emotion(Enum):
    NEUTRAL       = "neutral"
    FREUDE        = "freude"          # joy / happiness
    TRAUER        = "trauer"          # sadness
    ANGST         = "angst"           # fear
    MÜDE          = "müde"            # tired
    NACHDENKEN    = "nachdenken"      # thinking / pondering
    TANZEN        = "tanzen"          # dancing
    ÜBERRASCHUNG  = "ueberraschung"   # surprise
    NEUGIER       = "neugier"         # curiosity


# Map GPT label strings → Emotion (for flexible matching)
EMOTION_LABELS: dict[str, Emotion] = {e.value: e for e in Emotion}
EMOTION_LABELS.update({
    "joy": Emotion.FREUDE,
    "happy": Emotion.FREUDE,
    "happiness": Emotion.FREUDE,
    "sad": Emotion.TRAUER,
    "sadness": Emotion.TRAUER,
    "fear": Emotion.ANGST,
    "scared": Emotion.ANGST,
    "tired": Emotion.MÜDE,
    "sleepy": Emotion.MÜDE,
    "thinking": Emotion.NACHDENKEN,
    "ponder": Emotion.NACHDENKEN,
    "dance": Emotion.TANZEN,
    "dancing": Emotion.TANZEN,
    "surprise": Emotion.ÜBERRASCHUNG,
    "surprised": Emotion.ÜBERRASCHUNG,
    "curious": Emotion.NEUGIER,
    "curiosity": Emotion.NEUGIER,
})


def parse_emotion(label: str) -> Emotion:
    """Parse a string to an Emotion, falling back to NEUTRAL."""
    return EMOTION_LABELS.get(label.lower().strip(), Emotion.NEUTRAL)


# ---------------------------------------------------------------------------
# Keyframe data model
# ---------------------------------------------------------------------------

@dataclass
class Keyframe:
    """A single pose snapshot in an animation."""
    t:      float   # time offset in seconds from animation start
    pitch:  float   # neck pitch  (°) — positive = head up
    yaw:    float   # neck yaw    (°) — positive = head right
    roll:   float   # neck roll   (°) — positive = tilt right
    l_ant:  float   # left  antenna (°)
    r_ant:  float   # right antenna (°)


@dataclass
class Animation:
    keyframes: list[Keyframe]
    loop: bool = False       # repeat keyframes indefinitely
    loop_count: int = 0      # 0 = infinite when loop=True


def _coslerp(a: float, b: float, t: float) -> float:
    """Cosine-smoothed interpolation — nicer than raw linear."""
    t2 = (1.0 - math.cos(t * math.pi)) / 2.0
    return a + (b - a) * t2


def interpolate(kfs: list[Keyframe], t: float) -> Keyframe:
    """Return interpolated pose at time *t* (clamped to animation bounds)."""
    if t <= kfs[0].t:
        return kfs[0]
    if t >= kfs[-1].t:
        return kfs[-1]

    for i in range(len(kfs) - 1):
        k0, k1 = kfs[i], kfs[i + 1]
        if k0.t <= t <= k1.t:
            span = k1.t - k0.t
            alpha = (t - k0.t) / span if span > 0 else 1.0
            return Keyframe(
                t=t,
                pitch=_coslerp(k0.pitch, k1.pitch, alpha),
                yaw=  _coslerp(k0.yaw,   k1.yaw,   alpha),
                roll= _coslerp(k0.roll,  k1.roll,  alpha),
                l_ant=_coslerp(k0.l_ant, k1.l_ant, alpha),
                r_ant=_coslerp(k0.r_ant, k1.r_ant, alpha),
            )
    return kfs[-1]


# ---------------------------------------------------------------------------
# Animation library
# ---------------------------------------------------------------------------
#
# Antenna convention used here:
#   0°  = resting / horizontal
#  +60° = raised high (excited)
#  -45° = drooped down (sad/scared)
#
# Neck convention (reachy-mini / create_head_pose):
#   pitch: +20 = head up,  -20 = head down
#   yaw:   +30 = head right, -30 = head left
#   roll:  +15 = tilt right, -15 = tilt left

NEUTRAL_POSE = Keyframe(t=0.0, pitch=0, yaw=0, roll=0, l_ant=0, r_ant=0)

ANIMATIONS: dict[Emotion, Animation] = {

    Emotion.NEUTRAL: Animation(keyframes=[
        Keyframe(t=0.0, pitch=0,   yaw=0,   roll=0,   l_ant=0,   r_ant=0),
        Keyframe(t=0.6, pitch=0,   yaw=0,   roll=0,   l_ant=0,   r_ant=0),
    ]),

    # -----------------------------------------------------------------------
    # FREUDE — bouncy head bobs, antennas spring up and wiggle
    Emotion.FREUDE: Animation(keyframes=[
        Keyframe(t=0.0, pitch=0,   yaw=0,   roll=0,   l_ant=0,   r_ant=0),
        Keyframe(t=0.2, pitch=12,  yaw=8,   roll=5,   l_ant=50,  r_ant=50),
        Keyframe(t=0.4, pitch=-3,  yaw=-5,  roll=-3,  l_ant=20,  r_ant=20),
        Keyframe(t=0.6, pitch=12,  yaw=8,   roll=5,   l_ant=55,  r_ant=40),
        Keyframe(t=0.8, pitch=-3,  yaw=-5,  roll=-3,  l_ant=25,  r_ant=25),
        Keyframe(t=1.0, pitch=10,  yaw=5,   roll=3,   l_ant=45,  r_ant=55),
        Keyframe(t=1.2, pitch=-3,  yaw=-3,  roll=-2,  l_ant=20,  r_ant=20),
        Keyframe(t=1.5, pitch=5,   yaw=0,   roll=0,   l_ant=35,  r_ant=35),
        Keyframe(t=2.0, pitch=0,   yaw=0,   roll=0,   l_ant=0,   r_ant=0),
    ]),

    # -----------------------------------------------------------------------
    # TRAUER — slow droop forward, antennas hang down, slight head tilt
    Emotion.TRAUER: Animation(keyframes=[
        Keyframe(t=0.0, pitch=0,   yaw=0,   roll=0,   l_ant=0,   r_ant=0),
        Keyframe(t=1.2, pitch=-8,  yaw=-2,  roll=6,   l_ant=-25, r_ant=-25),
        Keyframe(t=2.5, pitch=-14, yaw=-4,  roll=10,  l_ant=-40, r_ant=-40),
        Keyframe(t=4.0, pitch=-16, yaw=-4,  roll=10,  l_ant=-45, r_ant=-45),
        Keyframe(t=5.5, pitch=-14, yaw=-2,  roll=8,   l_ant=-40, r_ant=-40),
        Keyframe(t=7.0, pitch=-16, yaw=-4,  roll=10,  l_ant=-45, r_ant=-45),
    ]),

    # -----------------------------------------------------------------------
    # ANGST — fast head shaking, both antennas pressed down/trembling
    Emotion.ANGST: Animation(keyframes=[
        Keyframe(t=0.0, pitch=-3,  yaw=0,   roll=0,   l_ant=-15, r_ant=-15),
        Keyframe(t=0.10, pitch=-4, yaw=-12, roll=0,   l_ant=-25, r_ant=-30),
        Keyframe(t=0.20, pitch=-4, yaw=12,  roll=0,   l_ant=-30, r_ant=-25),
        Keyframe(t=0.30, pitch=-4, yaw=-10, roll=0,   l_ant=-28, r_ant=-32),
        Keyframe(t=0.40, pitch=-4, yaw=10,  roll=0,   l_ant=-32, r_ant=-28),
        Keyframe(t=0.50, pitch=-4, yaw=-8,  roll=0,   l_ant=-30, r_ant=-30),
        Keyframe(t=0.60, pitch=-4, yaw=8,   roll=0,   l_ant=-30, r_ant=-30),
        Keyframe(t=0.70, pitch=-4, yaw=-5,  roll=0,   l_ant=-25, r_ant=-25),
        Keyframe(t=0.85, pitch=-3, yaw=0,   roll=0,   l_ant=-20, r_ant=-20),
        Keyframe(t=1.5,  pitch=-3, yaw=0,   roll=0,   l_ant=-20, r_ant=-20),
        Keyframe(t=2.2,  pitch=0,  yaw=0,   roll=0,   l_ant=0,   r_ant=0),
    ]),

    # -----------------------------------------------------------------------
    # MÜDE — very slow droop; a small involuntary "jerk awake" mid-way
    Emotion.MÜDE: Animation(keyframes=[
        Keyframe(t=0.0, pitch=0,   yaw=0,  roll=0,   l_ant=0,   r_ant=0),
        Keyframe(t=1.5, pitch=-5,  yaw=0,  roll=8,   l_ant=-10, r_ant=-10),
        Keyframe(t=3.0, pitch=-10, yaw=0,  roll=14,  l_ant=-20, r_ant=-20),
        Keyframe(t=5.0, pitch=-16, yaw=0,  roll=18,  l_ant=-32, r_ant=-32),
        Keyframe(t=6.5, pitch=-18, yaw=0,  roll=20,  l_ant=-38, r_ant=-38),
        # jerk awake
        Keyframe(t=7.0, pitch=-4,  yaw=0,  roll=4,   l_ant=8,   r_ant=8),
        # droop again
        Keyframe(t=8.5, pitch=-16, yaw=0,  roll=18,  l_ant=-32, r_ant=-32),
        Keyframe(t=9.5, pitch=-18, yaw=0,  roll=20,  l_ant=-38, r_ant=-38),
    ]),

    # -----------------------------------------------------------------------
    # NACHDENKEN — head tilts right + up, left antenna raised, small sways
    Emotion.NACHDENKEN: Animation(keyframes=[
        Keyframe(t=0.0, pitch=0,  yaw=0,   roll=0,   l_ant=0,  r_ant=0),
        Keyframe(t=0.7, pitch=6,  yaw=6,   roll=14,  l_ant=45, r_ant=5),
        Keyframe(t=2.0, pitch=6,  yaw=6,   roll=14,  l_ant=45, r_ant=5),
        # slight re-consideration sway
        Keyframe(t=2.6, pitch=5,  yaw=-4,  roll=10,  l_ant=45, r_ant=15),
        Keyframe(t=3.2, pitch=6,  yaw=6,   roll=14,  l_ant=45, r_ant=5),
        Keyframe(t=4.2, pitch=6,  yaw=6,   roll=14,  l_ant=45, r_ant=5),
        # small antenna tap
        Keyframe(t=4.5, pitch=6,  yaw=6,   roll=14,  l_ant=55, r_ant=5),
        Keyframe(t=4.8, pitch=6,  yaw=6,   roll=14,  l_ant=45, r_ant=5),
        Keyframe(t=5.5, pitch=0,  yaw=0,   roll=0,   l_ant=0,  r_ant=0),
    ]),

    # -----------------------------------------------------------------------
    # TANZEN — rhythmic left/right head swings, antennas counter-sway
    Emotion.TANZEN: Animation(loop=True, loop_count=4, keyframes=[
        Keyframe(t=0.0, pitch=4,  yaw=0,   roll=0,   l_ant=0,   r_ant=0),
        Keyframe(t=0.3, pitch=6,  yaw=18,  roll=12,  l_ant=40,  r_ant=-15),
        Keyframe(t=0.6, pitch=4,  yaw=0,   roll=0,   l_ant=10,  r_ant=10),
        Keyframe(t=0.9, pitch=6,  yaw=-18, roll=-12, l_ant=-15, r_ant=40),
        Keyframe(t=1.2, pitch=4,  yaw=0,   roll=0,   l_ant=10,  r_ant=10),
    ]),

    # -----------------------------------------------------------------------
    # ÜBERRASCHUNG — sharp snap back, antennas shoot up, then settle
    Emotion.ÜBERRASCHUNG: Animation(keyframes=[
        Keyframe(t=0.0,  pitch=0,   yaw=0,  roll=0,  l_ant=0,   r_ant=0),
        Keyframe(t=0.08, pitch=18,  yaw=0,  roll=0,  l_ant=60,  r_ant=60),
        Keyframe(t=0.25, pitch=20,  yaw=0,  roll=0,  l_ant=65,  r_ant=65),
        Keyframe(t=0.6,  pitch=15,  yaw=0,  roll=0,  l_ant=50,  r_ant=50),
        Keyframe(t=1.2,  pitch=8,   yaw=0,  roll=0,  l_ant=25,  r_ant=25),
        Keyframe(t=2.0,  pitch=0,   yaw=0,  roll=0,  l_ant=0,   r_ant=0),
    ]),

    # -----------------------------------------------------------------------
    # NEUGIER — head leans forward+sideways, antennas perk up equally
    Emotion.NEUGIER: Animation(keyframes=[
        Keyframe(t=0.0, pitch=0,   yaw=0,   roll=0,   l_ant=0,   r_ant=0),
        Keyframe(t=0.6, pitch=10,  yaw=12,  roll=10,  l_ant=25,  r_ant=25),
        Keyframe(t=1.8, pitch=10,  yaw=12,  roll=10,  l_ant=25,  r_ant=25),
        # tilt other way briefly — comparing
        Keyframe(t=2.3, pitch=10,  yaw=-6,  roll=-5,  l_ant=20,  r_ant=30),
        Keyframe(t=3.0, pitch=10,  yaw=12,  roll=10,  l_ant=25,  r_ant=25),
        Keyframe(t=4.0, pitch=10,  yaw=12,  roll=10,  l_ant=25,  r_ant=25),
        Keyframe(t=5.0, pitch=0,   yaw=0,   roll=0,   l_ant=0,   r_ant=0),
    ]),
}


# ---------------------------------------------------------------------------
# Emotion engine
# ---------------------------------------------------------------------------

TICK = 0.04   # seconds between pose updates (~25 Hz)


class EmotionEngine:
    """
    Plays keyframe animations on Reachy Mini's head and antennas.

    Parameters
    ----------
    reachy : reachy_mini.ReachyMini or None
        None → simulation / logging only.
    """

    def __init__(self, reachy=None) -> None:
        self.reachy = reachy
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._current_emotion = Emotion.NEUTRAL

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
        """
        Play *emotion* animation.  Any currently running animation is
        cancelled immediately.  Set block=True to wait for completion.
        """
        self._cancel()
        self._set_emotion(emotion)
        anim = ANIMATIONS.get(emotion, ANIMATIONS[Emotion.NEUTRAL])
        self._stop_evt.clear()

        self._thread = threading.Thread(
            target=self._run,
            args=(anim,),
            name=f"emotion-{emotion.value}",
            daemon=True,
        )
        self._thread.start()
        logger.info("[emotion] playing: %s", emotion.value)

        if block:
            self._thread.join()

    def stop(self) -> None:
        """Stop current animation and return to neutral pose."""
        self._cancel()
        self._apply(NEUTRAL_POSE)
        self._set_emotion(Emotion.NEUTRAL)

    def idle(self) -> None:
        """Return gently to neutral (used when no face is detected)."""
        if self.current_emotion != Emotion.NEUTRAL:
            self.play(Emotion.NEUTRAL)

    # ------------------------------------------------------------------
    # Animation runner
    # ------------------------------------------------------------------

    def _cancel(self) -> None:
        self._stop_evt.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.3)

    def _run(self, anim: Animation) -> None:
        repeats = 0
        max_repeats = anim.loop_count if (anim.loop and anim.loop_count > 0) else (999 if anim.loop else 1)
        duration = anim.keyframes[-1].t

        while repeats < max_repeats and not self._stop_evt.is_set():
            t_start = time.monotonic()
            while not self._stop_evt.is_set():
                elapsed = time.monotonic() - t_start
                if elapsed >= duration:
                    break
                pose = interpolate(anim.keyframes, elapsed)
                self._apply(pose)
                time.sleep(TICK)

            # Hold the final frame for one tick before looping/ending
            if not self._stop_evt.is_set():
                self._apply(anim.keyframes[-1])
            repeats += 1

        # Always glide back to neutral when done (unless interrupted externally)
        if not self._stop_evt.is_set():
            self._glide_to_neutral(duration=0.8)
            self._set_emotion(Emotion.NEUTRAL)

    def _glide_to_neutral(self, duration: float = 0.8) -> None:
        """Smoothly interpolate from current pose to neutral."""
        # Build a two-keyframe mini-animation
        last_pose = NEUTRAL_POSE  # safe approximation; robot tracks last goal_position
        glide = Animation(keyframes=[
            Keyframe(t=0.0, **{f: getattr(last_pose, f) for f in ("pitch","yaw","roll","l_ant","r_ant")}),
            Keyframe(t=duration, pitch=0, yaw=0, roll=0, l_ant=0, r_ant=0),
        ])
        t_start = time.monotonic()
        while not self._stop_evt.is_set():
            elapsed = time.monotonic() - t_start
            if elapsed >= duration:
                break
            self._apply(interpolate(glide.keyframes, elapsed))
            time.sleep(TICK)
        self._apply(NEUTRAL_POSE)

    # ------------------------------------------------------------------
    # Hardware interface
    # ------------------------------------------------------------------

    def _apply(self, pose: Keyframe) -> None:
        """Send *pose* to the robot, or log it in sim mode."""
        if self.reachy is None:
            logger.debug(
                "[sim] emotion  pitch=%+.1f  yaw=%+.1f  roll=%+.1f  "
                "l_ant=%+.1f  r_ant=%+.1f",
                pose.pitch, pose.yaw, pose.roll, pose.l_ant, pose.r_ant,
            )
            return

        try:
            from reachy_mini.utils import create_head_pose
            head_pose = create_head_pose(
                pitch=pose.pitch, yaw=pose.yaw, roll=pose.roll, degrees=True
            )
            # SDK antenna order: [right_rad, left_rad]
            antennas = np.deg2rad([pose.r_ant, pose.l_ant])
            self.reachy.set_target(head=head_pose, antennas=antennas)
        except Exception:
            logger.debug("Motion command failed", exc_info=True)


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)-8s %(message)s")

    parser = argparse.ArgumentParser(description="Emotion animation demo (no robot)")
    parser.add_argument("--demo", action="store_true", default=True)
    parser.add_argument(
        "--emotion",
        default=None,
        help="Play one emotion then exit (e.g. tanzen, freude, trauer)",
    )
    args = parser.parse_args()

    engine = EmotionEngine(reachy=None)  # sim mode

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
