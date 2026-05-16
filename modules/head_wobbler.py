"""Moves head in sync with TTS audio — from pollen-robotics/reachy_mini_conversation_app."""

from __future__ import annotations
import time
import queue
import logging
import threading
from typing import Tuple
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from modules.speech_tapper import HOP_MS, SwayRollRT


SAMPLE_RATE = 24000
MOVEMENT_LATENCY_S = 0.2
logger = logging.getLogger(__name__)


class HeadWobbler:
    """Converts audio (PCM int16) into head movement offsets via SwayRollRT."""

    def __init__(self, set_speech_offsets: Callable[[Tuple[float, float, float, float, float, float]], None]) -> None:
        self._apply_offsets = set_speech_offsets
        self._base_ts: float | None = None
        self._hops_done: int = 0
        self.audio_queue: "queue.Queue[Tuple[int, int, NDArray[np.int16], float]]" = queue.Queue()
        self.sway = SwayRollRT()
        self._state_lock = threading.Lock()
        self._sway_lock = threading.Lock()
        self._generation = 0
        self._reset_after_audio = False
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def feed_pcm(self, pcm: NDArray[np.int16], sample_rate: int, start_delay_s: float = 0.0) -> None:
        with self._state_lock:
            generation = self._generation
            self._reset_after_audio = False
        self.audio_queue.put((generation, sample_rate, pcm, max(0.0, start_delay_s)))

    def request_reset_after_current_audio(self) -> None:
        should_reset_now = False
        with self._state_lock:
            self._reset_after_audio = True
            should_reset_now = self._base_ts is None and self.audio_queue.empty()
        if should_reset_now:
            self.reset()

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._working_loop, daemon=True, name="head-wobbler")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def reset(self) -> None:
        with self._state_lock:
            self._generation += 1
            self._base_ts = None
            self._hops_done = 0
            self._reset_after_audio = False
        while True:
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except queue.Empty:
                break
        with self._sway_lock:
            self.sway.reset()
        self._apply_offsets((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    def _should_reset_after_audio(self, hop_dt: float) -> bool:
        with self._state_lock:
            if not self._reset_after_audio or self._base_ts is None:
                return False
            if not self.audio_queue.empty():
                return False
            reset_at = self._base_ts + MOVEMENT_LATENCY_S + self._hops_done * hop_dt
        return time.monotonic() >= reset_at

    def _working_loop(self) -> None:
        hop_dt = HOP_MS / 1000.0
        while not self._stop_event.is_set():
            try:
                chunk_gen, sr, chunk, start_delay = self.audio_queue.get(timeout=hop_dt)
            except queue.Empty:
                if self._should_reset_after_audio(hop_dt):
                    self.reset()
                continue
            try:
                with self._state_lock:
                    if chunk_gen != self._generation:
                        continue
                    if self._base_ts is None:
                        self._base_ts = time.monotonic() + start_delay

                pcm = np.asarray(chunk).squeeze()
                if pcm.ndim == 0:
                    continue
                with self._sway_lock:
                    results = self.sway.feed(pcm, sr)

                i = 0
                while i < len(results):
                    with self._state_lock:
                        if self._generation != chunk_gen:
                            break
                        base_ts = self._base_ts
                        hops_done = self._hops_done

                    if base_ts is None:
                        base_ts = time.monotonic()
                        with self._state_lock:
                            if self._base_ts is None:
                                self._base_ts = base_ts
                                hops_done = self._hops_done

                    target = base_ts + MOVEMENT_LATENCY_S + hops_done * hop_dt
                    now = time.monotonic()

                    if now - target >= hop_dt:
                        lag_hops = int((now - target) / hop_dt)
                        drop = min(lag_hops, len(results) - i - 1)
                        if drop > 0:
                            with self._state_lock:
                                self._hops_done += drop
                                hops_done = self._hops_done
                            i += drop
                            continue

                    if target > now:
                        time.sleep(target - now)
                        with self._state_lock:
                            if self._generation != chunk_gen:
                                break

                    r = results[i]
                    offsets = (
                        r["x_mm"] / 1000.0, r["y_mm"] / 1000.0, r["z_mm"] / 1000.0,
                        r["roll_rad"], r["pitch_rad"], r["yaw_rad"],
                    )
                    with self._state_lock:
                        if self._generation != chunk_gen:
                            break
                    self._apply_offsets(offsets)
                    with self._state_lock:
                        self._hops_done += 1
                    i += 1
            finally:
                self.audio_queue.task_done()
