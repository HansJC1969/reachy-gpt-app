"""
GPT-4o Vision — Umgebungserkennung für Reachy.

Sendet Kamera-Frames als JPEG-Base64 an GPT-4o und gibt eine Beschreibung
zurück.  Zwei Modi:
  • analyze_periodic(frame)  — nur wenn das Intervall abgelaufen ist (passiv)
  • analyze_on_command(frame, question) — sofortige Antwort auf Nutzerfrage

Standalone:
    python -m modules.vision --camera 0
"""

import base64
import logging
import os
import threading
import time
from typing import Optional

import cv2
import numpy as np
import openai
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Default prompt for periodic scene description
_SCENE_PROMPT = (
    "Describe what you see in this image briefly and concisely (2-3 sentences). "
    "Focus on: people present, objects, setting/environment, and anything unusual. "
    "Speak in first person as Reachy the robot."
)

# Default image quality and max size (keeps token cost manageable)
_JPEG_QUALITY = 82
_MAX_DIM = 512


def _encode_frame(frame: np.ndarray, max_dim: int = _MAX_DIM) -> str:
    """Resize frame to fit within *max_dim* × *max_dim* and return base64 JPEG."""
    h, w = frame.shape[:2]
    scale = min(max_dim / max(w, h), 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


class VisionAnalyzer:
    """
    Wraps GPT-4o vision for periodic scene awareness and on-demand queries.

    Parameters
    ----------
    api_key : str | None
        OpenAI API key.  Falls back to OPENAI_API_KEY env var.
    interval : float
        Minimum seconds between automatic scene analyses.
    model : str
        OpenAI model that supports vision (default: gpt-4.1-nano).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        interval: float = 10.0,
        model: str = "gpt-4.1-nano",
    ) -> None:
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise EnvironmentError("OPENAI_API_KEY not set")
        self._client = openai.OpenAI(api_key=key)
        self._model = model
        self._interval = interval
        self._last_analyzed: float = 0.0
        self._last_description: Optional[str] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def last_description(self) -> Optional[str]:
        """Most recent scene description, or None if none yet."""
        with self._lock:
            return self._last_description

    def analyze(
        self,
        frame: np.ndarray,
        question: Optional[str] = None,
        max_tokens: int = 150,
    ) -> str:
        """
        Send *frame* to GPT-4o vision and return a description.
        If *question* is given it is used as the prompt; otherwise
        the generic scene-description prompt is used.
        """
        if frame is None or frame.size == 0:
            raise ValueError("analyze() received an invalid (None or empty) frame")
        prompt = question or _SCENE_PROMPT
        b64 = _encode_frame(frame)

        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}",
                                "detail": "low",   # "low" = ~85 tokens, cheaper
                            },
                        },
                    ],
                }
            ],
        )
        description = response.choices[0].message.content.strip()
        logger.debug("Vision: %s", description[:120])
        return description

    def analyze_periodic(self, frame: np.ndarray) -> Optional[str]:
        """
        Run analyze() only if *interval* seconds have passed since last call.
        Returns the description, or None if skipped.
        Updates `last_description` on success.
        """
        with self._lock:
            last = self._last_analyzed
        if time.monotonic() - last < self._interval:
            return None
        try:
            desc = self.analyze(frame)
            with self._lock:
                self._last_description = desc
                self._last_analyzed = time.monotonic()
            logger.info("Scene update: %s", desc[:100])
            return desc
        except Exception:
            logger.exception("Vision periodic analysis failed")
            return None

    def analyze_on_command(
        self,
        frame: np.ndarray,
        user_question: str,
        max_tokens: int = 200,
    ) -> str:
        """
        Answer the user's visual question directly.
        Always runs regardless of interval.
        """
        prompt = (
            f"You are Reachy the robot looking through your camera. "
            f"Answer this question about what you see: {user_question}"
        )
        try:
            answer = self.analyze(frame, question=prompt, max_tokens=max_tokens)
            with self._lock:
                self._last_description = answer
                self._last_analyzed = time.monotonic()
            return answer
        except Exception:
            logger.exception("Vision on-command failed")
            return "Sorry, I couldn't process the image right now."

    def reset_timer(self) -> None:
        """Force next analyze_periodic() call to run immediately."""
        with self._lock:
            self._last_analyzed = 0.0


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=int(os.environ.get("CAMERA_INDEX", "0")))
    parser.add_argument("--interval", type=float, default=10.0)
    args = parser.parse_args()

    analyzer = VisionAnalyzer(interval=args.interval)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Cannot open camera {args.camera}")
        raise SystemExit(1)

    print("Press 's' to describe scene, 'q' to quit, or wait for auto-interval.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        desc = analyzer.analyze_periodic(frame)
        if desc:
            print(f"\n[auto] {desc}\n")

        cv2.imshow("VisionAnalyzer", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            print("\n[manual] Analysing…")
            print(analyzer.analyze(frame))
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
