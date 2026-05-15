"""
Personality profile loader.

Profiles live in  profiles/<name>/
  instructions.txt  — personality text appended after the base system prompt
  voice.txt         — optional OpenAI TTS voice override (e.g. "echo", "onyx")

Built-in profiles
-----------------
  default          Warm, curious, lightly witty Reachy (no voice override)
  captain_circuit  Pirate robot — echo voice
  hype_bot         Ultra-enthusiastic hype machine — shimmer voice
  noir_detective   Hard-boiled noir detective — onyx voice
  mars_rover       Space-exploration scientist — alloy voice
  mad_scientist    Eccentric Professor Zap — fable voice
  time_traveler    Temporal explorer from many eras — sage voice

Usage
-----
    from modules.profiles import list_profiles, load_profile

    names = list_profiles()
    instructions, voice = load_profile("captain_circuit")
"""

from pathlib import Path
from typing import Optional

PROFILES_DIR = Path(__file__).parent.parent / "profiles"


def list_profiles() -> list[str]:
    """Return sorted list of available profile names."""
    if not PROFILES_DIR.exists():
        return []
    return sorted(
        d.name
        for d in PROFILES_DIR.iterdir()
        if d.is_dir() and (d / "instructions.txt").exists()
    )


def load_profile(name: str) -> tuple[str, Optional[str]]:
    """
    Load a profile by name.

    Returns
    -------
    (instructions, voice)
        instructions : str  — contents of instructions.txt (empty string if absent)
        voice        : str | None — contents of voice.txt, or None if not set
    """
    profile_dir = PROFILES_DIR / name

    instructions_path = profile_dir / "instructions.txt"
    instructions = (
        instructions_path.read_text(encoding="utf-8").strip()
        if instructions_path.exists()
        else ""
    )

    voice_path = profile_dir / "voice.txt"
    voice: Optional[str] = None
    if voice_path.exists():
        v = voice_path.read_text(encoding="utf-8").strip()
        if v:
            voice = v

    return instructions, voice
