"""Heuristic scoring layer.

These heuristics complement the VLM's subjective score with cheap,
objective signals about clip shape. They run on the ``ClipPlan`` after
validation and before output.

v0.1.0 scope (intentionally minimal):
- ``duration_score``: only penalises clips at the extremes (under 3 s or
  over 60 s). Anything in between gets a neutral score so the VLM's
  judgement on the right clip length is not second-guessed.

Deferred to later versions:
- Motion intensity (needs a second ffmpeg decode pass) → v0.1.1
- Audio peak detection (applause, laughter) → v0.1.1 alongside the
  smart-sampling track.

The output is clamped to the integer range ``[0, 10]`` so it can be mixed
1:1 with the VLM score by ``ranker.final_score``.
"""

from __future__ import annotations

from autocut.models import Clip

# Hard guardrails: outside these bounds the clip is almost certainly broken
# (under 3 s reads as an accident; over 60 s is a scene, not a clip).
# Between the bounds we stay neutral and trust the VLM.
_HARD_MIN_SEC: float = 3.0
_HARD_MAX_SEC: float = 60.0
_HARD_PENALTY: float = 3.0

_BASE_SCORE: float = 5.0


def duration_score(clip: Clip) -> float:
    """Return a duration-based score in ``[0, 10]`` (float).

    Curve:
    - <``_HARD_MIN_SEC``: base - 3 (too short to be readable)
    - ``[_HARD_MIN_SEC, _HARD_MAX_SEC]``: base (neutral — trust the VLM)
    - >``_HARD_MAX_SEC``: base - 3 (a scene, not a clip)
    """
    d = clip.duration_sec
    score = _BASE_SCORE
    if d < _HARD_MIN_SEC or d > _HARD_MAX_SEC:
        score -= _HARD_PENALTY
    return _clamp(score)


def heuristic_score(clip: Clip) -> int:
    """Aggregate every heuristic into a single integer score in ``[0, 10]``."""
    return round(duration_score(clip))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, low: float = 0.0, high: float = 10.0) -> float:
    return max(low, min(high, value))
