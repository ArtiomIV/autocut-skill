"""Heuristic scoring layer.

These heuristics complement the VLM's subjective score with cheap,
objective signals about clip shape. They run on the ``ClipPlan`` after
validation and before output.

v0.1.0 scope (intentionally minimal):
- ``duration_score``: penalises clips that are too short or too long, and
  rewards clips in a "sweet spot" range that works well on social-media
  platforms (8-25 s).

Deferred to later versions:
- Motion intensity (needs a second ffmpeg decode pass) → v0.4.0
- Audio peak detection (applause, laughter) → v0.2.0 alongside the
  spoken-content / audio-driven pipeline track.

The output is clamped to the integer range ``[0, 10]`` so it can be mixed
1:1 with the VLM score by ``ranker.final_score``.
"""

from __future__ import annotations

from autocut.models import Clip

# Sweet-spot window: clips in this range are bumped, outside the bonus tails off.
_SWEET_SPOT_LOW_SEC: float = 8.0
_SWEET_SPOT_HIGH_SEC: float = 25.0

# Hard penalties: under/over these limits the heuristic actively docks points.
_HARD_MIN_SEC: float = 3.0
_HARD_MAX_SEC: float = 45.0

_BASE_SCORE: float = 5.0


def duration_score(clip: Clip) -> float:
    """Return a duration-based score in ``[0, 10]`` (float).

    Curve:
    - <``_HARD_MIN_SEC``: -2 (very short clips read as accidents)
    - ``[_HARD_MIN_SEC, _SWEET_SPOT_LOW_SEC)``: -1 (a little short)
    - ``[_SWEET_SPOT_LOW_SEC, _SWEET_SPOT_HIGH_SEC]``: +2 (ideal range)
    - ``(_SWEET_SPOT_HIGH_SEC, _HARD_MAX_SEC]``: 0 (acceptable, no bonus)
    - >``_HARD_MAX_SEC``: -1 (longer than most attention spans)
    """
    d = clip.duration_sec
    score = _BASE_SCORE
    if d < _HARD_MIN_SEC:
        score -= 2.0
    elif d < _SWEET_SPOT_LOW_SEC:
        score -= 1.0
    elif d <= _SWEET_SPOT_HIGH_SEC:
        score += 2.0
    elif d > _HARD_MAX_SEC:
        score -= 1.0
    return _clamp(score)


def heuristic_score(clip: Clip) -> int:
    """Aggregate every heuristic into a single integer score in ``[0, 10]``."""
    return round(duration_score(clip))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, low: float = 0.0, high: float = 10.0) -> float:
    return max(low, min(high, value))
