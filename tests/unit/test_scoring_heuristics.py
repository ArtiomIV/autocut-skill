"""Unit tests for ``autocut.scoring.heuristics`` — duration-curve behaviour."""

from __future__ import annotations

from datetime import timedelta

import pytest

from autocut.models import Category, Clip
from autocut.scoring.heuristics import duration_score, heuristic_score


def _clip(duration_sec: float, score: int = 5) -> Clip:
    return Clip(
        id="c1",
        start=timedelta(seconds=0),
        end=timedelta(seconds=duration_sec),
        category=Category.highlight,
        description="a clip",
        score=score,
        rationale="because",
    )


# ---------------------------------------------------------------------------
# duration_score: every band of the curve
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("duration", "expected"),
    [
        (1.0, 2.0),  # < HARD_MIN (3.0)  -> base 5 - 3 = 2
        (2.9, 2.0),  # still under hard min
        (3.0, 5.0),  # [3, 60]           -> base 5 (neutral, trust the VLM)
        (7.9, 5.0),  # neutral band
        (8.0, 5.0),  # neutral band
        (15.0, 5.0),  # neutral band
        (25.0, 5.0),  # neutral band
        (45.0, 5.0),  # neutral band
        (60.0, 5.0),  # upper edge inclusive
        (60.1, 2.0),  # > HARD_MAX (60.0) -> base 5 - 3 = 2
        (120.0, 2.0),  # very long
    ],
)
def test_duration_score_curve(duration: float, expected: float) -> None:
    assert duration_score(_clip(duration)) == expected


def test_duration_score_is_clamped_to_unit_range() -> None:
    # The current curve only produces 2.0 or 5.0, but the clamp is part of
    # the contract: future heuristics must not be able to overflow.
    assert 0.0 <= duration_score(_clip(0.1)) <= 10.0
    assert 0.0 <= duration_score(_clip(9999.0)) <= 10.0


# ---------------------------------------------------------------------------
# heuristic_score: integer aggregate
# ---------------------------------------------------------------------------


def test_heuristic_score_returns_int_in_unit_range() -> None:
    value = heuristic_score(_clip(15.0))
    assert isinstance(value, int)
    assert 0 <= value <= 10


def test_heuristic_score_matches_rounded_duration_score() -> None:
    clip = _clip(15.0)
    assert heuristic_score(clip) == round(duration_score(clip))
