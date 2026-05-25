"""Unit tests for ``autocut.scoring.ranker`` — final_score, sort, filters."""

from __future__ import annotations

from datetime import timedelta

import pytest

from autocut.config import ScoringConfig
from autocut.models import Category, Clip, ClipPlan, ClipPlanMetadata
from autocut.scoring.ranker import final_score, rank_clips


def _clip(
    id_: str,
    start_sec: float,
    end_sec: float,
    *,
    score: int,
    category: Category = Category.highlight,
) -> Clip:
    return Clip(
        id=id_,
        start=timedelta(seconds=start_sec),
        end=timedelta(seconds=end_sec),
        category=category,
        description=f"clip {id_}",
        score=score,
        rationale="because",
    )


def _plan(clips: list[Clip]) -> ClipPlan:
    return ClipPlan(
        video_id="vid",
        duration_sec=600.0,
        clips=clips,
        metadata=ClipPlanMetadata(vlm_provider="test", vlm_model="m"),
    )


# ---------------------------------------------------------------------------
# final_score
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("vlm", "heur", "expected"),
    [
        (10, 10, 10),  # 0.7*10 + 0.3*10 = 10
        (0, 0, 0),
        (10, 0, 7),  # 0.7*10 = 7.0
        (0, 10, 3),  # 0.3*10 = 3.0
        (8, 5, 7),  # 5.6 + 1.5 = 7.1 -> 7
        (5, 5, 5),
    ],
)
def test_final_score_weights(vlm: int, heur: int, expected: int) -> None:
    assert final_score(vlm, heur) == expected


def test_final_score_is_clamped() -> None:
    # Defensive: inputs outside [0, 10] should not overflow the output.
    assert final_score(99, 99) == 10
    assert final_score(-5, -5) == 0


# ---------------------------------------------------------------------------
# rank_clips: order
# ---------------------------------------------------------------------------


def test_rank_orders_by_final_score_desc_then_start_asc() -> None:
    # Three clips, hand-tuned so we can predict ordering:
    # - high_late: vlm 10, duration 15s (heur 7) -> final 9, starts at 100s
    # - high_early: vlm 10, duration 15s (heur 7) -> final 9, starts at 10s
    # - mid: vlm 6, duration 15s (heur 7) -> final 6, starts at 50s
    high_late = _clip("late", 100, 115, score=10)
    high_early = _clip("early", 10, 25, score=10)
    mid = _clip("mid", 50, 65, score=6)
    ranked = rank_clips(_plan([high_late, high_early, mid]), ScoringConfig(min_score=0))
    assert [r.clip.id for r in ranked] == ["early", "late", "mid"]


def test_rank_drops_clips_below_min_score() -> None:
    # Very-short clip: duration 1s -> heur 3, vlm 0 -> final 1
    weak = _clip("weak", 0, 1, score=0)
    strong = _clip("strong", 30, 45, score=9)
    ranked = rank_clips(_plan([weak, strong]), ScoringConfig(min_score=5))
    ids = [r.clip.id for r in ranked]
    assert ids == ["strong"]


def test_rank_caps_to_max_clips() -> None:
    clips = [_clip(f"c{i}", float(i * 30), float(i * 30 + 15), score=9) for i in range(5)]
    ranked = rank_clips(_plan(clips), ScoringConfig(min_score=0, max_clips=2))
    assert len(ranked) == 2


def test_rank_attaches_both_raw_scores() -> None:
    # A sweet-spot clip with vlm=8 should show vlm_score=8, heuristic_score=7,
    # final_score=round(0.7*8 + 0.3*7) = round(7.7) = 8.
    clip = _clip("c", 0, 15, score=8)
    ranked = rank_clips(_plan([clip]), ScoringConfig(min_score=0))
    assert len(ranked) == 1
    r = ranked[0]
    assert r.vlm_score == 8
    assert r.heuristic_score == 7
    assert r.final_score == 8


def test_rank_empty_plan_returns_empty_list() -> None:
    assert rank_clips(_plan([]), ScoringConfig()) == []
