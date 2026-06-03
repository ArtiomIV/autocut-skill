"""Unit tests for the pure host two-pass maths (candidates + dedup)."""

from __future__ import annotations

from datetime import timedelta

from autocut.host_analysis import (
    FINE_PAD_SEC,
    dedup_plan,
    empty_plan,
    select_fine_windows,
)
from autocut.models import Clip, ClipPlan, ClipPlanMetadata


def _clip(start: float, end: float, score: int, cid: str = "c") -> Clip:
    return Clip(
        id=cid,
        start=timedelta(seconds=start),
        end=timedelta(seconds=end),
        category="highlight",
        description="x",
        score=score,
        rationale="r",
        tags=["t"],
    )


def _plan(clips: list[Clip]) -> ClipPlan:
    return ClipPlan(
        video_id="v",
        duration_sec=1000.0,
        clips=clips,
        metadata=ClipPlanMetadata(vlm_provider="host", vlm_model="claude", prompt_version="v12"),
    )


# ---------------------------------------------------------------------------
# select_fine_windows
# ---------------------------------------------------------------------------


def test_select_pads_and_clamps_to_source() -> None:
    plan = _plan([_clip(2.0, 5.0, 9)])
    windows = select_fine_windows(plan, duration_sec=100.0)
    # ±FINE_PAD_SEC padding, clamped to [0, duration].
    assert windows == [(0.0, 5.0 + FINE_PAD_SEC)]


def test_select_does_not_merge_overlapping_windows() -> None:
    # Two clips 10s apart: after ±20s padding their windows overlap, but they are
    # deliberately NOT merged — each keeps its own densely-sampled window (the
    # duplicate clips are dropped later by dedup_plan). Overlap is allowed.
    plan = _plan([_clip(100.0, 105.0, 9, "a"), _clip(115.0, 120.0, 8, "b")])
    windows = select_fine_windows(plan, duration_sec=1000.0)
    assert windows == [(80.0, 125.0), (95.0, 140.0)]


def test_select_keeps_separate_distant_windows_sorted() -> None:
    plan = _plan([_clip(500.0, 505.0, 7, "late"), _clip(100.0, 105.0, 9, "early")])
    windows = select_fine_windows(plan, duration_sec=1000.0)
    assert windows == [(80.0, 125.0), (480.0, 525.0)]


def test_select_keeps_only_candidates_above_min_score() -> None:
    # Scores 0..10; with min_score=7 only the regions scored 7,8,9,10 survive
    # (no fixed top-N cap any more).
    clips = [_clip(float(i) * 100, float(i) * 100 + 5, i, f"c{i}") for i in range(11)]
    windows = select_fine_windows(_plan(clips), duration_sec=10_000.0, min_score=7)
    assert len(windows) == 4


def test_select_ceiling_guards_against_runaway() -> None:
    # 60 strong candidates, all above the threshold, are capped at the ceiling.
    clips = [_clip(float(i) * 100, float(i) * 100 + 5, 9, f"c{i}") for i in range(60)]
    windows = select_fine_windows(_plan(clips), duration_sec=100_000.0, min_score=5, ceiling=40)
    assert len(windows) == 40


def test_select_empty_when_no_clips() -> None:
    assert select_fine_windows(_plan([]), duration_sec=100.0) == []


# ---------------------------------------------------------------------------
# dedup_plan
# ---------------------------------------------------------------------------


def test_dedup_drops_high_iou_keeping_higher_score() -> None:
    # Two near-identical windows; the lower-scored one is dropped.
    plan = _plan([_clip(10.0, 20.0, 6, "lo"), _clip(10.5, 20.5, 9, "hi")])
    out = dedup_plan(plan)
    assert [c.id for c in out.clips] == ["hi"]


def test_dedup_keeps_disjoint_clips_chronological() -> None:
    plan = _plan([_clip(50.0, 60.0, 8, "b"), _clip(10.0, 20.0, 9, "a")])
    out = dedup_plan(plan)
    assert [c.id for c in out.clips] == ["a", "b"]


# ---------------------------------------------------------------------------
# empty_plan
# ---------------------------------------------------------------------------


def test_empty_plan_is_valid_and_empty() -> None:
    plan = empty_plan("vid", 42.0, agent_hint="claude", prompt_version="v12")
    assert plan.clips == []
    assert plan.video_id == "vid"
    assert plan.duration_sec == 42.0
    assert plan.metadata.vlm_provider == "host"
