"""Unit tests for ``autocut.chunking`` — pure logic, no I/O."""

from __future__ import annotations

from datetime import timedelta

import pytest

from autocut.chunking import (
    Chunk,
    merge_plans,
    split_into_chunks,
    temporal_iou,
)
from autocut.models import Category, Clip, ClipPlan, ClipPlanMetadata

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clip(
    id_: str,
    start_sec: float,
    end_sec: float,
    *,
    score: int = 7,
    category: Category = Category.highlight,
) -> Clip:
    return Clip(
        id=id_,
        start=timedelta(seconds=start_sec),
        end=timedelta(seconds=end_sec),
        category=category,
        description=f"clip {id_}",
        score=score,
        rationale="r",
    )


def _plan(clips: list[Clip], *, provider: str = "openrouter") -> ClipPlan:
    return ClipPlan(
        video_id="vid",
        duration_sec=600.0,
        clips=clips,
        metadata=ClipPlanMetadata(vlm_provider=provider, vlm_model="m"),
    )


# ---------------------------------------------------------------------------
# Chunk shape
# ---------------------------------------------------------------------------


def test_chunk_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError, match="end_sec must be strictly greater"):
        Chunk(index=0, start_sec=10, end_sec=10)
    with pytest.raises(ValueError, match="end_sec must be strictly greater"):
        Chunk(index=0, start_sec=10, end_sec=5)


def test_chunk_rejects_negative_start() -> None:
    with pytest.raises(ValueError, match="start_sec must be >= 0"):
        Chunk(index=0, start_sec=-1, end_sec=5)


def test_chunk_duration_property() -> None:
    assert Chunk(index=0, start_sec=10, end_sec=25).duration_sec == 15


# ---------------------------------------------------------------------------
# split_into_chunks
# ---------------------------------------------------------------------------


def test_split_returns_empty_for_short_video() -> None:
    # Source < chunk_duration → no chunking needed.
    assert split_into_chunks(60.0, chunk_duration_sec=120.0) == []


def test_split_produces_expected_chunks_without_overlap() -> None:
    # 200 s / 60 s = 3 full chunks + a 20 s tail = 4 chunks.
    chunks = split_into_chunks(200.0, chunk_duration_sec=60.0, overlap_sec=0.0)
    assert len(chunks) == 4
    assert chunks[0] == Chunk(index=0, start_sec=0.0, end_sec=60.0)
    assert chunks[1] == Chunk(index=1, start_sec=60.0, end_sec=120.0)
    assert chunks[2] == Chunk(index=2, start_sec=120.0, end_sec=180.0)
    # Final chunk clipped at EOF.
    assert chunks[3] == Chunk(index=3, start_sec=180.0, end_sec=200.0)


def test_split_with_overlap_extends_each_chunk() -> None:
    # 200 s, chunk=60 s, overlap=10 s → each chunk except the last covers
    # 70 s but advances by 60 s.
    chunks = split_into_chunks(200.0, chunk_duration_sec=60.0, overlap_sec=10.0)
    assert chunks[0].start_sec == 0.0
    assert chunks[0].end_sec == 70.0
    assert chunks[1].start_sec == 60.0
    assert chunks[1].end_sec == 130.0
    assert chunks[2].start_sec == 120.0
    assert chunks[2].end_sec == 190.0
    # Adjacent chunks share 10 s.
    assert chunks[1].start_sec < chunks[0].end_sec
    assert chunks[0].end_sec - chunks[1].start_sec == 10.0


def test_split_rejects_invalid_params() -> None:
    with pytest.raises(ValueError, match="duration_sec"):
        split_into_chunks(0, chunk_duration_sec=60.0)
    with pytest.raises(ValueError, match="chunk_duration_sec"):
        split_into_chunks(120.0, chunk_duration_sec=10.0)  # below _MIN
    with pytest.raises(ValueError, match="overlap_sec must be >= 0"):
        split_into_chunks(120.0, chunk_duration_sec=60.0, overlap_sec=-1.0)
    with pytest.raises(ValueError, match="overlap_sec must be <"):
        split_into_chunks(120.0, chunk_duration_sec=60.0, overlap_sec=60.0)


# ---------------------------------------------------------------------------
# temporal_iou
# ---------------------------------------------------------------------------


def test_iou_identical_intervals_is_one() -> None:
    assert temporal_iou(10, 20, 10, 20) == 1.0


def test_iou_disjoint_intervals_is_zero() -> None:
    assert temporal_iou(10, 20, 30, 40) == 0.0


def test_iou_partial_overlap() -> None:
    # [10, 30] vs [20, 40]: intersection=10, union=30 → IoU = 1/3.
    assert temporal_iou(10, 30, 20, 40) == pytest.approx(1 / 3)


def test_iou_one_inside_the_other() -> None:
    # [0, 100] vs [40, 60]: intersection=20, union=100 → IoU = 0.2.
    assert temporal_iou(0, 100, 40, 60) == pytest.approx(0.2)


def test_iou_touching_intervals_is_zero() -> None:
    # Sharing only an endpoint → intersection 0.
    assert temporal_iou(10, 20, 20, 30) == 0.0


# ---------------------------------------------------------------------------
# merge_plans
# ---------------------------------------------------------------------------


def test_merge_shifts_clip_timestamps_by_chunk_offset() -> None:
    chunks = [
        Chunk(index=0, start_sec=0.0, end_sec=60.0),
        Chunk(index=1, start_sec=60.0, end_sec=120.0),
    ]
    # Chunk 0: clip at 5-15 s (local) → 5-15 s global.
    # Chunk 1: clip at 5-15 s (local) → 65-75 s global.
    plans = [_plan([_clip("a", 5, 15, score=7)]), _plan([_clip("b", 5, 15, score=7)])]
    merged = merge_plans(plans, chunks, video_id="vid", duration_sec=120.0)
    assert len(merged.clips) == 2
    starts = sorted(c.start.total_seconds() for c in merged.clips)
    ends = sorted(c.end.total_seconds() for c in merged.clips)
    assert starts == [5.0, 65.0]
    assert ends == [15.0, 75.0]


def test_merge_dedupes_overlapping_clips_keeping_higher_score() -> None:
    # Two chunks proposing essentially the same clip in the overlap zone:
    # chunk 0 says "65-75 s, score 5"; chunk 1 says "65-75 s, score 9".
    chunks = [
        Chunk(index=0, start_sec=0.0, end_sec=80.0),
        Chunk(index=1, start_sec=60.0, end_sec=140.0),
    ]
    plans = [
        _plan([_clip("low", 65, 75, score=5)]),
        _plan([_clip("high", 5, 15, score=9)]),  # local 5-15 → global 65-75
    ]
    merged = merge_plans(plans, chunks, video_id="vid", duration_sec=140.0)
    assert len(merged.clips) == 1
    assert merged.clips[0].score == 9


def test_merge_keeps_distinct_clips_when_iou_below_threshold() -> None:
    chunks = [
        Chunk(index=0, start_sec=0.0, end_sec=60.0),
        Chunk(index=1, start_sec=60.0, end_sec=120.0),
    ]
    plans = [
        _plan([_clip("a", 10, 20, score=7)]),
        _plan([_clip("b", 30, 40, score=7)]),  # local 30-40 → global 90-100
    ]
    merged = merge_plans(plans, chunks, video_id="vid", duration_sec=120.0)
    # No temporal overlap → both kept.
    assert len(merged.clips) == 2


def test_merge_chronological_order_of_survivors() -> None:
    chunks = [
        Chunk(index=0, start_sec=0.0, end_sec=60.0),
        Chunk(index=1, start_sec=60.0, end_sec=120.0),
    ]
    plans = [
        _plan([_clip("late", 40, 50, score=7)]),
        _plan([_clip("early", 5, 15, score=9)]),  # local 5-15 → global 65-75
    ]
    merged = merge_plans(plans, chunks, video_id="vid", duration_sec=120.0)
    timestamps = [c.start.total_seconds() for c in merged.clips]
    assert timestamps == sorted(timestamps)


def test_merge_clip_ids_carry_chunk_prefix() -> None:
    chunks = [Chunk(index=0, start_sec=0.0, end_sec=60.0)]
    plans = [_plan([_clip("highlight1", 5, 15)])]
    merged = merge_plans(plans, chunks, video_id="vid", duration_sec=60.0)
    assert merged.clips[0].id.startswith("c0_")


def test_merge_preserves_metadata_from_first_plan() -> None:
    chunks = [
        Chunk(index=0, start_sec=0.0, end_sec=60.0),
        Chunk(index=1, start_sec=60.0, end_sec=120.0),
    ]
    plans = [
        _plan([_clip("a", 5, 15)], provider="openrouter"),
        _plan([_clip("b", 5, 15)], provider="openrouter"),
    ]
    merged = merge_plans(plans, chunks, video_id="vid", duration_sec=120.0)
    assert merged.metadata.vlm_provider == "openrouter"
    assert merged.metadata.vlm_model == "m"


def test_merge_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        merge_plans(
            [_plan([_clip("a", 5, 15)])],
            [
                Chunk(index=0, start_sec=0.0, end_sec=60.0),
                Chunk(index=1, start_sec=60.0, end_sec=120.0),
            ],
            video_id="vid",
            duration_sec=120.0,
        )


def test_merge_rejects_empty_plans_list() -> None:
    with pytest.raises(ValueError, match="at least one ClipPlan"):
        merge_plans([], [], video_id="vid", duration_sec=120.0)


def test_merge_rejects_invalid_iou_threshold() -> None:
    chunks = [Chunk(index=0, start_sec=0.0, end_sec=60.0)]
    plans = [_plan([_clip("a", 5, 15)])]
    with pytest.raises(ValueError, match="iou_threshold"):
        merge_plans(plans, chunks, video_id="vid", duration_sec=60.0, iou_threshold=0)
    with pytest.raises(ValueError, match="iou_threshold"):
        merge_plans(plans, chunks, video_id="vid", duration_sec=60.0, iou_threshold=1.5)
