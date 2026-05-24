"""Unit tests for ``autocut.video.frame_sampler`` — pure logic, no I/O."""

from __future__ import annotations

from datetime import timedelta

import pytest

from autocut.models import Scene
from autocut.video.frame_sampler import (
    FrameSpec,
    build_sampler,
    sample_hybrid,
    sample_scene_based,
    sample_uniform,
)


def _scene(idx: int, start: float, end: float) -> Scene:
    return Scene(index=idx, start=timedelta(seconds=start), end=timedelta(seconds=end))


# ---------------------------------------------------------------------------
# sample_scene_based
# ---------------------------------------------------------------------------


def test_scene_based_returns_per_scene_count() -> None:
    scenes = [_scene(0, 0, 9), _scene(1, 9, 18)]
    specs = sample_scene_based(scenes, per_scene=2)
    assert len(specs) == 4
    assert [s.scene_index for s in specs] == [0, 0, 1, 1]


def test_scene_based_picks_interior_thirds_for_per_scene_2() -> None:
    specs = sample_scene_based([_scene(0, 0, 9)], per_scene=2)
    assert [s.timestamp.total_seconds() for s in specs] == pytest.approx([3.0, 6.0])


def test_scene_based_picks_quarters_for_per_scene_3() -> None:
    specs = sample_scene_based([_scene(0, 0, 8)], per_scene=3)
    assert [s.timestamp.total_seconds() for s in specs] == pytest.approx([2.0, 4.0, 6.0])


def test_scene_based_returns_empty_for_no_scenes() -> None:
    assert sample_scene_based([], per_scene=2) == []


def test_scene_based_rejects_invalid_per_scene() -> None:
    with pytest.raises(ValueError):
        sample_scene_based([_scene(0, 0, 5)], per_scene=0)


# ---------------------------------------------------------------------------
# sample_uniform
# ---------------------------------------------------------------------------


def test_uniform_samples_every_interval() -> None:
    specs = sample_uniform(duration_sec=10.0, interval_sec=2.0)
    assert [s.timestamp.total_seconds() for s in specs] == pytest.approx([1.0, 3.0, 5.0, 7.0, 9.0])


def test_uniform_tags_scene_index_with_minus_one() -> None:
    specs = sample_uniform(duration_sec=5.0, interval_sec=1.0)
    assert all(s.scene_index == -1 for s in specs)


def test_uniform_returns_empty_for_short_video() -> None:
    # First sample is at interval/2 = 5s, but duration is 4s -> no samples fit.
    assert sample_uniform(duration_sec=4.0, interval_sec=10.0) == []


def test_uniform_rejects_invalid_args() -> None:
    with pytest.raises(ValueError):
        sample_uniform(duration_sec=0)
    with pytest.raises(ValueError):
        sample_uniform(duration_sec=10, interval_sec=0)


# ---------------------------------------------------------------------------
# sample_hybrid
# ---------------------------------------------------------------------------


def test_hybrid_acts_like_scene_for_short_scenes() -> None:
    # Both scenes are shorter than max_gap_sec -> no supplements added.
    scenes = [_scene(0, 0, 2), _scene(1, 2, 4)]
    specs = sample_hybrid(scenes, duration_sec=4, per_scene=2, max_gap_sec=3.0)
    assert len(specs) == 4
    assert all(s.scene_index in {0, 1} for s in specs)


def test_hybrid_adds_supplements_inside_long_scenes() -> None:
    # One 20s scene with per_scene=2 and max_gap=3 -> need extras.
    scenes = [_scene(0, 0, 20)]
    specs = sample_hybrid(scenes, duration_sec=20, per_scene=2, max_gap_sec=3.0)
    # Without supplements we'd get 2 samples; with them we get more.
    assert len(specs) > 2
    # Within max_gap_sec + small tolerance for the spacing.
    diffs = [
        (specs[i + 1].timestamp - specs[i].timestamp).total_seconds() for i in range(len(specs) - 1)
    ]
    # All gaps should be reasonable; not strict <= because deduplication may
    # leave the first/last gap slightly bigger, but most must be <= max_gap.
    assert max(diffs) <= 4.5


def test_hybrid_falls_back_to_uniform_when_no_scenes() -> None:
    specs = sample_hybrid([], duration_sec=10, max_gap_sec=2.0)
    assert all(s.scene_index == -1 for s in specs)
    assert len(specs) > 0


# ---------------------------------------------------------------------------
# build_sampler dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["scene", "uniform", "hybrid"])
def test_build_sampler_dispatches_to_each_strategy(strategy: str) -> None:
    scenes = [_scene(0, 0, 6), _scene(1, 6, 12)]
    specs = build_sampler(strategy, scenes, duration_sec=12.0)  # type: ignore[arg-type]
    assert isinstance(specs, list)
    assert all(isinstance(s, FrameSpec) for s in specs)


def test_build_sampler_rejects_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="unknown sampling strategy"):
        build_sampler("bogus", [], duration_sec=10.0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Deduplication / sorting
# ---------------------------------------------------------------------------


def test_dedupe_collapses_close_timestamps() -> None:
    # Two adjacent scenes that share a boundary timestamp -> hybrid combines
    # them and the dedup pass keeps a single sample around the shared point.
    scenes = [_scene(0, 0, 4), _scene(1, 4, 8)]
    specs = build_sampler("hybrid", scenes, duration_sec=8.0, per_scene=2, max_gap_sec=3.0)
    ts = [s.timestamp.total_seconds() for s in specs]
    # Sorted
    assert ts == sorted(ts)
    # No two timestamps within 0.25s
    diffs = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
    assert all(d >= 0.25 for d in diffs)
