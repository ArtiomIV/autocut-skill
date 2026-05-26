"""Unit tests for ``autocut.video.frame_sampler`` — pure logic, no I/O."""

from __future__ import annotations

from datetime import timedelta

import pytest

from autocut.models import Scene
from autocut.video.frame_sampler import (
    FrameSpec,
    build_sampler,
    sample_hybrid,
    sample_motion,
    sample_scene_based,
    sample_uniform,
)
from autocut.video.hot_windows import HotWindow


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
# build_sampler safety nets (A.2): short-video reroute + min_keyframes floor
# ---------------------------------------------------------------------------


def test_build_sampler_reroutes_short_hybrid_to_dense_uniform() -> None:
    # 41s video with hybrid + a single uncut scene would yield only 2 samples
    # via scene logic. The short-video reroute uses 1.5s uniform instead.
    scenes = [_scene(0, 0, 41)]
    specs = build_sampler("hybrid", scenes, duration_sec=41.0)
    # All samples should be scene_index=-1 (uniform, not scene-based).
    assert all(s.scene_index == -1 for s in specs)
    # 41s / 1.5s ≈ 27 samples.
    assert 25 <= len(specs) <= 28


def test_build_sampler_keeps_long_hybrid_as_scene_based() -> None:
    # 200s video (above the short-video threshold) uses real hybrid logic.
    scenes = [_scene(0, 0, 100), _scene(1, 100, 200)]
    specs = build_sampler("hybrid", scenes, duration_sec=200.0)
    # At least one spec must carry a real scene_index (hybrid base samples).
    assert any(s.scene_index >= 0 for s in specs)


def test_build_sampler_enforces_min_keyframes_floor() -> None:
    # Uniform with interval=10s on a 5s video naturally yields 0 samples.
    # The min_keyframes floor densifies to at least the requested count.
    specs = build_sampler("uniform", [], duration_sec=5.0, interval_sec=10.0, min_keyframes=4)
    assert len(specs) >= 4


def test_build_sampler_min_keyframes_does_not_override_when_already_dense() -> None:
    # Hybrid reroute on 41s already produces ~27 samples; min_keyframes=3
    # must not shrink that.
    scenes = [_scene(0, 0, 41)]
    specs = build_sampler("hybrid", scenes, duration_sec=41.0, min_keyframes=3)
    assert len(specs) > 10


# ---------------------------------------------------------------------------
# sample_motion / motion strategy (D.4)
# ---------------------------------------------------------------------------


def test_sample_motion_baseline_only_when_no_hot_windows() -> None:
    # No hot windows → falls back to pure baseline uniform sampling.
    specs = sample_motion([], duration_sec=20.0, base_interval_sec=4.0, dense_interval_sec=0.5)
    # 20 s / 4 s = 5 samples at 2, 6, 10, 14, 18.
    assert len(specs) == 5
    assert all(s.scene_index == -1 for s in specs)


def test_sample_motion_densifies_inside_hot_window() -> None:
    # Hot window from 4-6 s with 1 s padding → dense range 3-7 s.
    hot = [HotWindow(start_sec=4.0, end_sec=6.0, score=1.0)]
    specs = sample_motion(
        hot,
        duration_sec=20.0,
        base_interval_sec=4.0,
        dense_interval_sec=0.5,
        hot_padding_sec=1.0,
    )
    # Baseline at 2, 6, 10, 14, 18 (5) + ~8 dense in [3,7) at 0.5 stride.
    # Dedup tolerance can collapse one or two boundary collisions.
    assert len(specs) >= 11
    # At least 5 samples should sit in the dense region.
    dense_count = sum(1 for s in specs if 3.0 <= s.timestamp.total_seconds() < 7.0)
    assert dense_count >= 5


def test_sample_motion_padding_is_clamped_at_video_bounds() -> None:
    hot = [HotWindow(start_sec=0.0, end_sec=2.0, score=1.0)]
    specs = sample_motion(
        hot,
        duration_sec=10.0,
        base_interval_sec=4.0,
        dense_interval_sec=0.5,
        hot_padding_sec=5.0,  # would overshoot to -5, but should clamp at 0
    )
    # Nothing should be at negative time.
    assert all(s.timestamp.total_seconds() >= 0 for s in specs)
    # First sample should be at or after 0.
    assert specs[0].timestamp.total_seconds() >= 0


def test_sample_motion_rejects_invalid_intervals() -> None:
    with pytest.raises(ValueError, match="intervals"):
        sample_motion([], duration_sec=10.0, base_interval_sec=0)
    with pytest.raises(ValueError, match="intervals"):
        sample_motion([], duration_sec=10.0, dense_interval_sec=-1)
    with pytest.raises(ValueError, match="hot_padding_sec"):
        sample_motion([], duration_sec=10.0, hot_padding_sec=-0.1)
    with pytest.raises(ValueError, match="duration_sec"):
        sample_motion([], duration_sec=0)


def test_build_sampler_motion_requires_hot_windows() -> None:
    with pytest.raises(ValueError, match="hot_windows"):
        build_sampler("motion", [], duration_sec=10.0)


def test_build_sampler_motion_dispatches_correctly() -> None:
    hot = [HotWindow(start_sec=2.0, end_sec=4.0, score=1.0)]
    specs = build_sampler("motion", [], duration_sec=10.0, hot_windows=hot)
    assert len(specs) >= 3
    assert all(isinstance(s, FrameSpec) for s in specs)


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
