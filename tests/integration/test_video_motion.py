"""End-to-end tests for ``autocut.video.motion`` (requires ffmpeg)."""

from __future__ import annotations

from pathlib import Path

import pytest

from autocut.video.motion import MotionSample, compute_motion_profile

pytestmark = pytest.mark.integration


def test_compute_motion_profile_returns_samples_for_moving_source(short_video: Path) -> None:
    # The lavfi ``testsrc`` fixture has a moving counter, so every consecutive
    # frame pair should produce a non-trivial motion magnitude.
    samples = compute_motion_profile(short_video, target_fps=10.0)
    assert len(samples) >= 30  # 4s at 10fps minus 1 (no flow for first frame)
    assert all(isinstance(s, MotionSample) for s in samples)
    # Magnitudes are always non-negative; for testsrc they should be > 0.
    assert all(s.magnitude >= 0 for s in samples)
    assert max(s.magnitude for s in samples) > 0.01


def test_compute_motion_profile_timestamps_are_monotonic(short_video: Path) -> None:
    samples = compute_motion_profile(short_video, target_fps=10.0)
    timestamps = [s.timestamp_sec for s in samples]
    assert timestamps == sorted(timestamps)
    # At 10 fps, consecutive samples should be exactly 0.1 s apart.
    deltas = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
    assert all(abs(d - 0.1) < 1e-6 for d in deltas)


def test_compute_motion_profile_respects_target_fps(short_video: Path) -> None:
    # Halve the rate; expect roughly half the samples.
    fast = compute_motion_profile(short_video, target_fps=10.0)
    slow = compute_motion_profile(short_video, target_fps=5.0)
    assert len(slow) <= len(fast) / 1.5  # generous bound for short-video edge effects
