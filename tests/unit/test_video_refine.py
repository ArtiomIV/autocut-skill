"""Unit tests for the impact-snap decision (``_decide_impact``).

These exercise the pure peak logic with synthetic motion profiles, so they run
without ffmpeg/opencv. The slice extraction and ffmpeg-backed analysis in
``refine_start_to_impact`` are integration-level and covered by the live E2E.
"""

from __future__ import annotations

from autocut.video.motion import MotionSample
from autocut.video.refine import RefineConfig, _decide_impact


def _profile(magnitudes: list[float], *, fps: float = 15.0) -> list[MotionSample]:
    """Build a slice-relative motion profile sampled at ``fps``."""
    dt = 1.0 / fps
    return [MotionSample(timestamp_sec=(i + 1) * dt, magnitude=m) for i, m in enumerate(magnitudes)]


def test_snaps_start_to_a_dominant_spike() -> None:
    # Flat low motion (the count) with a sharp spike near the front (the punch).
    mags = [0.1] * 60
    mags[10] = 5.0  # spike ~0.73s into the window
    mags[11] = 4.0
    motion = _profile(mags)
    # Window starts at 5:25 (325s); the VLM (mis)placed the start at 5:37 (337s).
    result = _decide_impact(
        motion, [], win_start=325.0, clip_start_sec=337.0, clip_end_sec=347.0, cfg=RefineConfig()
    )
    assert result.moved is True
    assert result.impact_sec is not None
    # Impact lands ~325.7s and we keep a 0.5s lead -> well before the 337s start.
    assert 325.0 <= result.refined_start_sec < 337.0
    assert result.refined_start_sec < result.impact_sec


def test_flat_motion_leaves_boundary_unchanged() -> None:
    # A ring entrance: smooth, low-variance motion, no sharp impact.
    motion = _profile([1.0, 1.05, 0.98, 1.02, 1.01, 0.99] * 10)
    result = _decide_impact(
        motion, [], win_start=1980.0, clip_start_sec=1992.0, clip_end_sec=2010.0, cfg=RefineConfig()
    )
    assert result.moved is False
    assert result.refined_start_sec == 1992.0
    assert "dominant" in result.reason


def test_spike_after_the_vlm_start_is_not_moved() -> None:
    # The dominant spike sits AFTER the VLM start -> nothing to gain, leave it.
    mags = [0.1] * 60
    mags[58] = 6.0  # ~3.9s in, i.e. after clip_start when window is short
    motion = _profile(mags)
    result = _decide_impact(
        motion, [], win_start=100.0, clip_start_sec=101.0, clip_end_sec=120.0, cfg=RefineConfig()
    )
    assert result.moved is False


def test_audio_onset_near_spike_is_flagged_confirmed() -> None:
    mags = [0.1] * 60
    mags[10] = 5.0
    motion = _profile(mags)
    # An onset at ~0.73s (slice-relative), right on the spike's rising edge.
    onsets = [0.70]
    result = _decide_impact(
        motion,
        onsets,
        win_start=325.0,
        clip_start_sec=337.0,
        clip_end_sec=347.0,
        cfg=RefineConfig(),
    )
    assert result.moved is True
    assert result.audio_confirmed is True


def test_too_few_samples_is_safe() -> None:
    result = _decide_impact(
        _profile([1.0, 2.0]),
        [],
        win_start=0.0,
        clip_start_sec=5.0,
        clip_end_sec=10.0,
        cfg=RefineConfig(),
    )
    assert result.moved is False
