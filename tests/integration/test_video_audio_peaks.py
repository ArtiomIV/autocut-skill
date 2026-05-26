"""End-to-end tests for ``autocut.video.audio_peaks`` (requires ffmpeg)."""

from __future__ import annotations

from pathlib import Path

import pytest

from autocut.video.audio_peaks import AudioSample, compute_audio_profile

pytestmark = pytest.mark.integration


def test_compute_audio_profile_returns_empty_for_silent_video(short_video: Path) -> None:
    # The ``short_video`` lavfi fixture has no audio stream at all → empty list.
    samples = compute_audio_profile(short_video)
    assert samples == []


def test_compute_audio_profile_detects_step_transition(
    short_video_with_audio_step: Path,
) -> None:
    samples = compute_audio_profile(short_video_with_audio_step, window_ms=100.0)
    assert len(samples) >= 30  # 4s / 0.1s = 40 windows minus trailing partial
    assert all(isinstance(s, AudioSample) for s in samples)

    # RMS should be near-zero in the silence half, clearly non-zero in the sine half.
    silence_rms = [s.rms for s in samples if s.timestamp_sec < 1.8]
    sine_rms = [s.rms for s in samples if 2.2 < s.timestamp_sec < 3.9]
    assert max(silence_rms) < 0.01
    assert min(sine_rms) > 0.1

    # An onset must be flagged around the 2 s mark.
    transition_onsets = [s for s in samples if s.is_onset and 1.8 <= s.timestamp_sec <= 2.4]
    assert transition_onsets, (
        "expected at least one onset around the silence→sine transition; "
        f"got onsets at {[round(s.timestamp_sec, 2) for s in samples if s.is_onset]}"
    )


def test_compute_audio_profile_timestamps_are_monotonic(
    short_video_with_audio_step: Path,
) -> None:
    samples = compute_audio_profile(short_video_with_audio_step, window_ms=100.0)
    timestamps = [s.timestamp_sec for s in samples]
    assert timestamps == sorted(timestamps)
    # 100 ms spacing.
    deltas = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
    assert all(abs(d - 0.1) < 1e-3 for d in deltas)
