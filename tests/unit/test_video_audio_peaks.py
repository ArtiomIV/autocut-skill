"""Unit tests for ``autocut.video.audio_peaks`` — pure-math helpers.

These tests exercise ``_windowed_rms`` and ``_detect_onsets`` directly
without going through ffmpeg, so they are fast and have no
external-binary dependency. Full pipeline coverage with real audio is
in ``tests/integration/test_video_audio_peaks.py``.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from autocut.video.audio_peaks import (
    AudioAnalysisError,
    AudioSample,
    _detect_onsets,
    _windowed_rms,
    compute_audio_profile,
)

# ---------------------------------------------------------------------------
# AudioSample shape
# ---------------------------------------------------------------------------


def test_audio_sample_is_immutable() -> None:
    s = AudioSample(timestamp_sec=1.0, rms=0.5, is_onset=False)
    with pytest.raises(AttributeError):
        s.rms = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def test_rejects_non_positive_sample_rate(tmp_path: Path) -> None:
    fake = tmp_path / "video.mp4"
    fake.write_bytes(b"\x00")
    with pytest.raises(ValueError, match="sample_rate_hz"):
        compute_audio_profile(fake, sample_rate_hz=0)


def test_rejects_non_positive_window_ms(tmp_path: Path) -> None:
    fake = tmp_path / "video.mp4"
    fake.write_bytes(b"\x00")
    with pytest.raises(ValueError, match="window_ms"):
        compute_audio_profile(fake, window_ms=0)


def test_rejects_non_positive_z_threshold(tmp_path: Path) -> None:
    fake = tmp_path / "video.mp4"
    fake.write_bytes(b"\x00")
    with pytest.raises(ValueError, match="onset_z_threshold"):
        compute_audio_profile(fake, onset_z_threshold=-1.0)


def test_raises_for_missing_input_file(tmp_path: Path) -> None:
    with pytest.raises(AudioAnalysisError, match="not found"):
        compute_audio_profile(tmp_path / "does_not_exist.mp4")


# ---------------------------------------------------------------------------
# _windowed_rms
# ---------------------------------------------------------------------------


def test_windowed_rms_constant_signal_yields_constant_rms() -> None:
    # 1 s of pure DC at amplitude 0.5 → every window must have RMS == 0.5.
    pcm = np.full(16_000, 0.5, dtype=np.float32)
    rms, ts = _windowed_rms(pcm, sample_rate_hz=16_000, window_ms=100.0)
    assert rms.size == 10  # 1000 ms / 100 ms
    np.testing.assert_allclose(rms, 0.5, atol=1e-6)
    # First window midpoint is 50 ms.
    np.testing.assert_allclose(ts[0], 0.05, atol=1e-6)
    np.testing.assert_allclose(ts[-1], 0.95, atol=1e-6)


def test_windowed_rms_returns_empty_for_too_short_input() -> None:
    pcm = np.zeros(10, dtype=np.float32)  # << one window of 1600 samples
    rms, ts = _windowed_rms(pcm, sample_rate_hz=16_000, window_ms=100.0)
    assert rms.size == 0
    assert ts.size == 0


def test_windowed_rms_handles_silence() -> None:
    pcm = np.zeros(16_000, dtype=np.float32)
    rms, _ = _windowed_rms(pcm, sample_rate_hz=16_000, window_ms=100.0)
    assert rms.size == 10
    assert (rms == 0).all()


# ---------------------------------------------------------------------------
# _detect_onsets
# ---------------------------------------------------------------------------


def test_detect_onsets_returns_no_flags_for_constant_rms() -> None:
    rms = np.full(20, 0.3, dtype=np.float32)
    ts = np.arange(20, dtype=np.float32) * 0.1
    flags = _detect_onsets(rms, ts, window_ms=100.0, z_threshold=1.5)
    assert flags.size == 20
    assert not flags.any()


def test_detect_onsets_flags_clear_energy_jump() -> None:
    # 1 s of low energy then 1 s of high energy. The transition window
    # should be flagged as an onset.
    rms = np.concatenate([np.full(10, 0.05), np.full(10, 0.5)]).astype(np.float32)
    ts = np.arange(20, dtype=np.float32) * 0.1
    flags = _detect_onsets(rms, ts, window_ms=100.0, z_threshold=1.5)
    # The transition is at index 10. Detector compares with index 8 (200ms back).
    assert flags[10:12].any(), f"expected onset around transition, got {flags}"


def test_detect_onsets_suppresses_adjacent_duplicates() -> None:
    # A short staircase of three rising plateaus. After non-maximum
    # suppression at 300 ms min separation, we keep at most one onset
    # per ~3 windows.
    plateau_a = np.full(5, 0.05, dtype=np.float32)
    plateau_b = np.full(5, 0.3, dtype=np.float32)
    plateau_c = np.full(5, 0.6, dtype=np.float32)
    rms = np.concatenate([plateau_a, plateau_b, plateau_c]).astype(np.float32)
    ts = np.arange(15, dtype=np.float32) * 0.1
    flags = _detect_onsets(rms, ts, window_ms=100.0, z_threshold=1.0)
    onset_indices = np.where(flags)[0]
    # Whatever the exact indices, NO two of them should be within 3
    # windows of each other (300 ms / 100 ms window = 3).
    for a, b in itertools.pairwise(onset_indices):
        assert b - a >= 3, f"onsets too close: {onset_indices}"


def test_detect_onsets_no_signal_returns_empty_flags() -> None:
    rms = np.zeros(5, dtype=np.float32)
    ts = np.zeros(5, dtype=np.float32)
    flags = _detect_onsets(rms, ts, window_ms=100.0, z_threshold=1.5)
    assert flags.size == 5
    assert not flags.any()
