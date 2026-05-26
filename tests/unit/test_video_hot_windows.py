"""Unit tests for ``autocut.video.hot_windows`` — pure logic, no I/O."""

from __future__ import annotations

import pytest

from autocut.video.audio_peaks import AudioSample
from autocut.video.hot_windows import HotWindow, HotWindowConfig, find_hot_windows
from autocut.video.motion import MotionSample

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _motion_at(ts: float, mag: float) -> MotionSample:
    return MotionSample(timestamp_sec=ts, magnitude=mag)


def _audio_at(ts: float, rms: float, *, is_onset: bool = False) -> AudioSample:
    return AudioSample(timestamp_sec=ts, rms=rms, is_onset=is_onset)


def _flat_motion(duration_sec: float, mag: float, *, fps: float = 10.0) -> list[MotionSample]:
    n = int(duration_sec * fps)
    return [_motion_at(i / fps, mag) for i in range(n)]


# ---------------------------------------------------------------------------
# HotWindow shape
# ---------------------------------------------------------------------------


def test_hotwindow_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError, match="end must be strictly greater"):
        HotWindow(start_sec=5.0, end_sec=5.0, score=1.0)
    with pytest.raises(ValueError, match="end must be strictly greater"):
        HotWindow(start_sec=5.0, end_sec=3.0, score=1.0)


def test_hotwindow_duration_is_difference() -> None:
    w = HotWindow(start_sec=2.0, end_sec=7.5, score=1.0)
    assert w.duration_sec == 5.5


# ---------------------------------------------------------------------------
# Empty / degenerate inputs
# ---------------------------------------------------------------------------


def test_find_hot_windows_empty_when_no_signals() -> None:
    assert find_hot_windows([], [], video_duration_sec=10.0) == []


def test_find_hot_windows_rejects_zero_duration() -> None:
    with pytest.raises(ValueError, match="video_duration_sec"):
        find_hot_windows([], [], video_duration_sec=0)


def test_find_hot_windows_flat_motion_yields_nothing() -> None:
    # Every motion sample identical -> std == 0 -> no signal.
    motion = _flat_motion(duration_sec=10.0, mag=1.0)
    assert find_hot_windows(motion, [], video_duration_sec=10.0) == []


# ---------------------------------------------------------------------------
# Motion-only windows
# ---------------------------------------------------------------------------


def test_motion_burst_in_middle_produces_a_window() -> None:
    # 10 s of baseline 0.1, with a 2 s burst at 5.0 magnitude from t=4 to t=6.
    motion: list[MotionSample] = []
    for i in range(100):  # 10 s @ 10 fps
        t = i / 10.0
        mag = 5.0 if 4.0 <= t < 6.0 else 0.1
        motion.append(_motion_at(t, mag))

    windows = find_hot_windows(motion, [], video_duration_sec=10.0)
    assert windows, "expected at least one hot window over the burst"
    # The burst is somewhere between t=4 and t=6; at least one window must
    # overlap that range.
    assert any(w.start_sec < 6.0 and w.end_sec > 4.0 for w in windows)


def test_motion_window_score_grows_with_intensity() -> None:
    # Two bursts: a mild one and a strong one. Strong one should score higher.
    motion: list[MotionSample] = []
    for i in range(100):
        t = i / 10.0
        if 2.0 <= t < 3.0:
            mag = 2.0
        elif 6.0 <= t < 7.0:
            mag = 10.0
        else:
            mag = 0.1
        motion.append(_motion_at(t, mag))

    # Lower z_threshold so the mild burst still clears the bar; we care
    # about score *ranking*, not about which bursts pass selection.
    windows = find_hot_windows(
        motion,
        [],
        video_duration_sec=10.0,
        config=HotWindowConfig(
            motion_window_sec=1.0,
            motion_stride_sec=0.5,
            motion_z_threshold=0.2,
        ),
    )
    mild = max((w for w in windows if w.start_sec < 4.0), key=lambda w: w.score, default=None)
    strong = max((w for w in windows if w.start_sec >= 4.0), key=lambda w: w.score, default=None)
    assert mild is not None and strong is not None
    assert strong.score > mild.score


# ---------------------------------------------------------------------------
# Audio-only windows
# ---------------------------------------------------------------------------


def test_audio_onset_produces_window_with_lookback() -> None:
    onset = _audio_at(ts=5.0, rms=0.8, is_onset=True)
    windows = find_hot_windows(
        [],
        [onset],
        video_duration_sec=20.0,
        config=HotWindowConfig(audio_window_sec=2.0, audio_lookback_sec=1.0),
    )
    assert len(windows) == 1
    w = windows[0]
    # Centre is (5.0 - 1.0) = 4.0, width 2.0 -> [3.0, 5.0].
    assert w.start_sec == pytest.approx(3.0)
    assert w.end_sec == pytest.approx(5.0)
    assert w.score == pytest.approx(0.8)


def test_audio_non_onset_samples_ignored() -> None:
    audio = [_audio_at(ts=t, rms=0.5, is_onset=False) for t in (1.0, 2.0, 3.0)]
    assert find_hot_windows([], audio, video_duration_sec=10.0) == []


# ---------------------------------------------------------------------------
# Clamping and merging
# ---------------------------------------------------------------------------


def test_windows_are_clamped_to_video_duration() -> None:
    # Onset near the end produces a window that runs past the video duration.
    onset = _audio_at(ts=9.5, rms=0.5, is_onset=True)
    windows = find_hot_windows(
        [],
        [onset],
        video_duration_sec=10.0,
        config=HotWindowConfig(audio_window_sec=4.0, audio_lookback_sec=0.0),
    )
    assert len(windows) == 1
    assert windows[0].end_sec <= 10.0
    assert windows[0].start_sec >= 0.0


def test_adjacent_windows_merge_within_gap() -> None:
    # Two onsets close together -> one merged window.
    onsets = [
        _audio_at(ts=3.0, rms=0.5, is_onset=True),
        _audio_at(ts=4.0, rms=0.7, is_onset=True),
    ]
    windows = find_hot_windows(
        [],
        onsets,
        video_duration_sec=20.0,
        config=HotWindowConfig(audio_window_sec=1.0, audio_lookback_sec=0.0, merge_gap_sec=1.5),
    )
    assert len(windows) == 1
    # Scores are summed when merging.
    assert windows[0].score == pytest.approx(1.2)


def test_far_apart_windows_do_not_merge() -> None:
    onsets = [
        _audio_at(ts=2.0, rms=0.5, is_onset=True),
        _audio_at(ts=15.0, rms=0.5, is_onset=True),
    ]
    windows = find_hot_windows(
        [],
        onsets,
        video_duration_sec=20.0,
        config=HotWindowConfig(audio_window_sec=1.0, audio_lookback_sec=0.0, merge_gap_sec=2.0),
    )
    assert len(windows) == 2


# ---------------------------------------------------------------------------
# Combined motion + audio
# ---------------------------------------------------------------------------


def test_combined_motion_and_audio_yield_windows_from_both() -> None:
    # Motion burst at t=2-3, audio onset at t=8.
    motion: list[MotionSample] = []
    for i in range(100):
        t = i / 10.0
        mag = 5.0 if 2.0 <= t < 3.0 else 0.1
        motion.append(_motion_at(t, mag))
    audio = [_audio_at(ts=8.0, rms=0.5, is_onset=True)]

    windows = find_hot_windows(
        motion,
        audio,
        video_duration_sec=10.0,
        config=HotWindowConfig(motion_window_sec=0.5, motion_stride_sec=0.5),
    )
    # Should have at least one window in the motion region AND one in
    # the audio region.
    assert any(w.start_sec < 4.0 for w in windows), windows
    assert any(w.start_sec > 5.0 for w in windows), windows
