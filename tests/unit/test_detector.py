"""Unit tests for ``autocut.content.detector`` — Phase E.

Covers the pure helpers (stratified random picker, DetectionResult
schema). The async orchestrator ``detect_content_hint`` is left to
integration tests that stub the provider + filesystem.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autocut.content.detector import (
    DETECTION_KEYFRAME_COUNT,
    _file_hash_seed,
    _pick_stratified_timestamps,
)
from autocut.models import ContentHint, DetectionResult

# ---------------------------------------------------------------------------
# Stratified random timestamp picker
# ---------------------------------------------------------------------------


def _make_video(tmp_path: Path, *, content: bytes = b"\x00" * 4096) -> Path:
    # Ensure the parent directory exists so callers that nest paths
    # (``tmp_path / "a" / "fake.mp4"``) don't trip on a missing dir.
    tmp_path.mkdir(parents=True, exist_ok=True)
    video = tmp_path / "fake.mp4"
    video.write_bytes(content)
    return video


def test_stratified_picks_n_timestamps_inside_bounds(tmp_path: Path) -> None:
    video = _make_video(tmp_path)
    timestamps = _pick_stratified_timestamps(duration_sec=60.0, n=9, video_path=video)
    assert len(timestamps) == 9
    for ts in timestamps:
        assert 0.0 < ts < 60.0


def test_stratified_default_count_is_nine() -> None:
    # The exported default must stay 9 — STATUS.md ships the detection prompt
    # tuned for this count, downstream tests rely on it.
    assert DETECTION_KEYFRAME_COUNT == 9


def test_stratified_segments_are_each_covered_exactly_once(tmp_path: Path) -> None:
    video = _make_video(tmp_path)
    duration = 90.0
    n = 9
    timestamps = _pick_stratified_timestamps(duration_sec=duration, n=n, video_path=video)
    segment_len = duration / n
    for i, ts in enumerate(timestamps):
        seg_low = i * segment_len
        seg_high = (i + 1) * segment_len
        # Each timestamp must fall inside its own segment (stratification
        # guarantee). Padding may shrink the window but never invert it.
        assert seg_low <= ts <= seg_high, f"frame {i}: {ts} outside [{seg_low}, {seg_high}]"


def test_stratified_is_deterministic_for_same_file(tmp_path: Path) -> None:
    video = _make_video(tmp_path, content=b"\x42" * 8192)
    first = _pick_stratified_timestamps(duration_sec=60.0, n=9, video_path=video)
    second = _pick_stratified_timestamps(duration_sec=60.0, n=9, video_path=video)
    assert first == second


def test_stratified_differs_between_files(tmp_path: Path) -> None:
    video_a = _make_video(tmp_path / "a", content=b"\x42" * 8192)
    video_b = _make_video(tmp_path / "b", content=b"\xab" * 8192)
    # Different bytes → different SHA-256 → different seed → different draws.
    a = _pick_stratified_timestamps(duration_sec=60.0, n=9, video_path=video_a)
    b = _pick_stratified_timestamps(duration_sec=60.0, n=9, video_path=video_b)
    assert a != b


def test_stratified_short_video_falls_back_to_midpoints(tmp_path: Path) -> None:
    # A 2-second video sliced into 9 segments → segments are 0.22 s each;
    # with 10% padding the window collapses to ~0.18 s → still pickable.
    # Pick something tighter: 9 segments out of 0.1 s overall → fallback.
    video = _make_video(tmp_path)
    timestamps = _pick_stratified_timestamps(duration_sec=0.05, n=9, video_path=video)
    assert len(timestamps) == 9
    # All timestamps stay inside [0, duration] even in the degenerate case.
    for ts in timestamps:
        assert 0.0 <= ts <= 0.05


def test_stratified_rejects_non_positive_duration(tmp_path: Path) -> None:
    video = _make_video(tmp_path)
    with pytest.raises(ValueError, match="duration_sec must be positive"):
        _pick_stratified_timestamps(duration_sec=0.0, n=9, video_path=video)


def test_stratified_rejects_zero_n(tmp_path: Path) -> None:
    video = _make_video(tmp_path)
    with pytest.raises(ValueError, match="n must be at least 1"):
        _pick_stratified_timestamps(duration_sec=10.0, n=0, video_path=video)


def test_file_hash_seed_is_zero_when_file_missing(tmp_path: Path) -> None:
    # No graceful return path is the goal here — we want zero, not an exception.
    seed = _file_hash_seed(tmp_path / "does_not_exist.mp4")
    assert seed == 0


# ---------------------------------------------------------------------------
# DetectionResult schema
# ---------------------------------------------------------------------------


def test_detection_result_rejects_auto_as_output() -> None:
    with pytest.raises(ValueError, match="must commit"):
        DetectionResult(content_hint=ContentHint.auto, confidence=0.9, reasoning="bug")


def test_detection_result_accepts_committed_categories() -> None:
    for hint in (
        ContentHint.highlights,
        ContentHint.talk,
        ContentHint.hybrid,
    ):
        DetectionResult(content_hint=hint, confidence=0.5, reasoning="ok")


def test_detection_result_clamps_confidence_to_unit_interval() -> None:
    with pytest.raises(ValueError):
        DetectionResult(content_hint=ContentHint.talk, confidence=1.5, reasoning="overshoot")
    with pytest.raises(ValueError):
        DetectionResult(content_hint=ContentHint.talk, confidence=-0.1, reasoning="negative")


def test_detection_result_reasoning_has_length_cap() -> None:
    # Anything past 300 chars is over the configured Field max_length.
    too_long = "x" * 400
    with pytest.raises(ValueError):
        DetectionResult(content_hint=ContentHint.talk, confidence=0.5, reasoning=too_long)
