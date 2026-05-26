"""Unit tests for ``autocut.video.motion`` — argument validation only.

The actual optical-flow pipeline requires ffmpeg + a real video; see
``tests/integration/test_video_motion.py`` for the end-to-end coverage.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autocut.video.motion import (
    MotionAnalysisError,
    MotionSample,
    compute_motion_profile,
)


def test_motion_sample_is_immutable() -> None:
    s = MotionSample(timestamp_sec=1.0, magnitude=0.5)
    with pytest.raises(AttributeError):
        s.magnitude = 99.0  # type: ignore[misc]


def test_rejects_non_positive_target_fps(tmp_path: Path) -> None:
    fake = tmp_path / "video.mp4"
    fake.write_bytes(b"\x00")
    with pytest.raises(ValueError, match="target_fps"):
        compute_motion_profile(fake, target_fps=0.0)
    with pytest.raises(ValueError, match="target_fps"):
        compute_motion_profile(fake, target_fps=-1.5)


def test_rejects_tiny_downscale_edge(tmp_path: Path) -> None:
    fake = tmp_path / "video.mp4"
    fake.write_bytes(b"\x00")
    with pytest.raises(ValueError, match="downscale_long_edge"):
        compute_motion_profile(fake, downscale_long_edge=16)


def test_raises_for_missing_input_file(tmp_path: Path) -> None:
    with pytest.raises(MotionAnalysisError, match="not found"):
        compute_motion_profile(tmp_path / "does_not_exist.mp4")
