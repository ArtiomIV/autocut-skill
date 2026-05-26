"""Shared fixtures for integration tests that require ffmpeg.

Tests in this folder need the ffmpeg binary in PATH. When ffmpeg is missing
(e.g. a developer who hasn't installed it yet), the whole folder is skipped
rather than producing confusing failures. CI installs ffmpeg explicitly so
this skip never fires there.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)


def _make_lavfi_video(out_path: Path, duration: float, size: str, fps: int = 30) -> Path:
    """Create a deterministic test video using ffmpeg's lavfi source."""
    args = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration={duration}:size={size}:rate={fps}",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-g",
        str(fps),  # one keyframe per second -> tight stream-copy cuts
        str(out_path),
    ]
    completed = subprocess.run(args, capture_output=True, text=True, check=False, timeout=60)
    if completed.returncode != 0 or not out_path.is_file():
        raise RuntimeError(f"failed to build fixture video: {completed.stderr.strip()}")
    return out_path


@pytest.fixture
def short_video(tmp_path: Path) -> Path:
    """A 4s 320x240 30fps H.264 video. Small and fast for most tests."""
    return _make_lavfi_video(tmp_path / "short.mp4", duration=4.0, size="320x240")


@pytest.fixture
def medium_video(tmp_path: Path) -> Path:
    """A 10s 640x480 video, big enough to exercise sampling at meaningful counts."""
    return _make_lavfi_video(tmp_path / "medium.mp4", duration=10.0, size="640x480")


def _make_video_with_audio_step(out_path: Path, duration: float = 4.0) -> Path:
    """Create a deterministic test video that carries a stepped audio track:
    silence for the first half, then a 440 Hz sine for the second half. Used
    by audio-peak tests to verify onset detection on a known signal.
    """
    half = duration / 2.0
    # Single aevalsrc with a conditional expression: zero before ``half``
    # seconds, then a 440 Hz sine. Avoids the concat-demuxer filter-graph
    # plumbing entirely.
    audio_src = (
        f"aevalsrc=if(gte(t\\,{half})\\,0.5*sin(2*PI*440*t)\\,0):"
        f"duration={duration}:sample_rate=16000"
    )
    args = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration={duration}:size=320x240:rate=30",
        "-f",
        "lavfi",
        "-i",
        audio_src,
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-c:a",
        "aac",
        "-shortest",
        str(out_path),
    ]
    completed = subprocess.run(args, capture_output=True, text=True, check=False, timeout=60)
    if completed.returncode != 0 or not out_path.is_file():
        raise RuntimeError(f"failed to build audio fixture: {completed.stderr.strip()}")
    return out_path


@pytest.fixture
def short_video_with_audio_step(tmp_path: Path) -> Path:
    """A 4s 320x240 video with audio: silence for 2s, then a 440 Hz sine for 2s.

    The clear silence→tone transition guarantees the onset detector has a
    real, predictable event to find.
    """
    return _make_video_with_audio_step(tmp_path / "step.mp4", duration=4.0)
