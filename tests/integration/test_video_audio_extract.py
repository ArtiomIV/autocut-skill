"""Tests for ``autocut.video.audio_extract`` (requires ffmpeg)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from autocut.video.audio_extract import AudioExtractionError, extract_audio_for_vlm
from autocut.video.ffmpeg_path import ffprobe_binary

pytestmark = pytest.mark.integration


def _audio_stream_info(path: Path) -> tuple[str, int]:
    """Return (codec_name, channels) of the first audio stream via ffprobe."""
    out = subprocess.run(
        [
            ffprobe_binary(None),
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,channels",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = out.stdout.split()
    return lines[0], int(lines[1])


def test_extract_produces_mono_mp3(short_video_with_audio_step: Path, tmp_path: Path) -> None:
    out = extract_audio_for_vlm(short_video_with_audio_step, tmp_path / "a.mp3")
    assert out.is_file()
    assert out.stat().st_size > 0
    codec, channels = _audio_stream_info(out)
    assert codec == "mp3"
    assert channels == 1  # downmixed to mono


def test_extract_missing_source_raises(tmp_path: Path) -> None:
    with pytest.raises(AudioExtractionError, match="not found"):
        extract_audio_for_vlm(tmp_path / "nope.mp4", tmp_path / "a.mp3")


def test_extract_from_video_without_audio_raises(short_video: Path, tmp_path: Path) -> None:
    # short_video is a silent lavfi clip with no audio stream → ffmpeg can't
    # produce an audio output and we surface a clear error.
    with pytest.raises(AudioExtractionError):
        extract_audio_for_vlm(short_video, tmp_path / "a.mp3")
