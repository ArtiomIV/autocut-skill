"""Unit tests for ``autocut.video.probe`` — JSON parsing and error paths.

These tests mock ``subprocess.run`` so they don't require ffmpeg to be
installed and run identically across all CI environments.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from autocut.video.ffmpeg_path import FFmpegResolveError
from autocut.video.probe import FFprobeError, probe_video


def _valid_ffprobe_payload(duration: float = 60.0) -> dict[str, Any]:
    return {
        "format": {
            "duration": str(duration),
            "size": "1048576",
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
        },
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30000/1001",
                "r_frame_rate": "30000/1001",
            },
            {"codec_type": "audio", "codec_name": "aac"},
        ],
    }


@pytest.fixture
def fake_video(tmp_path: Path) -> Path:
    p = tmp_path / "fake.mp4"
    p.write_bytes(b"not a real video, just a placeholder")
    return p


def _patch_ffprobe(
    monkeypatch: pytest.MonkeyPatch,
    *,
    stdout: str,
    returncode: int = 0,
    stderr: str = "",
) -> None:
    def fake_run(args: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=args, returncode=returncode, stdout=stdout, stderr=stderr
        )

    monkeypatch.setattr("autocut.video.probe.subprocess.run", fake_run)
    monkeypatch.setattr("autocut.video.probe.ffprobe_binary", lambda explicit=None: "/fake/ffprobe")


def test_parses_full_payload(monkeypatch: pytest.MonkeyPatch, fake_video: Path) -> None:
    _patch_ffprobe(monkeypatch, stdout=json.dumps(_valid_ffprobe_payload()))
    meta = probe_video(fake_video)
    assert meta.width == 1920
    assert meta.height == 1080
    assert meta.video_codec == "h264"
    assert meta.audio_codec == "aac"
    assert meta.fps == pytest.approx(30000 / 1001)
    assert meta.container == "mov"
    assert meta.duration_sec == pytest.approx(60.0)
    assert meta.aspect_ratio == pytest.approx(1920 / 1080)


def test_handles_video_without_audio(monkeypatch: pytest.MonkeyPatch, fake_video: Path) -> None:
    payload = _valid_ffprobe_payload()
    payload["streams"] = [payload["streams"][0]]
    _patch_ffprobe(monkeypatch, stdout=json.dumps(payload))
    meta = probe_video(fake_video)
    assert meta.audio_codec is None


def test_falls_back_to_filesystem_size(monkeypatch: pytest.MonkeyPatch, fake_video: Path) -> None:
    payload = _valid_ffprobe_payload()
    payload["format"].pop("size")
    _patch_ffprobe(monkeypatch, stdout=json.dumps(payload))
    meta = probe_video(fake_video)
    assert meta.size_bytes == fake_video.stat().st_size


def test_falls_back_to_extension_when_format_name_missing(
    monkeypatch: pytest.MonkeyPatch, fake_video: Path
) -> None:
    payload = _valid_ffprobe_payload()
    payload["format"].pop("format_name")
    _patch_ffprobe(monkeypatch, stdout=json.dumps(payload))
    assert probe_video(fake_video).container == "mp4"


def test_falls_back_to_r_frame_rate_when_avg_is_zero(
    monkeypatch: pytest.MonkeyPatch, fake_video: Path
) -> None:
    payload = _valid_ffprobe_payload()
    payload["streams"][0]["avg_frame_rate"] = "0/0"
    payload["streams"][0]["r_frame_rate"] = "60/1"
    _patch_ffprobe(monkeypatch, stdout=json.dumps(payload))
    assert probe_video(fake_video).fps == 60.0


def test_uses_stream_duration_if_format_duration_missing(
    monkeypatch: pytest.MonkeyPatch, fake_video: Path
) -> None:
    payload = _valid_ffprobe_payload()
    payload["format"].pop("duration")
    payload["streams"][0]["duration"] = "12.5"
    _patch_ffprobe(monkeypatch, stdout=json.dumps(payload))
    assert probe_video(fake_video).duration_sec == pytest.approx(12.5)


def test_raises_when_file_missing(tmp_path: Path) -> None:
    with pytest.raises(FFprobeError, match="does not exist"):
        probe_video(tmp_path / "no_such_file.mp4")


def test_raises_when_ffprobe_unavailable(monkeypatch: pytest.MonkeyPatch, fake_video: Path) -> None:
    # Neither system PATH nor the bundled fallback could supply ffprobe.
    def boom(explicit: str | None = None) -> str:
        raise FFmpegResolveError("no ffprobe on PATH and bundled fetch failed")

    monkeypatch.setattr("autocut.video.probe.ffprobe_binary", boom)
    with pytest.raises(FFprobeError, match="ffprobe not found"):
        probe_video(fake_video)


def test_raises_when_ffprobe_returns_nonzero(
    monkeypatch: pytest.MonkeyPatch, fake_video: Path
) -> None:
    _patch_ffprobe(monkeypatch, stdout="", returncode=1, stderr="boom")
    with pytest.raises(FFprobeError, match="exited with 1"):
        probe_video(fake_video)


def test_raises_on_invalid_json(monkeypatch: pytest.MonkeyPatch, fake_video: Path) -> None:
    _patch_ffprobe(monkeypatch, stdout="not json")
    with pytest.raises(FFprobeError, match="invalid JSON"):
        probe_video(fake_video)


def test_raises_when_no_video_stream(monkeypatch: pytest.MonkeyPatch, fake_video: Path) -> None:
    payload = _valid_ffprobe_payload()
    payload["streams"] = [s for s in payload["streams"] if s["codec_type"] != "video"]
    _patch_ffprobe(monkeypatch, stdout=json.dumps(payload))
    with pytest.raises(FFprobeError, match="no video stream"):
        probe_video(fake_video)


def test_raises_when_fps_unparseable(monkeypatch: pytest.MonkeyPatch, fake_video: Path) -> None:
    payload = _valid_ffprobe_payload()
    payload["streams"][0]["avg_frame_rate"] = "0/0"
    payload["streams"][0]["r_frame_rate"] = "0/0"
    _patch_ffprobe(monkeypatch, stdout=json.dumps(payload))
    with pytest.raises(FFprobeError, match="frame rate"):
        probe_video(fake_video)


def test_raises_when_subprocess_times_out(
    monkeypatch: pytest.MonkeyPatch, fake_video: Path
) -> None:
    def fake_run(args: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=args, timeout=30)

    monkeypatch.setattr("autocut.video.probe.subprocess.run", fake_run)
    monkeypatch.setattr("autocut.video.probe.ffprobe_binary", lambda explicit=None: "/fake/ffprobe")
    with pytest.raises(FFprobeError, match="timed out"):
        probe_video(fake_video)
