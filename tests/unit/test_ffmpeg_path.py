"""Unit tests for the ffmpeg/ffprobe resolver (system PATH vs bundled fallback)."""

from __future__ import annotations

import pytest

from autocut.video import ffmpeg_path
from autocut.video.ffmpeg_path import (
    FFmpegResolveError,
    describe_ffmpeg,
    ffmpeg_binary,
    ffprobe_binary,
)


def test_explicit_path_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    # An explicit path is returned verbatim, even if the system has ffmpeg.
    monkeypatch.setattr(ffmpeg_path.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    assert ffmpeg_binary("/custom/ffmpeg") == "/custom/ffmpeg"


def test_system_path_used_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ffmpeg_path.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert ffmpeg_binary() == "/usr/bin/ffmpeg"
    assert ffprobe_binary() == "/usr/bin/ffprobe"


def test_falls_back_to_bundled_when_system_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ffmpeg_path.shutil, "which", lambda _: None)
    monkeypatch.setattr(
        ffmpeg_path, "_bundled_binaries", lambda: ("/bundle/ffmpeg", "/bundle/ffprobe")
    )
    assert ffmpeg_binary() == "/bundle/ffmpeg"
    assert ffprobe_binary() == "/bundle/ffprobe"


def test_describe_reports_system_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ffmpeg_path.shutil, "which", lambda name: f"/usr/bin/{name}")
    path, source = describe_ffmpeg()
    assert source == "system"
    assert path == "/usr/bin/ffmpeg"


def test_resolve_error_when_bundled_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ffmpeg_path.shutil, "which", lambda _: None)

    def boom() -> tuple[str, str]:
        raise FFmpegResolveError("download failed")

    monkeypatch.setattr(ffmpeg_path, "_bundled_binaries", boom)
    with pytest.raises(FFmpegResolveError):
        ffmpeg_binary()
