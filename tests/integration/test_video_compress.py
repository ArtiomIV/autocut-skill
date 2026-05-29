"""Tests for ``autocut.video.compress`` (requires ffmpeg)."""

from __future__ import annotations

from pathlib import Path

import pytest

from autocut.video.compress import CompressionError, compress_for_vlm
from autocut.video.probe import probe_video

pytestmark = pytest.mark.integration


def test_compress_caps_long_edge_and_keeps_even_dims(medium_video: Path, tmp_path: Path) -> None:
    # medium_video is 640x480; capping the long edge at 320 must scale it down
    # to 320x240 (aspect preserved) with both dimensions even for H.264.
    out = compress_for_vlm(medium_video, tmp_path / "c.mp4", long_edge_px=320, fps=10)
    assert out.is_file()
    meta = probe_video(out)
    assert max(meta.width, meta.height) <= 320
    assert meta.width % 2 == 0 and meta.height % 2 == 0
    # fps must be capped at the requested rate (small tolerance for rounding).
    assert meta.fps <= 11


def test_compress_does_not_upscale_a_smaller_source(short_video: Path, tmp_path: Path) -> None:
    # short_video is 320x240; asking for a 640 long edge must NOT upscale it.
    out = compress_for_vlm(short_video, tmp_path / "c.mp4", long_edge_px=640)
    meta = probe_video(out)
    assert max(meta.width, meta.height) <= 320


def test_compress_missing_source_raises(tmp_path: Path) -> None:
    with pytest.raises(CompressionError):
        compress_for_vlm(tmp_path / "nope.mp4", tmp_path / "out.mp4")
