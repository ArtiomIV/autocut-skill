"""Compress a video to an analysis-grade copy before sending it to a cloud VLM.

Video-capable models (Gemini et al.) downsample the input internally to
roughly one low-resolution frame per second, so a high-quality source is
wasted bandwidth: it inflates the base64 payload past the inline request
limit without improving what the model actually perceives. This module
re-encodes the source to a small H.264 copy — long edge capped, frame rate
reduced, low-bitrate audio — preserving every signal the model uses while
shrinking the file by roughly an order of magnitude.

The same compression is what makes inline base64 viable: a 5-minute batch
of analysis-grade video fits comfortably under the request-size ceiling.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from autocut.video.ffmpeg_path import FFmpegResolveError, ffmpeg_binary

log = logging.getLogger(__name__)


class CompressionError(RuntimeError):
    """Raised when ffmpeg fails to produce the compressed output."""


_DEFAULT_LONG_EDGE_PX = 640
_DEFAULT_FPS = 15
_DEFAULT_CRF = 30
_DEFAULT_AUDIO_BITRATE = "64k"
# Re-encoding scales with input length; a long source on a slow CPU can take
# minutes. Generous ceiling so legitimate long encodes are not killed.
_DEFAULT_TIMEOUT_SEC = 1800


def compress_for_vlm(
    src: Path,
    output: Path,
    *,
    long_edge_px: int = _DEFAULT_LONG_EDGE_PX,
    fps: int = _DEFAULT_FPS,
    crf: int = _DEFAULT_CRF,
    audio_bitrate: str = _DEFAULT_AUDIO_BITRATE,
    ffmpeg_path: str | None = None,
    timeout_sec: int = _DEFAULT_TIMEOUT_SEC,
) -> Path:
    """Re-encode ``src`` to a small analysis-grade MP4 at ``output``.

    Caps the long edge at ``long_edge_px`` (aspect ratio preserved, never
    upscaling a smaller source, dimensions forced even for H.264), limits the
    frame rate to ``fps``, and encodes H.264 ``crf`` with low-bitrate AAC.
    Returns ``output``.

    Raises ``CompressionError`` if the source is missing, ffmpeg cannot be
    resolved, or ffmpeg exits non-zero.
    """
    if not src.is_file():
        raise CompressionError(f"source file not found: {src}")
    if long_edge_px <= 0 or fps <= 0 or crf < 0:
        raise CompressionError("long_edge_px and fps must be > 0 and crf >= 0")

    try:
        binary = ffmpeg_binary(ffmpeg_path)
    except FFmpegResolveError as exc:
        raise CompressionError(f"ffmpeg not found: {exc}") from exc

    output.parent.mkdir(parents=True, exist_ok=True)

    args = _build_args(binary, src, output, long_edge_px, fps, crf, audio_bitrate)
    try:
        completed = subprocess.run(  # noqa: S603 - args is a fixed list, no shell
            args,
            capture_output=True,
            encoding="utf-8",
            errors="replace",  # ffmpeg speaks UTF-8; Windows ANSI codepages have holes
            timeout=timeout_sec,
            check=False,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        raise CompressionError(f"failed to invoke ffmpeg: {exc}") from exc
    if completed.returncode != 0 or not output.is_file():
        raise CompressionError(f"ffmpeg compression failed: {completed.stderr.strip()}")

    log.info(
        "compress: %s -> %s (%d px long edge, %d fps, crf %d)",
        src.name,
        output.name,
        long_edge_px,
        fps,
        crf,
    )
    return output


def _build_args(
    binary: str,
    src: Path,
    output: Path,
    long_edge_px: int,
    fps: int,
    crf: int,
    audio_bitrate: str,
) -> list[str]:
    """Assemble the ffmpeg argument list (no shell, fixed list)."""
    # Fit the frame inside a long_edge x long_edge box without upscaling:
    # min(long_edge, iw/ih) caps the box at the source's own size, and
    # force_original_aspect_ratio=decrease scales down preserving aspect.
    # force_divisible_by=2 keeps both dimensions even, required by H.264.
    scale = (
        f"scale=w='min({long_edge_px},iw)':h='min({long_edge_px},ih)'"
        ":force_original_aspect_ratio=decrease:force_divisible_by=2"
    )
    return [
        binary,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vf",
        scale,
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        "veryfast",
        "-c:a",
        "aac",
        "-b:a",
        audio_bitrate,
        "-movflags",
        "+faststart",
        str(output),
    ]
