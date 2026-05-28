"""Extract JPEG keyframes at a list of timestamps chosen upstream.

This module no longer decides WHICH timestamps to pick — that belongs to
``autocut.video.frame_sampler``. Here we just decode the requested frames
with ``ffmpeg``, resize to the configured long edge, and write them out.

Why ``ffmpeg`` and not Pillow + OpenCV: a single ffmpeg invocation does seek,
decode, and downscale in one pass. That keeps the pipeline fast and avoids
shipping OpenCV-only image conversions that would diverge from ffmpeg's
colour space handling.

The output is the input of the VLM provider: keep the file count small, keep
the resolution modest (768 px on the long side is enough context for content
classification and sits in the sweet spot for Anthropic / OpenAI / Gemini
image tokenisation).
"""

from __future__ import annotations

import subprocess
from datetime import timedelta
from pathlib import Path

from autocut.models import Keyframe
from autocut.video.ffmpeg_path import FFmpegResolveError, ffmpeg_binary
from autocut.video.frame_sampler import FrameSpec


class KeyframeExtractionError(RuntimeError):
    """Raised when ffmpeg cannot extract a frame at the requested timestamp."""


_DEFAULT_FFMPEG_TIMEOUT_SEC = 30
_DEFAULT_LONG_EDGE_PX = 768
# ffmpeg JPEG quality scale: 2 = best, 31 = worst. 5 ~= visually 85.
_DEFAULT_JPEG_QSCALE = 5


def extract_keyframes(
    video_path: str | Path,
    specs: list[FrameSpec],
    output_dir: str | Path,
    *,
    long_edge_px: int = _DEFAULT_LONG_EDGE_PX,
    ffmpeg_path: str | None = None,
) -> list[Keyframe]:
    """Decode each ``FrameSpec`` to a JPEG and return ``Keyframe`` records.

    Output files are named ``kf_<NNNN>_s<scene>.jpg``. The zero-padded index
    keeps alphabetical sort == chronological sort, which matters when the
    VLM ingests the list. ``output_dir`` is created if missing.
    """
    video = Path(video_path)
    if not video.is_file():
        raise KeyframeExtractionError(f"input file does not exist: {video}")

    try:
        binary = ffmpeg_binary(ffmpeg_path)
    except FFmpegResolveError as exc:
        raise KeyframeExtractionError(f"ffmpeg not found: {exc}") from exc

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    keyframes: list[Keyframe] = []
    for n, spec in enumerate(specs):
        scene_part = "u" if spec.scene_index < 0 else f"{spec.scene_index:04d}"
        out_path = out_dir / f"kf_{n:05d}_s{scene_part}.jpg"
        _extract_single(binary, video, spec.timestamp, out_path, long_edge_px)
        # Uniform-sampling specs use -1 to signal "no originating scene";
        # the model stores that as None.
        scene_idx = spec.scene_index if spec.scene_index >= 0 else None
        keyframes.append(Keyframe(scene_index=scene_idx, timestamp=spec.timestamp, path=out_path))
    return keyframes


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _extract_single(
    ffmpeg_binary: str,
    video: Path,
    timestamp: timedelta,
    out_path: Path,
    long_edge_px: int,
) -> None:
    # ``-ss <ts> -i <video>`` (i.e. ss BEFORE -i) makes ffmpeg seek to the
    # nearest keyframe before timestamp, which is fast. For exact-frame
    # accuracy we would put -ss AFTER -i (slower decode-from-start), but
    # for VLM input frame-accurate seeking is unnecessary.
    scale_filter = f"scale='if(gt(iw,ih),{long_edge_px},-2)':'if(gt(iw,ih),-2,{long_edge_px})'"
    args = [
        ffmpeg_binary,
        "-y",
        "-loglevel",
        "error",
        "-ss",
        _format_ts(timestamp),
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-vf",
        scale_filter,
        "-q:v",
        str(_DEFAULT_JPEG_QSCALE),
        "-update",
        "1",
        str(out_path),
    ]
    try:
        completed = subprocess.run(  # noqa: S603 - args is a fixed list, no shell
            args,
            capture_output=True,
            text=True,
            timeout=_DEFAULT_FFMPEG_TIMEOUT_SEC,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise KeyframeExtractionError(f"ffmpeg timed out extracting frame at {timestamp}") from exc
    except OSError as exc:
        raise KeyframeExtractionError(f"failed to invoke ffmpeg: {exc}") from exc

    if completed.returncode != 0 or not out_path.is_file():
        raise KeyframeExtractionError(
            f"ffmpeg failed extracting frame at {timestamp}: {completed.stderr.strip()}"
        )


def _format_ts(ts: timedelta) -> str:
    """Format a ``timedelta`` as ``HH:MM:SS.mmm`` for ffmpeg ``-ss``."""
    total_ms = max(0, int(ts.total_seconds() * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, milliseconds = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
