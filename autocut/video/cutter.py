"""Cut clips out of the source video with ``ffmpeg``.

Two modes:

- ``stream-copy`` (default): ``-c copy`` -- fast and lossless but cuts snap
  to the nearest keyframe (typically every 1-2s). Good enough for highlight
  reels.
- ``accurate``: re-encode with libx264 + aac so every cut starts exactly at
  the requested timestamp. Slower (~real-time per clip on a modern laptop),
  used when the caller passes ``accurate=True``.

We never write outside the caller-provided ``output_dir``. Path validation
is the caller's responsibility (see ``autocut.security.paths.ensure_inside``).
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path


class CutterError(RuntimeError):
    """Raised when ffmpeg fails to produce the requested clip."""


_DEFAULT_FFMPEG_TIMEOUT_SEC = 600  # generous cap for re-encode of long clips
_REENCODE_VIDEO_CODEC = "libx264"
_REENCODE_AUDIO_CODEC = "aac"
_REENCODE_PRESET = "veryfast"
_REENCODE_CRF = "20"


@dataclass(frozen=True, slots=True)
class CutRequest:
    """A single clip the caller wants cut from the source video."""

    start: timedelta
    end: timedelta
    output_path: Path

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError("end must be strictly greater than start")


def cut_clip(
    video_path: str | Path,
    request: CutRequest,
    *,
    accurate: bool = False,
    ffmpeg_path: str | None = None,
) -> Path:
    """Cut a single clip. Returns the path to the produced file.

    Stream-copy uses ``-ss`` BEFORE ``-i`` for fast seek; accurate mode puts
    ``-ss`` AFTER ``-i`` so ffmpeg decodes from the start and lands on the
    exact requested frame.
    """
    video = Path(video_path)
    if not video.is_file():
        raise CutterError(f"input file does not exist: {video}")

    binary = ffmpeg_path or shutil.which("ffmpeg")
    if binary is None:
        raise CutterError("ffmpeg not found in PATH; install ffmpeg or pass ffmpeg_path explicitly")

    request.output_path.parent.mkdir(parents=True, exist_ok=True)
    args = _build_args(binary, video, request, accurate=accurate)

    try:
        completed = subprocess.run(  # noqa: S603 - args is a fixed list, no shell
            args,
            capture_output=True,
            text=True,
            timeout=_DEFAULT_FFMPEG_TIMEOUT_SEC,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise CutterError(f"ffmpeg timed out cutting clip {request.output_path.name}") from exc
    except OSError as exc:
        raise CutterError(f"failed to invoke ffmpeg: {exc}") from exc

    if completed.returncode != 0 or not request.output_path.is_file():
        raise CutterError(
            f"ffmpeg failed cutting clip {request.output_path.name}: {completed.stderr.strip()}"
        )
    return request.output_path


def cut_clips(
    video_path: str | Path,
    requests: list[CutRequest],
    *,
    accurate: bool = False,
    ffmpeg_path: str | None = None,
) -> list[Path]:
    """Cut every request sequentially. Returns the list of produced paths."""
    return [
        cut_clip(video_path, req, accurate=accurate, ffmpeg_path=ffmpeg_path) for req in requests
    ]


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _build_args(
    ffmpeg_binary: str,
    video: Path,
    request: CutRequest,
    *,
    accurate: bool,
) -> list[str]:
    start_ts = _format_ts(request.start)
    end_ts = _format_ts(request.end)

    if accurate:
        return [
            ffmpeg_binary,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-ss",
            start_ts,
            "-to",
            end_ts,
            "-c:v",
            _REENCODE_VIDEO_CODEC,
            "-preset",
            _REENCODE_PRESET,
            "-crf",
            _REENCODE_CRF,
            "-c:a",
            _REENCODE_AUDIO_CODEC,
            "-avoid_negative_ts",
            "make_zero",
            str(request.output_path),
        ]

    # Fast path: stream copy. ``-ss`` BEFORE ``-i`` seeks the container,
    # which is fast but lands on the previous keyframe.
    return [
        ffmpeg_binary,
        "-y",
        "-loglevel",
        "error",
        "-ss",
        start_ts,
        "-to",
        end_ts,
        "-i",
        str(video),
        "-c",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        str(request.output_path),
    ]


def _format_ts(ts: timedelta) -> str:
    total_ms = max(0, int(ts.total_seconds() * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, milliseconds = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
