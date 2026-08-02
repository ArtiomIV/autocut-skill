"""Thin wrapper around the ``ffprobe`` binary.

We invoke ``ffprobe`` via ``subprocess.run`` with a fixed argument list and
``shell=False``. The probe asks for JSON output covering both the container
(``format``) and every stream, then we map the result onto ``VideoMetadata``.

Why not a third-party wrapper (e.g. ``ffmpeg-python``):
- Subprocess gives us full control over arguments; wrappers tend to break
  whenever ffmpeg releases new flags.
- We only need a tiny subset of the ffprobe surface area.
"""

from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from autocut.models import VideoMetadata
from autocut.video.ffmpeg_path import FFmpegResolveError, ffprobe_binary


class FFprobeError(RuntimeError):
    """Raised when ``ffprobe`` is unavailable, fails, or returns unparseable data."""


_PROBE_TIMEOUT_SEC = 30
_DEFAULT_FFPROBE_ARGS = (
    "-v",
    "error",
    "-print_format",
    "json",
    "-show_format",
    "-show_streams",
)


def probe_video(path: str | Path, *, ffprobe_path: str | None = None) -> VideoMetadata:
    """Return ``VideoMetadata`` for ``path`` by invoking ``ffprobe``.

    Parameters
    ----------
    path:
        Path to the input video file. Must exist.
    ffprobe_path:
        Optional explicit path to the ffprobe binary. When omitted we resolve
        it via the system PATH, falling back to the bundled static-ffmpeg copy.

    Raises
    ------
    FFprobeError
        If ffprobe is missing, returns non-zero, produces invalid JSON, or
        lacks a video stream.
    """
    video_path = Path(path)
    if not video_path.is_file():
        raise FFprobeError(f"input file does not exist: {video_path}")

    try:
        binary = ffprobe_binary(ffprobe_path)
    except FFmpegResolveError as exc:
        raise FFprobeError(f"ffprobe not found: {exc}") from exc

    args = [binary, *_DEFAULT_FFPROBE_ARGS, str(video_path)]
    try:
        completed = subprocess.run(  # noqa: S603 - args is a fixed list, no shell
            args,
            capture_output=True,
            encoding="utf-8",
            errors="replace",  # ffmpeg speaks UTF-8; Windows ANSI codepages have holes
            timeout=_PROBE_TIMEOUT_SEC,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise FFprobeError(f"ffprobe timed out after {_PROBE_TIMEOUT_SEC}s") from exc
    except OSError as exc:
        raise FFprobeError(f"failed to invoke ffprobe: {exc}") from exc

    if completed.returncode != 0:
        raise FFprobeError(
            f"ffprobe exited with {completed.returncode}: {completed.stderr.strip()}"
        )

    try:
        payload: dict[str, Any] = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise FFprobeError(f"ffprobe returned invalid JSON: {exc}") from exc

    return _parse_payload(video_path, payload)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _parse_payload(video_path: Path, payload: dict[str, Any]) -> VideoMetadata:
    fmt = payload.get("format") or {}
    streams = payload.get("streams") or []

    video_stream = _first_stream(streams, "video")
    if video_stream is None:
        raise FFprobeError(f"no video stream found in {video_path}")
    audio_stream = _first_stream(streams, "audio")

    try:
        width = int(video_stream["width"])
        height = int(video_stream["height"])
        video_codec = str(video_stream["codec_name"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FFprobeError(f"video stream missing required field: {exc}") from exc

    fps = _parse_fps(video_stream)
    duration = _parse_duration(fmt, video_stream)
    container = _parse_container(fmt, video_path)
    size_bytes = _parse_size_bytes(fmt, video_path)
    audio_codec = str(audio_stream["codec_name"]) if audio_stream else None

    try:
        return VideoMetadata(
            path=video_path,
            duration_sec=duration,
            width=width,
            height=height,
            fps=fps,
            video_codec=video_codec,
            audio_codec=audio_codec,
            container=container,
            size_bytes=size_bytes,
        )
    except ValidationError as exc:
        raise FFprobeError(f"ffprobe payload failed validation: {exc}") from exc


def _first_stream(streams: list[dict[str, Any]], codec_type: str) -> dict[str, Any] | None:
    return next((s for s in streams if s.get("codec_type") == codec_type), None)


def _parse_fps(video_stream: dict[str, Any]) -> float:
    """Parse the frame rate. ffprobe reports a rational like ``"30000/1001"``."""
    for key in ("avg_frame_rate", "r_frame_rate"):
        raw = video_stream.get(key)
        if not raw or raw == "0/0":
            continue
        try:
            frac = Fraction(raw)
        except (ValueError, ZeroDivisionError):
            continue
        if frac > 0:
            return float(frac)
    raise FFprobeError("video stream is missing a usable frame rate")


def _parse_duration(fmt: dict[str, Any], video_stream: dict[str, Any]) -> float:
    for source in (fmt, video_stream):
        raw = source.get("duration")
        if raw is None:
            continue
        try:
            duration = float(raw)
        except (TypeError, ValueError):
            continue
        if duration > 0:
            return duration
    raise FFprobeError("could not determine video duration")


def _parse_container(fmt: dict[str, Any], video_path: Path) -> str:
    raw = fmt.get("format_name")
    if isinstance(raw, str) and raw:
        # ffprobe can report comma-separated families like "mov,mp4,m4a,3gp,3g2,mj2".
        return raw.split(",", 1)[0]
    return video_path.suffix.lstrip(".").lower() or "unknown"


def _parse_size_bytes(fmt: dict[str, Any], video_path: Path) -> int:
    raw = fmt.get("size")
    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            pass
    try:
        return video_path.stat().st_size
    except OSError:
        return 0
