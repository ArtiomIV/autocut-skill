"""Resolve the ``ffmpeg`` / ``ffprobe`` executables in one place.

Resolution order for each binary:

1. An explicit path passed by the caller (the ``*_path`` keyword arguments
   the video helpers already expose).
2. The system ``PATH`` (``shutil.which``).
3. The ``static-ffmpeg`` bundled binaries, fetched on first use and cached on
   disk by the library (network required the first time only).

Centralising this means every helper resolves binaries identically and
``autocut doctor`` can report which source is active. ``ffprobe`` matters as
much as ``ffmpeg``: we probe metadata with it, and a bundle that ships only
``ffmpeg`` (e.g. imageio-ffmpeg) would not cover the pipeline.
"""

from __future__ import annotations

import shutil
from functools import lru_cache
from typing import Literal

BinarySource = Literal["explicit", "system", "bundled"]


class FFmpegResolveError(RuntimeError):
    """Raised when neither the system nor the bundled binary can be located."""


@lru_cache(maxsize=1)
def _bundled_binaries() -> tuple[str, str]:
    """Return ``(ffmpeg, ffprobe)`` from static-ffmpeg, fetching on first call.

    The first invocation downloads the platform binaries; static-ffmpeg caches
    them on disk and ``lru_cache`` memoises the resolved paths in-process.
    """
    try:
        from static_ffmpeg.run import get_or_fetch_platform_executables_else_raise
    except ImportError as exc:  # pragma: no cover - static-ffmpeg is a hard dependency
        raise FFmpegResolveError(
            "static-ffmpeg is not installed; cannot provide a bundled ffmpeg/ffprobe"
        ) from exc
    try:
        ffmpeg, ffprobe = get_or_fetch_platform_executables_else_raise()
    except Exception as exc:  # any download/extract failure is terminal here
        raise FFmpegResolveError(
            f"could not fetch the bundled ffmpeg/ffprobe binaries: {exc}"
        ) from exc
    return str(ffmpeg), str(ffprobe)


def _resolve(name: str, explicit: str | None, bundled_index: int) -> tuple[str, BinarySource]:
    if explicit:
        return explicit, "explicit"
    found = shutil.which(name)
    if found:
        return found, "system"
    return _bundled_binaries()[bundled_index], "bundled"


def ffmpeg_binary(explicit: str | None = None) -> str:
    """Return a usable ffmpeg path (explicit > system PATH > bundled fallback)."""
    return _resolve("ffmpeg", explicit, 0)[0]


def ffprobe_binary(explicit: str | None = None) -> str:
    """Return a usable ffprobe path (explicit > system PATH > bundled fallback)."""
    return _resolve("ffprobe", explicit, 1)[0]


def describe_ffmpeg() -> tuple[str, BinarySource]:
    """``(path, source)`` for ffmpeg — used by ``autocut doctor``.

    May trigger the bundled fetch when no system ffmpeg exists; raises
    ``FFmpegResolveError`` if even that fails.
    """
    return _resolve("ffmpeg", None, 0)


def describe_ffprobe() -> tuple[str, BinarySource]:
    """``(path, source)`` for ffprobe — used by ``autocut doctor``."""
    return _resolve("ffprobe", None, 1)
