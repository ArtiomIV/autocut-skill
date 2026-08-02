"""Concat a list of MP4 files into a single output via ffmpeg concat demuxer.

This module exposes the low-level ``concat_videos`` primitive used by two
callers:

- ``autocut.output.merged.MergedWriter`` — internal pipeline writer that
  cuts chunks first, then concats them.
- ``autocut merge`` — standalone CLI subcommand that takes already-cut
  MP4 files (e.g. produced by a previous ``autocut run --output separate``
  or by ``autocut cut``) and merges them.

Both reuse the same ffmpeg invocation, so the concat behaviour stays
consistent and is fixed in a single place when ffmpeg quirks surface.

Stream-copy concat is the default: cheap, lossless, but requires every
input file to share container/codec/resolution/framerate. For files
that diverge (e.g. mixing clips from different sources) re-encode them
upstream before calling here — this primitive does not try to be smart
about mismatched inputs.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

from autocut.video.ffmpeg_path import FFmpegResolveError, ffmpeg_binary

log = logging.getLogger(__name__)


class ConcatError(RuntimeError):
    """Raised when ffmpeg fails to produce the concatenated output."""


_DEFAULT_CONCAT_TIMEOUT_SEC = 600


def concat_videos(
    inputs: list[Path],
    output: Path,
    *,
    ffmpeg_path: str | None = None,
    timeout_sec: int = _DEFAULT_CONCAT_TIMEOUT_SEC,
) -> Path:
    """Concatenate ``inputs`` into ``output`` with the ffmpeg concat demuxer.

    Returns ``output`` on success. Raises ``ConcatError`` if any input is
    missing, if ffmpeg is not on PATH (and ``ffmpeg_path`` is not given),
    or if ffmpeg exits non-zero.

    The function writes a temporary ``concat.txt`` list-file in a private
    temp directory, never inside the output directory — that way concurrent
    callers writing to the same output dir don't collide on the list-file.
    """
    if not inputs:
        raise ConcatError("concat_videos called with no input files")

    for inp in inputs:
        if not inp.is_file():
            raise ConcatError(f"input file not found: {inp}")

    try:
        binary = ffmpeg_binary(ffmpeg_path)
    except FFmpegResolveError as exc:
        raise ConcatError(f"ffmpeg not found: {exc}") from exc

    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="autocut_concat_") as tmp:
        list_path = _write_concat_list(Path(tmp), inputs)
        _run_concat(binary, list_path, output, timeout_sec=timeout_sec)

    log.info("concat: wrote %s from %d input(s)", output.name, len(inputs))
    return output


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _write_concat_list(tmp_dir: Path, inputs: list[Path]) -> Path:
    """Build the ffmpeg concat demuxer list-file under ``tmp_dir``.

    Each input is resolved to an absolute path so the list-file works even
    if ffmpeg is invoked with a different cwd than the caller. Single
    quotes in paths are escaped per the demuxer syntax.
    """
    list_file = tmp_dir / "concat.txt"
    lines = [f"file '{_escape_for_concat(p.resolve())}'" for p in inputs]
    list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return list_file


def _escape_for_concat(path: Path) -> str:
    """Escape single quotes per the ffmpeg concat demuxer's quoting rules."""
    # The concat demuxer reads the line as ``file 'path'``. Single quotes
    # inside the path must be closed, escaped, and reopened: '\''.
    return str(path).replace("'", r"'\''")


def _run_concat(
    ffmpeg_binary: str,
    concat_list: Path,
    out_path: Path,
    *,
    timeout_sec: int,
) -> None:
    """Invoke ``ffmpeg -f concat -safe 0 -i <list> -c copy <out>``."""
    args: list[str] = [
        ffmpeg_binary,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list),
        "-c",
        "copy",
        str(out_path),
    ]
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
        raise ConcatError(f"failed to invoke ffmpeg concat: {exc}") from exc
    if completed.returncode != 0 or not out_path.is_file():
        raise ConcatError(f"ffmpeg concat failed: {completed.stderr.strip()}")
