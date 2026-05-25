"""``merged`` output mode — single highlight reel produced via the concat demuxer.

Two-step process:

1. Cut every clip into a temporary directory (re-encoded so all chunks share
   the same codec parameters — required by the concat demuxer in stream-copy
   mode for the final stitch).
2. Build a ``concat.txt`` file list and run ``ffmpeg -f concat -safe 0 -i
   concat.txt -c copy highlights.mp4``.

Re-encoding the chunks is the safe path; it tolerates source videos with
mixed codecs/resolutions/framerates that would otherwise refuse to concat.
For v0.1.0 the re-encode cost is acceptable (we cut at most ~20 chunks per
run by default).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import ClassVar, Literal

from autocut.config import MergeOrder
from autocut.output.base import OutputWriter, WrittenClip
from autocut.scoring import RankedClip
from autocut.security.paths import ensure_inside
from autocut.video import CutRequest, cut_clip

log = logging.getLogger(__name__)

SUBDIR_NAME = "merged"
DEFAULT_OUTPUT_NAME = "highlights.mp4"
_CONCAT_TIMEOUT_SEC = 600


class MergedConcatError(RuntimeError):
    """Raised when the final concat ffmpeg invocation fails."""


class MergedWriter(OutputWriter):
    """Write a single ``highlights.mp4`` under ``<output_dir>/merged/``."""

    name: ClassVar[str] = "merged"

    def __init__(self, *, order: MergeOrder = "score") -> None:
        self._order = order

    def write(
        self,
        video_path: Path,
        clips: list[RankedClip],
        output_dir: Path,
        *,
        accurate: bool = False,
    ) -> list[WrittenClip]:
        if not clips:
            return []

        ordered_clips = _apply_order(clips, self._order)

        target_dir = (output_dir / SUBDIR_NAME).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        final_path = ensure_inside(target_dir / DEFAULT_OUTPUT_NAME, target_dir)

        ffmpeg_binary = shutil.which("ffmpeg")
        if ffmpeg_binary is None:
            raise MergedConcatError(
                "ffmpeg not found in PATH; install ffmpeg before producing merged output"
            )

        with tempfile.TemporaryDirectory(prefix="autocut_merge_") as tmp:
            tmp_dir = Path(tmp)
            chunk_paths = _cut_chunks(video_path, ordered_clips, tmp_dir, accurate=accurate)
            concat_list = _write_concat_list(tmp_dir, chunk_paths)
            _run_concat(ffmpeg_binary, concat_list, final_path)

        _write_order_log(target_dir, ordered_clips)
        log.info("merged: wrote %s (%d chunks)", final_path.name, len(ordered_clips))
        return [WrittenClip(path=final_path, source=ranked) for ranked in ordered_clips]


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _apply_order(clips: list[RankedClip], order: MergeOrder) -> list[RankedClip]:
    if order == "chronological":
        return sorted(clips, key=lambda r: r.clip.start)
    if order == "alternating":
        # Interleave by descending score: best clip first, then second-best, etc.
        return sorted(clips, key=lambda r: (-r.final_score, r.clip.start))
    # default "score" matches the ranker output order (already by score desc).
    return list(clips)


def _cut_chunks(
    video_path: Path,
    clips: list[RankedClip],
    tmp_dir: Path,
    *,
    accurate: bool,
) -> list[Path]:
    # ``accurate`` is unused in the merge path: we ALWAYS re-encode the chunks
    # so they share identical stream parameters — concat-copy requires that.
    del accurate
    chunk_paths: list[Path] = []
    for i, ranked in enumerate(clips):
        chunk = tmp_dir / f"chunk_{i:03d}.mp4"
        cut_clip(
            video_path,
            CutRequest(
                start=ranked.clip.start,
                end=ranked.clip.end,
                output_path=chunk,
            ),
            accurate=True,
        )
        chunk_paths.append(chunk)
    return chunk_paths


def _write_concat_list(tmp_dir: Path, chunk_paths: list[Path]) -> Path:
    list_file = tmp_dir / "concat.txt"
    lines = [f"file '{_escape_for_concat(p)}'" for p in chunk_paths]
    list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return list_file


def _escape_for_concat(path: Path) -> str:
    """Escape single quotes for the ffmpeg concat demuxer file syntax."""
    return str(path).replace("'", r"'\''")


def _run_concat(ffmpeg_binary: str, concat_list: Path, out_path: Path) -> None:
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
            text=True,
            timeout=_CONCAT_TIMEOUT_SEC,
            check=False,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        raise MergedConcatError(f"failed to invoke ffmpeg concat: {exc}") from exc
    if completed.returncode != 0 or not out_path.is_file():
        raise MergedConcatError(
            f"ffmpeg concat failed: {completed.stderr.strip()}"
        )


def _write_order_log(target_dir: Path, clips: list[RankedClip]) -> None:
    log_path = target_dir / "highlights.txt"
    lines = [
        f"{i:>3d}. [score {r.final_score:>2d}] {r.clip.start} → {r.clip.end}  {r.clip.description}"
        for i, r in enumerate(clips, start=1)
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# Re-export the literal so the dispatcher can reference the same source of truth.
_MergeOrderLiteral = Literal["score", "chronological", "alternating"]
