"""``merged`` output mode — single highlight reel produced via the concat demuxer.

Two-step process:

1. Cut every clip into a temporary directory (re-encoded so all chunks share
   the same codec parameters — required by the concat demuxer in stream-copy
   mode for the final stitch).
2. Delegate the actual concat to ``autocut.video.concat.concat_videos``,
   which runs ``ffmpeg -f concat -safe 0 -i concat.txt -c copy highlights.mp4``.

Re-encoding the chunks is the safe path; it tolerates source videos with
mixed codecs/resolutions/framerates that would otherwise refuse to concat.
For v0.1.0 the re-encode cost is acceptable (we cut at most ~20 chunks per
run by default).

The concat primitive lives in ``autocut.video.concat`` so the standalone
``autocut merge`` CLI can reuse it without dragging in the writer/ranker
machinery this module needs.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import ClassVar, Literal

from autocut.config import MergeOrder
from autocut.output.base import OutputWriter, WrittenClip
from autocut.scoring import RankedClip
from autocut.security.paths import ensure_inside
from autocut.video import CutRequest, cut_clip
from autocut.video.concat import ConcatError, concat_videos
from autocut.video.cutter import expand_request

log = logging.getLogger(__name__)

SUBDIR_NAME = "merged"
DEFAULT_OUTPUT_NAME = "highlights.mp4"


# Kept for backwards compatibility with callers that catch this specific
# error. The shared concat primitive raises ``ConcatError``; we re-export
# it under the legacy name so existing ``except MergedConcatError`` blocks
# keep working without code changes.
MergedConcatError = ConcatError


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
        pre_roll_sec: float = 0.0,
        post_roll_sec: float = 0.0,
        video_duration_sec: float = 0.0,
    ) -> list[WrittenClip]:
        if not clips:
            return []

        ordered_clips = _apply_order(clips, self._order)

        target_dir = (output_dir / SUBDIR_NAME).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        final_path = ensure_inside(target_dir / DEFAULT_OUTPUT_NAME, target_dir)

        with tempfile.TemporaryDirectory(prefix="autocut_merge_") as tmp:
            chunk_paths = _cut_chunks(
                video_path,
                ordered_clips,
                Path(tmp),
                accurate=accurate,
                pre_roll_sec=pre_roll_sec,
                post_roll_sec=post_roll_sec,
                video_duration_sec=video_duration_sec,
            )
            concat_videos(chunk_paths, final_path)

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
    pre_roll_sec: float = 0.0,
    post_roll_sec: float = 0.0,
    video_duration_sec: float = 0.0,
) -> list[Path]:
    # ``accurate`` is unused in the merge path: we ALWAYS re-encode the chunks
    # so they share identical stream parameters — concat-copy requires that.
    del accurate
    chunk_paths: list[Path] = []
    for i, ranked in enumerate(clips):
        chunk = tmp_dir / f"chunk_{i:03d}.mp4"
        request = CutRequest(
            start=ranked.clip.start,
            end=ranked.clip.end,
            output_path=chunk,
        )
        request = expand_request(
            request,
            pre_roll_sec=pre_roll_sec,
            post_roll_sec=post_roll_sec,
            video_duration_sec=video_duration_sec,
        )
        cut_clip(video_path, request, accurate=True)
        chunk_paths.append(chunk)
    return chunk_paths


def _write_order_log(target_dir: Path, clips: list[RankedClip]) -> None:
    log_path = target_dir / "highlights.txt"
    lines = [
        f"{i:>3d}. [score {r.final_score:>2d}] {r.clip.start} → {r.clip.end}  {r.clip.description}"
        for i, r in enumerate(clips, start=1)
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# Re-export the literal so the dispatcher can reference the same source of truth.
_MergeOrderLiteral = Literal["score", "chronological", "alternating"]
