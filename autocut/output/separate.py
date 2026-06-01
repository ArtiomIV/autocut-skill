"""``separate`` output mode — one MP4 per ranked clip.

Naming pattern: ``s<score>_clip_<NNN>_<slug>.mp4`` where
- ``<score>`` is the zero-padded final score 0-10 (LEADS the name so files
  group by score — sort the name column descending to put the best first)
- ``<NNN>`` is the zero-padded rank position
- ``<slug>`` is a sanitised excerpt of the clip description

We delegate the actual cutting to ``autocut.video.cutter`` so the
``accurate`` (re-encode) vs stream-copy decision is made in one place.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar

from autocut.output.base import OutputWriter, WrittenClip, slugify
from autocut.scoring import RankedClip
from autocut.security.paths import ensure_inside
from autocut.video import CutRequest, cut_clip
from autocut.video.cutter import expand_request

log = logging.getLogger(__name__)

SUBDIR_NAME = "separate"


class SeparateWriter(OutputWriter):
    """Write one MP4 per clip under ``<output_dir>/separate/``."""

    name: ClassVar[str] = "separate"

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
        target_dir = (output_dir / SUBDIR_NAME).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        written: list[WrittenClip] = []
        for rank, ranked in enumerate(clips, start=1):
            filename = _clip_filename(rank, ranked)
            # Defence in depth: ensure no slug ever lets us escape target_dir.
            out_path = ensure_inside(target_dir / filename, target_dir)
            request = CutRequest(
                start=ranked.clip.start,
                end=ranked.clip.end,
                output_path=out_path,
            )
            request = expand_request(
                request,
                pre_roll_sec=pre_roll_sec,
                post_roll_sec=post_roll_sec,
                video_duration_sec=video_duration_sec,
            )
            cut_clip(video_path, request, accurate=accurate)
            log.info("separate: wrote %s", out_path.name)
            written.append(WrittenClip(path=out_path, source=ranked))
        return written


def _clip_filename(rank: int, ranked: RankedClip) -> str:
    slug = slugify(ranked.clip.description)
    return f"s{ranked.final_score:02d}_clip_{rank:03d}_{slug}.mp4"
