"""Integration tests for the output writers (real ffmpeg required).

The unit suite covers slugify, manifest shape, and ranker logic with stubs;
this file proves that ``SeparateWriter`` and ``MergedWriter`` actually
produce playable MP4s when wired to the real ffmpeg binary.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from autocut.config import MergeOrder
from autocut.models import Category, Clip, VideoMetadata
from autocut.output.dispatcher import dispatch_outputs
from autocut.output.merged import DEFAULT_OUTPUT_NAME as MERGED_FILENAME
from autocut.output.merged import MergedWriter
from autocut.output.separate import SeparateWriter
from autocut.scoring import RankedClip
from autocut.video import probe_video

pytestmark = pytest.mark.integration


def _ranked(id_: str, start: float, end: float, *, score: int = 8) -> RankedClip:
    clip = Clip(
        id=id_,
        start=timedelta(seconds=start),
        end=timedelta(seconds=end),
        category=Category.highlight,
        description=f"clip {id_}",
        score=score,
        rationale="because",
    )
    return RankedClip(clip=clip, vlm_score=score, heuristic_score=7, final_score=score)


def _metadata(path: Path, duration: float) -> VideoMetadata:
    return VideoMetadata(
        path=path,
        duration_sec=duration,
        width=640,
        height=480,
        fps=30.0,
        video_codec="h264",
        audio_codec=None,
        container="mp4",
        size_bytes=path.stat().st_size,
    )


# ---------------------------------------------------------------------------
# SeparateWriter
# ---------------------------------------------------------------------------


def test_separate_writer_produces_one_mp4_per_clip(medium_video: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "CLIPS"
    clips = [
        _ranked("a", 1.0, 3.0, score=9),
        _ranked("b", 4.0, 7.0, score=7),
    ]
    written = SeparateWriter().write(medium_video, clips, out_dir, accurate=True)

    assert len(written) == 2
    for i, w in enumerate(written, start=1):
        assert w.path.is_file()
        assert w.path.parent.name == "separate"
        # Filename must encode rank + final_score + slug.
        assert w.path.name.startswith(f"clip_{i:03d}_s")
        assert w.path.suffix == ".mp4"

    # Durations: probe each output and confirm we got what we asked for.
    d1 = probe_video(written[0].path).duration_sec
    d2 = probe_video(written[1].path).duration_sec
    assert d1 == pytest.approx(2.0, abs=0.2)
    assert d2 == pytest.approx(3.0, abs=0.2)


def test_separate_writer_returns_empty_on_no_clips(medium_video: Path, tmp_path: Path) -> None:
    written = SeparateWriter().write(medium_video, [], tmp_path / "CLIPS", accurate=True)
    assert written == []


# ---------------------------------------------------------------------------
# MergedWriter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", ["score", "chronological", "alternating"])
def test_merged_writer_produces_single_highlights_mp4(
    medium_video: Path, tmp_path: Path, order: MergeOrder
) -> None:
    out_dir = tmp_path / "CLIPS"
    clips = [
        _ranked("a", 1.0, 3.0, score=7),
        _ranked("b", 4.0, 7.0, score=9),
        _ranked("c", 7.5, 9.5, score=8),
    ]
    written = MergedWriter(order=order).write(medium_video, clips, out_dir, accurate=False)

    # All entries point at the same merged file.
    assert {w.path for w in written} == {out_dir / "merged" / MERGED_FILENAME}
    final = written[0].path
    assert final.is_file()
    # Sum of chunk durations: 2 + 3 + 2 = 7s. Allow ±0.5s for muxer overhead.
    assert probe_video(final).duration_sec == pytest.approx(7.0, abs=0.5)
    # Order log accompanies the output.
    assert (out_dir / "merged" / "highlights.txt").is_file()


def test_merged_writer_skips_when_no_clips(medium_video: Path, tmp_path: Path) -> None:
    assert MergedWriter().write(medium_video, [], tmp_path / "CLIPS") == []


# ---------------------------------------------------------------------------
# dispatch_outputs end-to-end with both modes
# ---------------------------------------------------------------------------


def test_dispatch_with_separate_and_merged_writes_both(medium_video: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "CLIPS"
    clips = [_ranked("a", 1.0, 3.0, score=9), _ranked("b", 5.0, 7.0, score=7)]
    metadata = _metadata(medium_video, duration=10.0)

    result = dispatch_outputs(
        medium_video,
        clips,
        metadata,
        out_dir,
        modes=["separate", "merged"],
        accurate=True,
        extra_manifest={"sampling": {"strategy": "uniform"}},
    )

    assert set(result.by_mode.keys()) == {"separate", "merged"}
    assert len(result.by_mode["separate"]) == 2
    assert len(result.by_mode["merged"]) == 2  # one entry per source clip

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["outputs"]["separate"]
    assert manifest["outputs"]["merged"]
    assert manifest["sampling"]["strategy"] == "uniform"
    assert len(manifest["clips"]) == 2
