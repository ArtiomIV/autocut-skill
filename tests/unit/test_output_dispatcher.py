"""Unit tests for ``autocut.output.dispatcher`` — mode selection + manifest shape."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from autocut.models import Category, Clip, VideoMetadata
from autocut.output.dispatcher import MANIFEST_FILENAME, dispatch_outputs
from autocut.scoring import RankedClip


def _ranked(id_: str, start: float, end: float, *, final: int = 8) -> RankedClip:
    clip = Clip(
        id=id_,
        start=timedelta(seconds=start),
        end=timedelta(seconds=end),
        category=Category.highlight,
        description=f"clip {id_}",
        score=final,
        rationale="because",
        tags=["t1"],
    )
    return RankedClip(clip=clip, vlm_score=final, heuristic_score=7, final_score=final)


def _metadata(path: Path) -> VideoMetadata:
    return VideoMetadata(
        path=path,
        duration_sec=60.0,
        width=640,
        height=480,
        fps=30.0,
        video_codec="h264",
        audio_codec="aac",
        container="mp4",
        size_bytes=1024,
    )


# ---------------------------------------------------------------------------
# dispatch_outputs — mode handling
# ---------------------------------------------------------------------------


def test_unknown_mode_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown output mode"):
        dispatch_outputs(
            tmp_path / "v.mp4",
            [_ranked("c1", 0, 15)],
            _metadata(tmp_path / "v.mp4"),
            tmp_path / "CLIPS",
            modes=["bogus"],  # type: ignore[list-item]
        )


def test_dispatch_creates_manifest_with_expected_shape(tmp_path: Path) -> None:
    # With modes=[] no writer runs, so the manifest is the only artefact
    # produced — handy for asserting its shape without invoking ffmpeg.
    video = tmp_path / "v.mp4"
    out_dir = tmp_path / "CLIPS"

    result = dispatch_outputs(
        video,
        [_ranked("a", 0, 15, final=9), _ranked("b", 20, 35, final=7)],
        _metadata(video),
        out_dir,
        modes=[],
    )

    assert result.manifest_path is not None
    assert result.manifest_path.name == MANIFEST_FILENAME
    data = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert data["video"]["duration_sec"] == 60.0
    assert data["video"]["width"] == 640
    assert len(data["clips"]) == 2
    assert data["clips"][0]["rank"] == 1
    assert data["clips"][0]["id"] == "a"
    assert data["clips"][0]["final_score"] == 9
    assert data["clips"][0]["start"] == "00:00:00.000"
    assert data["clips"][0]["end"] == "00:00:15.000"
    assert data["outputs"] == {}


def test_dispatch_passes_extra_manifest_keys(tmp_path: Path) -> None:
    video = tmp_path / "v.mp4"
    result = dispatch_outputs(
        video,
        [_ranked("a", 0, 15)],
        _metadata(video),
        tmp_path / "CLIPS",
        modes=[],
        extra_manifest={"sampling": {"strategy": "hybrid", "n_keyframes": 12}},
    )
    assert result.manifest_path is not None
    data = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert data["sampling"]["strategy"] == "hybrid"
    assert data["sampling"]["n_keyframes"] == 12


def test_dispatch_handles_no_clips_by_writing_empty_manifest(tmp_path: Path) -> None:
    video = tmp_path / "v.mp4"
    result = dispatch_outputs(
        video,
        [],
        _metadata(video),
        tmp_path / "CLIPS",
        modes=[],
    )
    assert result.by_mode == {}
    assert result.manifest_path is not None
    data = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert data["clips"] == []
