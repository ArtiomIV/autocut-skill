"""Tests for the run->plan.json / cut --from-json round trip (no ffmpeg)."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from autocut.models import Clip, VideoMetadata
from autocut.output.dispatcher import PlanReadError, read_plan_json, write_plan_json
from autocut.scoring import RankedClip


def _metadata(duration: float = 100.0) -> VideoMetadata:
    return VideoMetadata(
        path=Path("/src/video.mp4"),
        duration_sec=duration,
        width=1920,
        height=1080,
        fps=30.0,
        video_codec="h264",
        audio_codec="aac",
        container="mp4",
        size_bytes=123456,
    )


def _ranked(start: float, end: float, *, vlm: int = 9, final: int = 9) -> RankedClip:
    clip = Clip(
        id="c1",
        start=timedelta(seconds=start),
        end=timedelta(seconds=end),
        category="highlight",
        description="a knockdown",
        score=vlm,
        rationale="decisive",
        tags=["boxing"],
    )
    return RankedClip(clip=clip, vlm_score=vlm, heuristic_score=5, final_score=final)


def test_write_plan_json_bakes_roll_and_omits_outputs(tmp_path: Path) -> None:
    meta = _metadata(duration=100.0)
    path = write_plan_json(
        output_dir=tmp_path,
        video_path=meta.path,
        metadata=meta,
        ranked=[_ranked(50.0, 60.0)],
        pre_roll_sec=3.0,
        post_roll_sec=3.0,
        extra={"vlm": {"provider": "openrouter"}},
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    # No cut outputs in a plan (cutting happens later via `cut --from-json`).
    assert "outputs" not in data
    # Roll baked in: 50-3 -> 47, 60+3 -> 63.
    assert data["clips"][0]["start"] == "00:00:47.000"
    assert data["clips"][0]["end"] == "00:01:03.000"
    assert data["vlm"]["provider"] == "openrouter"


def test_roll_clamps_to_source_bounds(tmp_path: Path) -> None:
    meta = _metadata(duration=12.0)
    path = write_plan_json(
        output_dir=tmp_path,
        video_path=meta.path,
        metadata=meta,
        ranked=[_ranked(1.0, 11.0)],
        pre_roll_sec=3.0,
        post_roll_sec=3.0,
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    # start clamps to 0 (1-3 < 0), end clamps to duration (11+3 > 12).
    assert data["clips"][0]["start"] == "00:00:00.000"
    assert data["clips"][0]["end"] == "00:00:12.000"


def test_round_trip_read_plan_json(tmp_path: Path) -> None:
    meta = _metadata()
    write_plan_json(
        output_dir=tmp_path,
        video_path=meta.path,
        metadata=meta,
        ranked=[_ranked(10.0, 20.0, vlm=8, final=8), _ranked(40.0, 50.0, vlm=9, final=9)],
        pre_roll_sec=0.0,
        post_roll_sec=0.0,
    )
    video, md, ranked = read_plan_json(tmp_path / "plan.json")
    assert video == Path("/src/video.mp4")
    assert md.duration_sec == 100.0
    assert [r.final_score for r in ranked] == [8, 9]
    assert ranked[0].clip.start == timedelta(seconds=10)
    assert ranked[1].clip.end == timedelta(seconds=50)


def test_read_plan_json_min_score_filter(tmp_path: Path) -> None:
    meta = _metadata()
    write_plan_json(
        output_dir=tmp_path,
        video_path=meta.path,
        metadata=meta,
        ranked=[_ranked(10.0, 20.0, final=6), _ranked(40.0, 50.0, final=9)],
    )
    _, _, ranked = read_plan_json(tmp_path / "plan.json", min_score=7)
    assert [r.final_score for r in ranked] == [9]  # the 6 is dropped


def test_read_plan_json_handles_zero_clips(tmp_path: Path) -> None:
    meta = _metadata()
    write_plan_json(output_dir=tmp_path, video_path=meta.path, metadata=meta, ranked=[])
    _, _, ranked = read_plan_json(tmp_path / "plan.json")
    assert ranked == []


def test_read_plan_json_missing_file(tmp_path: Path) -> None:
    with pytest.raises(PlanReadError, match="plan not found"):
        read_plan_json(tmp_path / "nope.json")
