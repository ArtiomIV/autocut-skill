"""Unit tests for the resume-state sidecar and ``complete_from_plan`` reuse path.

The end-to-end host-agent resume flow is tested at integration level (it needs
ffmpeg for the dispatch step). This file focuses on the deterministic glue:

- ``_write_resume_state`` writes a parseable JSON sidecar with all required keys.
- ``load_resume_state`` round-trips the sidecar and complains about every
  failure mode (missing file, malformed JSON, wrong top-level shape, missing
  required keys).
- ``complete_from_plan`` produces an ``AnalysisResult`` with ranked clips and
  no dispatch when ``write_outputs=False`` — covering the dry-run / unit path.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from autocut.config import AutoCutConfig
from autocut.content import HIGHLIGHTS_PROFILE
from autocut.models import (
    Category,
    Clip,
    ClipPlan,
    ClipPlanMetadata,
    ContentHint,
    VideoMetadata,
)
from autocut.pipeline import (
    RESUME_STATE_FILENAME,
    ResumeStateError,
    _write_resume_state,
    complete_from_plan,
    load_resume_state,
)


def _metadata() -> VideoMetadata:
    return VideoMetadata(
        path=Path("/fake/video.mp4"),
        duration_sec=42.0,
        width=1920,
        height=1080,
        fps=60.0,
        video_codec="hevc",
        audio_codec="aac",
        container="mov",
        size_bytes=143_237_289,
    )


def _plan() -> ClipPlan:
    return ClipPlan(
        video_id="test",
        duration_sec=42.0,
        clips=[
            Clip(
                id="c1",
                start=timedelta(seconds=15),
                end=timedelta(seconds=23),
                category=Category.highlight,
                description="punch sequence",
                score=7,
                rationale="reads as a clean exchange",
                tags=["boxing"],
            ),
            Clip(
                id="c2",
                start=timedelta(seconds=32),
                end=timedelta(seconds=41),
                category=Category.highlight,
                description="final exchange",
                score=9,
                rationale="most dynamic moment",
                tags=["boxing", "finale"],
            ),
        ],
        metadata=ClipPlanMetadata(vlm_provider="host", vlm_model="claude-opus-4-7"),
    )


# ---------------------------------------------------------------------------
# write + load round-trip
# ---------------------------------------------------------------------------


def test_write_resume_state_serialises_every_field(tmp_path: Path) -> None:
    path = _write_resume_state(
        tmp_path,
        video=Path("/fake/video.mp4"),
        metadata=_metadata(),
        n_scenes=1,
        n_keyframes=11,
        sampling_strategy="hybrid",
        accurate_cuts=True,
        write_outputs=True,
        content_hint=ContentHint.highlights,
    )
    assert path == tmp_path / RESUME_STATE_FILENAME
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["video_path"].endswith("video.mp4")
    assert raw["video_metadata"]["duration_sec"] == 42.0
    assert raw["video_metadata"]["width"] == 1920
    assert raw["sampling_strategy"] == "hybrid"
    assert raw["accurate_cuts"] is True
    assert raw["write_outputs"] is True
    assert raw["n_scenes"] == 1
    assert raw["n_keyframes"] == 11
    assert raw["content_hint"] == "highlights"


def test_load_resume_state_round_trips(tmp_path: Path) -> None:
    _write_resume_state(
        tmp_path,
        video=Path("/fake/video.mp4"),
        metadata=_metadata(),
        n_scenes=2,
        n_keyframes=8,
        sampling_strategy="uniform",
        accurate_cuts=False,
        write_outputs=False,
        content_hint=ContentHint.talk,
    )
    loaded = load_resume_state(tmp_path)
    assert loaded["sampling_strategy"] == "uniform"
    assert loaded["accurate_cuts"] is False
    assert loaded["n_keyframes"] == 8
    assert loaded["content_hint"] == "talk"


# ---------------------------------------------------------------------------
# load: failure modes
# ---------------------------------------------------------------------------


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ResumeStateError, match="not found"):
        load_resume_state(tmp_path)


def test_load_malformed_json_raises(tmp_path: Path) -> None:
    (tmp_path / RESUME_STATE_FILENAME).write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ResumeStateError, match="failed to read"):
        load_resume_state(tmp_path)


def test_load_top_level_array_raises(tmp_path: Path) -> None:
    (tmp_path / RESUME_STATE_FILENAME).write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ResumeStateError, match="not an object"):
        load_resume_state(tmp_path)


def test_load_missing_required_keys_raises(tmp_path: Path) -> None:
    (tmp_path / RESUME_STATE_FILENAME).write_text(
        json.dumps({"video_path": "/x"}), encoding="utf-8"
    )
    with pytest.raises(ResumeStateError, match="missing required keys"):
        load_resume_state(tmp_path)


# ---------------------------------------------------------------------------
# complete_from_plan: ranking + (no) dispatch
# ---------------------------------------------------------------------------


def test_complete_from_plan_ranks_without_writing_when_write_outputs_false(
    tmp_path: Path,
) -> None:
    result = complete_from_plan(
        _plan(),
        video=Path("/fake/video.mp4"),
        metadata=_metadata(),
        config=AutoCutConfig(),
        output_root=tmp_path,
        n_scenes=1,
        n_keyframes=11,
        sampling_strategy="hybrid",
        accurate_cuts=False,
        write_outputs=False,
    )
    # Both clips survive ranking (score=7 and score=9, default min_score=5).
    assert len(result.ranked) == 2
    # Highest final_score first.
    assert result.ranked[0].clip.id == "c2"
    assert result.plan_path is None  # write_outputs=False -> no plan
    assert result.plan.clips == _plan().clips


def test_complete_from_plan_drops_clips_below_min_score(tmp_path: Path) -> None:
    # Build a plan where one clip will score below the default min_score=5.
    plan = ClipPlan(
        video_id="vid",
        duration_sec=60.0,
        clips=[
            Clip(
                id="weak",
                start=timedelta(seconds=0),
                end=timedelta(seconds=1),  # 1s -> heur 2 (penalty), vlm 0 -> final 0
                category=Category.filler,
                description="weak",
                score=0,
                rationale="placeholder",
            ),
            Clip(
                id="strong",
                start=timedelta(seconds=10),
                end=timedelta(seconds=25),
                category=Category.highlight,
                description="strong",
                score=9,
                rationale="placeholder",
            ),
        ],
        metadata=ClipPlanMetadata(vlm_provider="host", vlm_model="m"),
    )
    result = complete_from_plan(
        plan,
        video=Path("/fake/video.mp4"),
        metadata=_metadata(),
        config=AutoCutConfig(),  # min_score=5 default
        output_root=tmp_path,
        n_scenes=1,
        n_keyframes=4,
        sampling_strategy="hybrid",
        accurate_cuts=False,
        write_outputs=False,
    )
    assert [r.clip.id for r in result.ranked] == ["strong"]


def _weak_mid_strong_plan() -> ClipPlan:
    """A plan spanning the highlights keep boundary: weak/mid/strong clips.

    All clips last >3 s so the duration heuristic is neutral (5) and the nudge
    is zero -> ``final == vlm``:
    - weak  (vlm=5) -> final 5
    - mid   (vlm=7) -> final 7  (a strong-impact moment)
    - strong(vlm=9) -> final 9
    """
    return ClipPlan(
        video_id="vid",
        duration_sec=60.0,
        clips=[
            Clip(
                id="weak",
                start=timedelta(seconds=0),
                end=timedelta(seconds=10),
                category=Category.highlight,
                description="weak",
                score=5,
                rationale="placeholder",
            ),
            Clip(
                id="mid",
                start=timedelta(seconds=20),
                end=timedelta(seconds=30),
                category=Category.highlight,
                description="mid",
                score=7,
                rationale="placeholder",
            ),
            Clip(
                id="strong",
                start=timedelta(seconds=40),
                end=timedelta(seconds=55),
                category=Category.highlight,
                description="strong",
                score=9,
                rationale="placeholder",
            ),
        ],
        metadata=ClipPlanMetadata(vlm_provider="host", vlm_model="m"),
    )


def test_highlights_profile_keeps_threshold_at_7(tmp_path: Path) -> None:
    # Default config (min_score=5) keeps all three clips (weak final=5 clears 5).
    kwargs = {
        "video": Path("/fake/video.mp4"),
        "metadata": _metadata(),
        "config": AutoCutConfig(),
        "output_root": tmp_path,
        "n_scenes": 1,
        "n_keyframes": 4,
        "sampling_strategy": "hybrid",
        "accurate_cuts": False,
        "write_outputs": False,
    }
    default_result = complete_from_plan(_weak_mid_strong_plan(), **kwargs)  # type: ignore[arg-type]
    assert [r.clip.id for r in default_result.ranked] == ["strong", "mid", "weak"]

    # The highlights profile (min_score=7) drops the weak clip but KEEPS the mid
    # one (final=7) — the strong-impact moments the model rates 7 must survive.
    highlights_result = complete_from_plan(
        _weak_mid_strong_plan(),
        profile=HIGHLIGHTS_PROFILE,
        **kwargs,  # type: ignore[arg-type]
    )
    assert [r.clip.id for r in highlights_result.ranked] == ["strong", "mid"]


def test_highlights_profile_allows_empty_output(tmp_path: Path) -> None:
    # A plan whose only clip is weak (vlm=5 -> final=5, below the highlights
    # threshold of 7) yields ZERO clips — an empty result is a valid, honest
    # outcome rather than shipping a best-of-nothing.
    plan = ClipPlan(
        video_id="vid",
        duration_sec=60.0,
        clips=[
            Clip(
                id="weak",
                start=timedelta(seconds=0),
                end=timedelta(seconds=10),
                category=Category.highlight,
                description="weak",
                score=5,
                rationale="placeholder",
            ),
        ],
        metadata=ClipPlanMetadata(vlm_provider="host", vlm_model="m"),
    )
    result = complete_from_plan(
        plan,
        video=Path("/fake/video.mp4"),
        metadata=_metadata(),
        config=AutoCutConfig(),
        output_root=tmp_path,
        n_scenes=1,
        n_keyframes=4,
        sampling_strategy="hybrid",
        accurate_cuts=False,
        write_outputs=True,
        profile=HIGHLIGHTS_PROFILE,
    )
    assert result.ranked == []
    assert result.plan_path is None  # nothing cleared the bar -> no plan written
