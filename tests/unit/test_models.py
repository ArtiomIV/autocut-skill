"""Tests for ``autocut.models`` — Pydantic schema, timestamp parsing, validators."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from autocut.models import (
    AnalysisHints,
    Category,
    Clip,
    ClipPlan,
    ClipPlanMetadata,
    ContentHint,
    Keyframe,
    Scene,
    VideoMetadata,
)

# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("00:00:00", timedelta()),
        ("00:00:05", timedelta(seconds=5)),
        ("00:01:23.450", timedelta(minutes=1, seconds=23, milliseconds=450)),
        ("01:30:00.001", timedelta(hours=1, minutes=30, milliseconds=1)),
        ("00:00:00.5", timedelta(milliseconds=500)),  # right-padded to 500 ms
        (12.5, timedelta(seconds=12, milliseconds=500)),
        (0, timedelta()),
    ],
)
def test_timestamp_accepts_valid_inputs(value: object, expected: timedelta) -> None:
    clip = Clip(
        id="x",
        start=value,  # type: ignore[arg-type]
        end=timedelta(hours=2),
        category=Category.highlight,
        description="d",
        score=5,
        rationale="r",
    )
    assert clip.start == expected


@pytest.mark.parametrize(
    "bad",
    [
        "1:2:3",  # missing zero-padding on minutes/seconds
        "00:60:00",  # minutes overflow
        "abc",
        "",
        "00:00:00.",
        -1,
        -0.001,
    ],
)
def test_timestamp_rejects_invalid_inputs(bad: object) -> None:
    with pytest.raises(ValidationError):
        Clip(
            id="x",
            start=bad,  # type: ignore[arg-type]
            end=timedelta(hours=2),
            category=Category.highlight,
            description="d",
            score=5,
            rationale="r",
        )


# ---------------------------------------------------------------------------
# Clip validators
# ---------------------------------------------------------------------------


def test_clip_rejects_end_not_after_start() -> None:
    with pytest.raises(ValidationError, match="end must be strictly greater than start"):
        Clip(
            id="x",
            start="00:00:10",  # type: ignore[arg-type]
            end="00:00:10",  # type: ignore[arg-type]
            category=Category.highlight,
            description="d",
            score=5,
            rationale="r",
        )


def test_clip_rejects_score_out_of_range() -> None:
    with pytest.raises(ValidationError):
        Clip(
            id="x",
            start="00:00:00",  # type: ignore[arg-type]
            end="00:00:10",  # type: ignore[arg-type]
            category=Category.highlight,
            description="d",
            score=11,
            rationale="r",
        )


def test_clip_rejects_unknown_category() -> None:
    with pytest.raises(ValidationError):
        Clip(
            id="x",
            start="00:00:00",  # type: ignore[arg-type]
            end="00:00:10",  # type: ignore[arg-type]
            category="explosion",  # type: ignore[arg-type]
            description="d",
            score=5,
            rationale="r",
        )


def test_clip_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Clip.model_validate(
            {
                "id": "x",
                "start": "00:00:00",
                "end": "00:00:10",
                "category": "highlight",
                "description": "d",
                "score": 5,
                "rationale": "r",
                "importance": "very",  # not in schema
            }
        )


def test_clip_caps_tags_to_five() -> None:
    with pytest.raises(ValidationError):
        Clip(
            id="x",
            start="00:00:00",  # type: ignore[arg-type]
            end="00:00:10",  # type: ignore[arg-type]
            category=Category.highlight,
            description="d",
            score=5,
            rationale="r",
            tags=["a", "b", "c", "d", "e", "f"],
        )


def test_clip_duration_properties() -> None:
    clip = Clip(
        id="x",
        start="00:00:10",  # type: ignore[arg-type]
        end="00:00:25.500",  # type: ignore[arg-type]
        category=Category.highlight,
        description="d",
        score=5,
        rationale="r",
    )
    assert clip.duration == timedelta(seconds=15, milliseconds=500)
    assert clip.duration_sec == pytest.approx(15.5)


# ---------------------------------------------------------------------------
# ClipPlan
# ---------------------------------------------------------------------------


def _sample_clip() -> dict[str, object]:
    return {
        "id": "clip_001",
        "start": "00:00:00",
        "end": "00:00:10",
        "category": "highlight",
        "description": "d",
        "score": 5,
        "rationale": "r",
        "tags": [],
    }


def test_clipplan_validates_full_payload() -> None:
    plan = ClipPlan.model_validate(
        {
            "video_id": "vid_1",
            "duration_sec": 600.0,
            "clips": [_sample_clip()],
            "metadata": {
                "vlm_provider": "openrouter",
                "vlm_model": "anthropic/claude-sonnet-4-6",
            },
        }
    )
    assert len(plan.clips) == 1
    assert plan.metadata.prompt_version == "v3"


def test_clipplan_metadata_allows_extra_fields() -> None:
    # ClipPlanMetadata is the one place we accept ``extra="allow"`` because
    # future providers may add custom telemetry fields.
    meta = ClipPlanMetadata.model_validate(
        {
            "vlm_provider": "openrouter",
            "vlm_model": "anthropic/claude-sonnet-4-6",
            "cache_hit": True,
            "rate_limit_remaining": 42,
        }
    )
    assert meta.vlm_provider == "openrouter"


# ---------------------------------------------------------------------------
# AnalysisHints
# ---------------------------------------------------------------------------


def test_analysis_hints_defaults() -> None:
    hints = AnalysisHints()
    assert hints.content_hint == ContentHint.auto
    assert hints.language == "it"


def test_analysis_hints_rejects_inverted_duration_window() -> None:
    with pytest.raises(ValidationError):
        AnalysisHints(min_duration_sec=10.0, max_duration_sec=5.0)


@pytest.mark.parametrize("bad_lang", ["italiano", "IT", "it-it", "en_US"])
def test_analysis_hints_rejects_malformed_language(bad_lang: str) -> None:
    with pytest.raises(ValidationError):
        AnalysisHints(language=bad_lang)


@pytest.mark.parametrize("good_lang", ["it", "en", "it-IT", "pt-BR"])
def test_analysis_hints_accepts_valid_language(good_lang: str) -> None:
    AnalysisHints(language=good_lang)


# ---------------------------------------------------------------------------
# Video pipeline models
# ---------------------------------------------------------------------------


def _video_meta_kwargs() -> dict[str, object]:
    return {
        "path": Path("video.mp4"),
        "duration_sec": 60.0,
        "width": 1920,
        "height": 1080,
        "fps": 30.0,
        "video_codec": "h264",
        "audio_codec": "aac",
        "container": "mp4",
        "size_bytes": 1024,
    }


def test_video_metadata_basic() -> None:
    meta = VideoMetadata.model_validate(_video_meta_kwargs())
    assert meta.aspect_ratio == pytest.approx(1920 / 1080)
    assert meta.audio_codec == "aac"


def test_video_metadata_rejects_zero_dimension() -> None:
    kwargs = _video_meta_kwargs()
    kwargs["width"] = 0
    with pytest.raises(ValidationError):
        VideoMetadata.model_validate(kwargs)


def test_video_metadata_allows_missing_audio() -> None:
    kwargs = _video_meta_kwargs()
    kwargs["audio_codec"] = None
    VideoMetadata.model_validate(kwargs)


def test_video_metadata_rejects_extra_field() -> None:
    kwargs = _video_meta_kwargs()
    kwargs["color_space"] = "bt709"
    with pytest.raises(ValidationError):
        VideoMetadata.model_validate(kwargs)


def test_scene_validates_end_after_start() -> None:
    with pytest.raises(ValidationError):
        Scene(index=0, start=timedelta(seconds=5), end=timedelta(seconds=5))


def test_scene_duration_property() -> None:
    s = Scene(index=2, start=timedelta(seconds=1), end=timedelta(seconds=4.5))
    assert s.duration_sec == pytest.approx(3.5)
    assert s.index == 2


def test_keyframe_accepts_none_scene_index_for_uniform_sampling() -> None:
    kf = Keyframe(scene_index=None, timestamp=timedelta(seconds=1), path=Path("k.jpg"))
    assert kf.scene_index is None


def test_keyframe_accepts_zero_or_positive_scene_index() -> None:
    Keyframe(scene_index=0, timestamp=timedelta(), path=Path("k.jpg"))
    Keyframe(scene_index=42, timestamp=timedelta(seconds=1), path=Path("k.jpg"))


def test_keyframe_rejects_negative_scene_index() -> None:
    # The frame_sampler uses -1 as a sentinel; ``keyframes.py`` is responsible
    # for mapping that to ``None`` before constructing the model.
    with pytest.raises(ValidationError):
        Keyframe(scene_index=-1, timestamp=timedelta(), path=Path("k.jpg"))
