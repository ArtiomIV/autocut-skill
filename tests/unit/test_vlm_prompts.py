"""Unit tests for ``autocut.vlm.prompts`` — schema + template rendering."""

from __future__ import annotations

import json
from datetime import timedelta

from autocut.models import AnalysisHints, ContentHint
from autocut.video.frame_sampler import FrameSpec
from autocut.vlm.prompts import (
    PROMPT_VERSION,
    build_system_prompt,
    build_user_prompt,
    clip_plan_schema,
    format_timestamp,
)


def test_schema_is_valid_json() -> None:
    schema = json.loads(clip_plan_schema())
    assert isinstance(schema, dict)
    assert "properties" in schema
    assert "clips" in schema["properties"]


def test_schema_includes_extra_forbid_for_clip() -> None:
    schema = json.loads(clip_plan_schema())
    clip_schema = schema["$defs"]["Clip"]
    assert clip_schema.get("additionalProperties") is False


def test_schema_enumerates_categories() -> None:
    schema = json.loads(clip_plan_schema())
    enum_values = schema["$defs"]["Category"]["enum"]
    assert "highlight" in enum_values
    assert "filler" in enum_values
    assert len(enum_values) == 5


def test_system_prompt_includes_duration_bounds() -> None:
    hints = AnalysisHints(min_duration_sec=4.5, max_duration_sec=42.0)
    text = build_system_prompt(hints)
    assert "4.5" in text
    assert "42.0" in text


def test_user_prompt_lists_every_keyframe_timestamp() -> None:
    specs = [
        FrameSpec(timestamp=timedelta(seconds=1.0), scene_index=0),
        FrameSpec(timestamp=timedelta(seconds=4.5), scene_index=1),
        FrameSpec(timestamp=timedelta(seconds=12.0), scene_index=-1),
    ]
    text = build_user_prompt(
        video_id="match_001",
        duration_sec=60.0,
        hints=AnalysisHints(content_hint=ContentHint.boxing),
        specs=specs,
    )
    assert "frame 1" in text
    assert "00:00:01.000" in text
    assert "frame 2" in text
    assert "00:00:04.500" in text
    assert "frame 3" in text
    assert "00:00:12.000" in text
    assert "match_001" in text
    assert PROMPT_VERSION in text
    assert "boxing" in text


def test_format_timestamp_matches_extractor_format() -> None:
    assert format_timestamp(timedelta(seconds=0)) == "00:00:00.000"
    assert format_timestamp(timedelta(seconds=12.345)) == "00:00:12.345"
    assert format_timestamp(timedelta(hours=1, minutes=2, seconds=3)) == "01:02:03.000"


# ---------------------------------------------------------------------------
# A.5.2: specialized system prompt templates
# ---------------------------------------------------------------------------


def test_system_prompt_sport_template_includes_multi_frame_rule() -> None:
    hints = AnalysisHints(prompt_template="sport")
    text = build_system_prompt(hints)
    # The boxing-feedback rule must be present so the VLM is conservative.
    lowered = text.lower()
    assert "multi" in lowered or "consecutive" in lowered
    assert "sport" in lowered or "combat" in lowered or "action" in lowered


def test_system_prompt_talk_template_targets_verbal_content() -> None:
    hints = AnalysisHints(prompt_template="talk")
    text = build_system_prompt(hints).lower()
    assert "talk" in text or "podcast" in text or "interview" in text
    # Talk template should not require multi-frame action confirmation.
    assert "verbal" in text or "speaker" in text or "audience" in text


def test_system_prompt_hybrid_template_is_neutral() -> None:
    hints = AnalysisHints(prompt_template="hybrid")
    text = build_system_prompt(hints).lower()
    assert "mixed" in text or "unknown" in text or "general highlight" in text


def test_system_prompt_falls_back_to_hybrid_for_unknown_template() -> None:
    # Defensive: if a future caller bypasses Pydantic and sets an unknown
    # template ID, build_system_prompt must not crash.
    hints = AnalysisHints()
    hints.prompt_template = "nonexistent"  # type: ignore[assignment]
    text = build_system_prompt(hints).lower()
    # Should produce SOME prompt with shared core rules.
    assert "highlight" in text
    assert "json" in text
