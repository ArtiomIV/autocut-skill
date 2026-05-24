"""Pydantic data models shared across the pipeline.

These models are the single source of truth for the shape of data exchanged
between the VLM provider, the post-processing stage, and the output writers.
"""

from __future__ import annotations

import re
from datetime import timedelta
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------

_TIMESTAMP_RE = re.compile(r"^(?P<h>\d{1,2}):(?P<m>[0-5]\d):(?P<s>[0-5]\d)(?:\.(?P<ms>\d{1,3}))?$")


def _parse_timestamp(value: object) -> timedelta:
    """Accept either a ``HH:MM:SS(.mmm)`` string or a numeric seconds value."""
    if isinstance(value, timedelta):
        return value
    if isinstance(value, (int, float)):
        if value < 0:
            raise ValueError("timestamp cannot be negative")
        return timedelta(seconds=float(value))
    if isinstance(value, str):
        match = _TIMESTAMP_RE.match(value.strip())
        if not match:
            raise ValueError(f"invalid timestamp {value!r}; expected HH:MM:SS(.mmm) or seconds")
        h = int(match.group("h"))
        m = int(match.group("m"))
        s = int(match.group("s"))
        ms_raw = match.group("ms") or "0"
        # Right-pad to 3 digits so "0.5" reads as 500 ms, not 5 ms.
        ms = int(ms_raw.ljust(3, "0"))
        return timedelta(hours=h, minutes=m, seconds=s, milliseconds=ms)
    raise TypeError(f"unsupported timestamp type: {type(value).__name__}")


Timestamp = Annotated[timedelta, BeforeValidator(_parse_timestamp)]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Category(StrEnum):
    """Coarse category assigned by the VLM to every clip candidate."""

    highlight = "highlight"
    reaction = "reaction"
    dialogue = "dialogue"
    transition = "transition"
    filler = "filler"


class ContentHint(StrEnum):
    """Type of content. ``auto`` lets the VLM decide."""

    auto = "auto"
    boxing = "boxing"
    talk = "talk"
    podcast = "podcast"
    sport = "sport"
    gameplay = "gameplay"
    other = "other"


# ---------------------------------------------------------------------------
# Clip
# ---------------------------------------------------------------------------


class Clip(BaseModel):
    """A single highlight candidate produced by the VLM."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=64)
    start: Timestamp
    end: Timestamp
    category: Category
    description: str = Field(min_length=1, max_length=200)
    score: int = Field(ge=0, le=10)
    rationale: str = Field(min_length=1, max_length=300)
    tags: list[str] = Field(default_factory=list, max_length=5)

    @model_validator(mode="after")
    def _end_after_start(self) -> Clip:
        if self.end <= self.start:
            raise ValueError("end must be strictly greater than start")
        return self

    @property
    def duration(self) -> timedelta:
        return self.end - self.start

    @property
    def duration_sec(self) -> float:
        return self.duration.total_seconds()


# ---------------------------------------------------------------------------
# ClipPlan — the full VLM response
# ---------------------------------------------------------------------------


class ClipPlanMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    vlm_provider: str
    vlm_model: str
    prompt_version: str = "v1"
    analysis_time_sec: float | None = None


class ClipPlan(BaseModel):
    """Validated VLM output. The pipeline consumes this from stage 5 onward."""

    model_config = ConfigDict(extra="forbid")

    video_id: str = Field(min_length=1, max_length=128)
    duration_sec: float = Field(gt=0)
    clips: list[Clip]
    metadata: ClipPlanMetadata


# ---------------------------------------------------------------------------
# AnalysisHints — what we send TO the VLM
# ---------------------------------------------------------------------------


class AnalysisHints(BaseModel):
    """Hints passed in the prompt so the VLM can specialise its scoring."""

    content_hint: ContentHint = ContentHint.auto
    goal: str = Field(default="highlight reel", max_length=200)
    language: str = Field(default="it", pattern=r"^[a-z]{2}(-[A-Z]{2})?$")
    target_clip_count: int | None = Field(default=None, ge=1, le=100)
    min_duration_sec: float = Field(default=2.0, gt=0)
    max_duration_sec: float = Field(default=60.0, gt=0)

    @model_validator(mode="after")
    def _check_duration_window(self) -> AnalysisHints:
        if self.max_duration_sec <= self.min_duration_sec:
            raise ValueError("max_duration_sec must be greater than min_duration_sec")
        return self
