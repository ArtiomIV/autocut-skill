"""Pydantic data models shared across the pipeline.

These models are the single source of truth for the shape of data exchanged
between the VLM provider, the post-processing stage, and the output writers.
"""

from __future__ import annotations

import re
from datetime import timedelta
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# Prompt template IDs the VLM prompt builder dispatches on. Kept here (not
# in ``autocut.content.profiles``) so ``AnalysisHints`` can reference the
# Literal without an import cycle.
PromptTemplateId = Literal["highlights", "talk", "hybrid", "coarse", "query"]

# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------

# Accept both ``HH:MM:SS(.mmm)`` and ``MM:SS(.mmm)`` — different models format
# clip-relative timestamps differently (Gemini 3.5 Flash emits HH:MM:SS, Gemini
# 3.1 Pro emits MM:SS). The hours group is optional and defaults to 0.
_TIMESTAMP_RE = re.compile(
    r"^(?:(?P<h>\d{1,2}):)?(?P<m>[0-5]?\d):(?P<s>[0-5]\d)(?:\.(?P<ms>\d{1,3}))?$"
)


def _parse_timestamp(value: object) -> timedelta:
    """Accept either a ``HH:MM:SS(.mmm)`` string or a numeric seconds value."""
    if isinstance(value, timedelta):
        return value
    if isinstance(value, int | float):
        if value < 0:
            raise ValueError("timestamp cannot be negative")
        return timedelta(seconds=float(value))
    if isinstance(value, str):
        match = _TIMESTAMP_RE.match(value.strip())
        if not match:
            raise ValueError(
                f"invalid timestamp {value!r}; expected HH:MM:SS(.mmm), MM:SS(.mmm) or seconds"
            )
        h = int(match.group("h") or 0)
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
    """Editing MODE for a run, not a content domain.

    The orchestrating agent picks the mode from the user's request crossed with
    the kind of video (see the routing matrix in the skill docs):

    - ``highlights``: auto-select the best, viral-worthy moments. Content-agnostic
      (boxing, skate, goals, reactions — all the same mode). If nothing clears the
      bar, the run returns NO clip rather than a best-of-nothing.
    - ``hybrid``: generic / unclear content; conservative general judgement.
    - ``talk``: speech-driven (interview, podcast). Quality tuning waits on Whisper.
    - ``auto``: input asking a detector to commit to one of the above (host path).

    A request for a SPECIFIC described moment ("the ring girl", "the KO in round 3")
    is NOT ``highlights`` — it travels as a free-text ``query`` on ``AnalysisHints``.
    """

    auto = "auto"
    highlights = "highlights"
    hybrid = "hybrid"
    talk = "talk"


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
    description: str = Field(min_length=1, max_length=500)
    score: int = Field(ge=0, le=10)
    rationale: str = Field(min_length=1, max_length=300)
    tags: list[str] = Field(default_factory=list, max_length=5)

    @field_validator("tags", mode="before")
    @classmethod
    def _coerce_tags(cls, value: object) -> object:
        """Tolerate the model's occasional malformed ``tags``.

        Models sometimes emit ``null``, a bare string, or more than five tags.
        Coerce to a clean list of at most five strings rather than failing the
        whole clip (and, with it, a long paid run).
        """
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list | tuple):
            return [str(t) for t in value if t is not None][:5]
        return value

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
    prompt_version: str = "v12"
    analysis_time_sec: float | None = None
    # Real billed cost reported by the provider (via OpenRouter usage.include),
    # summed across batches. ``None`` when the provider does not report it
    # (e.g. host agent, or the keyframe path which uses a pre-call estimate).
    cost_usd: float | None = None


class ClipPlan(BaseModel):
    """Validated VLM output. The pipeline consumes this from stage 5 onward."""

    model_config = ConfigDict(extra="forbid")

    video_id: str = Field(min_length=1, max_length=128)
    duration_sec: float = Field(gt=0)
    clips: list[Clip]
    metadata: ClipPlanMetadata


# ---------------------------------------------------------------------------
# DetectionResult — VLM classification of the content type (Phase E)
# ---------------------------------------------------------------------------


class DetectionResult(BaseModel):
    """Output of the auto-detect pre-step: the VLM classifies the video.

    The pipeline invokes ``provider.detect_content`` when the caller passes
    ``content_hint=auto`` (or no hint at all). The pipeline then picks the
    matching ``ContentProfile``. ``confidence`` below a configurable
    threshold falls back to ``HYBRID_PROFILE`` rather than acting on a
    weak signal.
    """

    model_config = ConfigDict(extra="forbid")

    content_hint: ContentHint = Field(
        description="Detected category. ``auto`` is rejected — the detector must commit."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Self-reported certainty 0-1; below the pipeline threshold we fall back.",
    )
    reasoning: str = Field(
        default="",
        max_length=300,
        description="One-sentence explanation, surfaced in logs for debuggability.",
    )

    @model_validator(mode="after")
    def _content_hint_committed(self) -> DetectionResult:
        # ``auto`` is the *input* asking us to classify; allowing it as
        # *output* would defeat the purpose of detection.
        if self.content_hint == ContentHint.auto:
            raise ValueError("DetectionResult.content_hint must commit to a concrete category")
        return self


# ---------------------------------------------------------------------------
# AnalysisHints — what we send TO the VLM
# ---------------------------------------------------------------------------


class AnalysisHints(BaseModel):
    """Hints passed in the prompt so the VLM can specialise its scoring."""

    content_hint: ContentHint = ContentHint.auto
    goal: str = Field(default="highlight reel", max_length=200)
    # Free-text request for a SPECIFIC moment, elaborated by the orchestrating
    # agent from the user's words ("find when the ring girl enters"). When set,
    # the run switches to query mode: the model hunts THIS moment instead of
    # auto-selecting the best ones, and the motion pre-filter is disabled (a
    # low-motion target must not be sampled away). ``None`` = ordinary highlight
    # selection driven by ``content_hint``.
    query: str | None = Field(default=None, max_length=300)
    language: str = Field(default="it", pattern=r"^[a-z]{2}(-[A-Z]{2})?$")
    target_clip_count: int | None = Field(default=None, ge=1, le=100)
    min_duration_sec: float = Field(default=2.0, gt=0)
    max_duration_sec: float = Field(default=60.0, gt=0)
    prompt_template: PromptTemplateId = "hybrid"
    # (start_sec, end_sec) windows where motion + audio analysis detected high
    # activity. Surfaced in the user prompt so a stills-based VLM knows where
    # to focus. Populated by the pipeline only for the host path under the
    # ``motion`` sampler; empty otherwise, in which case no block is rendered.
    motion_windows_sec: list[tuple[float, float]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_duration_window(self) -> AnalysisHints:
        if self.max_duration_sec <= self.min_duration_sec:
            raise ValueError("max_duration_sec must be greater than min_duration_sec")
        return self


# ---------------------------------------------------------------------------
# Video pipeline models (populated by ``autocut.video.*``)
# ---------------------------------------------------------------------------


class VideoMetadata(BaseModel):
    """Container-level facts produced by ``autocut.video.probe.probe_video``."""

    model_config = ConfigDict(extra="forbid")

    path: Path
    duration_sec: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0)
    video_codec: str
    audio_codec: str | None = None
    container: str
    size_bytes: int = Field(ge=0)

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height


class Scene(BaseModel):
    """A single scene segment detected by ``autocut.video.scene_detect``."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    start: Timestamp
    end: Timestamp

    @model_validator(mode="after")
    def _end_after_start(self) -> Scene:
        if self.end <= self.start:
            raise ValueError("end must be strictly greater than start")
        return self

    @property
    def duration_sec(self) -> float:
        return (self.end - self.start).total_seconds()


class Keyframe(BaseModel):
    """A single keyframe image extracted from the video.

    ``scene_index`` is ``None`` for uniform-sampling specs that are not bound
    to any detected scene, and ``>= 0`` for scene-based or hybrid samples.
    """

    model_config = ConfigDict(extra="forbid")

    scene_index: int | None = Field(default=None, ge=0)
    timestamp: Timestamp
    path: Path
