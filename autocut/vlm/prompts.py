"""Prompt templates and the JSON schema we hand to the VLM.

Two design choices that matter for cost and correctness:

1. The JSON schema is derived from ``ClipPlan`` so the prompt and the
   downstream validator can never drift apart. If we add a field to
   ``Clip`` tomorrow, the schema in the prompt updates automatically.
2. The prompt is versioned (``PROMPT_VERSION``). Output ``ClipPlan``
   records carry the version they were generated with, so when we
   iterate on phrasing we can audit which prompt produced which run.
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Final

from autocut.models import AnalysisHints, ClipPlan
from autocut.video.frame_sampler import FrameSpec

PROMPT_VERSION: Final[str] = "v1"


# ---------------------------------------------------------------------------
# JSON schema (derived from ClipPlan)
# ---------------------------------------------------------------------------


def clip_plan_schema() -> str:
    """Return the JSON Schema for ``ClipPlan`` as a compact string."""
    schema = ClipPlan.model_json_schema()
    return json.dumps(schema, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT: Final[str] = """\
You are a video editor assistant analysing keyframes extracted from a video.
You will receive an ordered list of keyframes, each labelled with the
timestamp at which it was sampled from the source video.

Your job: identify the segments worth keeping as standalone highlight clips,
then return them as STRICT JSON matching the provided schema. No prose
before or after the JSON. No explanations. No markdown fences.

Hard rules:
- Each clip must be between {min_dur:.1f} and {max_dur:.1f} seconds long.
- Clips must NOT overlap.
- ``start`` and ``end`` use the ``HH:MM:SS.mmm`` format.
- ``score`` is an integer 0-10 (10 = mandatory, must keep).
- ``category`` is one of: highlight, reaction, dialogue, transition, filler.
- ``tags`` is a list of 1-5 short strings.
- Be conservative: prefer fewer high-score clips over many low-score ones.
- If no segment is worth keeping, return an empty ``clips`` array.
"""


_USER_TEMPLATE: Final[str] = """\
Video metadata: content type = {content_hint}, language = {language},
goal = {goal}, source duration = {duration_sec:.2f}s.

Keyframes (in chronological order):
{keyframe_lines}

Return JSON matching this schema (do not output the schema itself, output
your analysis):
{schema}

Fill ``video_id`` with {video_id!r}, ``duration_sec`` with {duration_sec},
and ``metadata.vlm_provider`` / ``metadata.vlm_model`` with the model
that produced this analysis. ``metadata.prompt_version`` must equal {pv!r}.
"""


def format_timestamp(ts: timedelta) -> str:
    """Format ``HH:MM:SS.mmm`` so timestamps in the prompt line up with the schema."""
    total_ms = max(0, int(ts.total_seconds() * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, milliseconds = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def build_system_prompt(hints: AnalysisHints) -> str:
    """Return the system message, hardcoding duration bounds from hints."""
    return _SYSTEM_PROMPT.format(min_dur=hints.min_duration_sec, max_dur=hints.max_duration_sec)


def build_user_prompt(
    video_id: str,
    duration_sec: float,
    hints: AnalysisHints,
    specs: list[FrameSpec],
) -> str:
    """Return the user message: hints + keyframe timeline + target schema.

    Note: the actual image bytes are attached by the provider as separate
    image parts in the multimodal payload; here we only include the
    *timestamps* so the model can map each image to a point in the video.
    """
    keyframe_lines = "\n".join(
        f"  - frame {i + 1}: t = {format_timestamp(spec.timestamp)}" for i, spec in enumerate(specs)
    )
    return _USER_TEMPLATE.format(
        content_hint=hints.content_hint.value,
        language=hints.language,
        goal=hints.goal,
        duration_sec=duration_sec,
        keyframe_lines=keyframe_lines,
        schema=clip_plan_schema(),
        video_id=video_id,
        pv=PROMPT_VERSION,
    )
