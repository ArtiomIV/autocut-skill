"""Per-content-type defaults for sampling, prompting, and cut padding.

Different kinds of source video need different treatment:

- **sport** (boxing, MMA, gameplay): action is dense and bursty inside long
  uncut shots; the VLM needs many keyframes and a prompt that demands
  multi-frame confirmation before scoring high. Clips tend to be short
  (3-20 s).
- **talk** (podcast, interview): action is verbal, frames change little;
  fewer keyframes are enough, and meaningful clips are longer (8-60 s).
- **hybrid** (auto, mixed/unknown): a balanced middle ground used when the
  content type is unclear.

Each ``ContentHint`` value maps to exactly one of these profiles via
``profile_for``. The pipeline reads the profile and forwards its fields to
the sampler, the prompt builder, and the cutter.

Pre/post-roll padding fields exist on the profile but every built-in
ships with ``0.0`` — the user wants tight cuts by default. The plumbing
is wired through the cutter so future profiles can opt in by changing a
single number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from autocut.models import ContentHint, PromptTemplateId


@dataclass(frozen=True, slots=True)
class ContentProfile:
    """Tunable knobs the pipeline reads to specialise a run for a content type."""

    name: str
    sampling_strategy: Literal["scene", "uniform", "hybrid"]
    min_keyframes: int
    min_duration_sec: float
    max_duration_sec: float
    prompt_template: PromptTemplateId
    pre_roll_sec: float = 0.0
    post_roll_sec: float = 0.0


# ---------------------------------------------------------------------------
# Built-in profiles
# ---------------------------------------------------------------------------


SPORT_PROFILE = ContentProfile(
    name="sport",
    sampling_strategy="hybrid",  # frame_sampler reroutes to dense uniform for short videos
    min_keyframes=8,
    min_duration_sec=3.0,
    max_duration_sec=20.0,
    prompt_template="sport",
)


TALK_PROFILE = ContentProfile(
    name="talk",
    sampling_strategy="hybrid",
    min_keyframes=5,
    min_duration_sec=8.0,
    max_duration_sec=30.0,
    prompt_template="talk",
)


HYBRID_PROFILE = ContentProfile(
    name="hybrid",
    sampling_strategy="hybrid",
    min_keyframes=5,
    min_duration_sec=5.0,
    max_duration_sec=45.0,
    prompt_template="hybrid",
)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


_HINT_TO_PROFILE: dict[ContentHint, ContentProfile] = {
    ContentHint.boxing: SPORT_PROFILE,
    ContentHint.sport: SPORT_PROFILE,
    ContentHint.gameplay: SPORT_PROFILE,
    ContentHint.talk: TALK_PROFILE,
    ContentHint.podcast: TALK_PROFILE,
    ContentHint.auto: HYBRID_PROFILE,
    ContentHint.other: HYBRID_PROFILE,
}


def profile_for(hint: ContentHint) -> ContentProfile:
    """Return the ``ContentProfile`` mapped to ``hint``; defaults to hybrid."""
    return _HINT_TO_PROFILE.get(hint, HYBRID_PROFILE)
