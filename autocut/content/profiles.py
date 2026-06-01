"""Per-content-type defaults for sampling, prompting, and cut padding.

Profiles are keyed by editing MODE, not content domain:

- **highlights**: auto-select the best, viral-worthy moments. Content-agnostic
  (boxing, skate, goals, reactions all share this mode). Action tends to be
  short and bursty, so clips are short (3-20 s) and the prompt is conservative —
  if nothing clears the bar it returns no clip at all.
- **talk** (interview, podcast): action is verbal, frames change little; fewer
  keyframes are enough and meaningful clips are longer (8-30 s). Quality tuning
  waits on Whisper.
- **hybrid** (auto / mixed / unknown): a balanced middle ground used when the
  mode is unclear.

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
    # Per-mode keep threshold: when set, overrides ``ScoringConfig.min_score``
    # so a mode can be stricter than the global default. ``None`` means "use
    # the config default". Highlights keeps final >= 7: with the heuristic now a
    # neutral nudge, ``final == vlm`` for a well-sized clip, so 7 means "keep the
    # model's strong moments (a near-KO, a decisive beat) and above", while VLM=6
    # filler (ring walk-ins, clinches) stays out. An empty result is still valid
    # when nothing clears the bar.
    min_score: int | None = None


# ---------------------------------------------------------------------------
# Built-in profiles
# ---------------------------------------------------------------------------


HIGHLIGHTS_PROFILE = ContentProfile(
    name="highlights",
    sampling_strategy="hybrid",  # frame_sampler reroutes to dense uniform for short videos
    min_keyframes=8,
    min_duration_sec=3.0,
    max_duration_sec=20.0,
    prompt_template="highlights",
    min_score=7,  # final == vlm now: keep the model's strong moments (>=7), drop VLM=6 filler
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
    ContentHint.highlights: HIGHLIGHTS_PROFILE,
    ContentHint.talk: TALK_PROFILE,
    ContentHint.hybrid: HYBRID_PROFILE,
    ContentHint.auto: HYBRID_PROFILE,
}


def profile_for(hint: ContentHint) -> ContentProfile:
    """Return the ``ContentProfile`` mapped to ``hint``; defaults to hybrid."""
    return _HINT_TO_PROFILE.get(hint, HYBRID_PROFILE)
