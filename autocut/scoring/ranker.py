"""Final scoring + ranking layer.

The VLM's subjective ``Clip.score`` is the PRIMARY signal. The objective
``heuristic_score`` is not a co-equal vote but a NUDGE around a neutral
baseline: a neutral heuristic leaves the VLM score untouched, a heuristic
below neutral penalises (e.g. an absurd duration), above neutral boosts. In
v0.1.0 the only heuristic is the duration guardrail, so for a well-sized clip
``final_score == vlm`` — the model decides, the heuristic only protects
against a malformed too-short / too-long clip. The motion + audio signals
landing in v0.1.1 plug into the SAME nudge without re-plumbing the blend.

Three filters happen here:
1. ``min_score`` from ``ScoringConfig.min_score`` — drop low-quality clips.
2. overlap suppression — drop a clip that overlaps a higher-scored one (the
   final gate guaranteeing a non-overlapping timeline, regardless of source).
3. ``max_clips`` from ``ScoringConfig.max_clips`` — keep only the top N.
"""

from __future__ import annotations

from dataclasses import dataclass

from autocut.config import ScoringConfig
from autocut.models import Clip, ClipPlan
from autocut.scoring.heuristics import heuristic_score

# The heuristic is expressed on a 0-10 scale where this value means "no signal,
# neither good nor bad" — at neutral the final score equals the VLM score.
_HEURISTIC_NEUTRAL: int = 5

# Two kept clips that cover more than this fraction of the SHORTER clip are the
# same moment (e.g. the two-pass found a near-KO twice with slightly different
# bounds). Keep the higher-scored one. A fraction (not IoU) so a small clip
# swallowed by a larger one is caught even when their IoU is modest.
_MAX_OVERLAP_FRACTION: float = 0.5


@dataclass(frozen=True, slots=True)
class RankedClip:
    """A clip with both raw scores attached, ready for the output writers."""

    clip: Clip
    vlm_score: int
    heuristic_score: int
    final_score: int


def final_score(vlm: int, heur: int) -> int:
    """Adjust the VLM score by the heuristic's deviation from neutral.

    ``final = vlm + (heur - neutral)``, clamped to ``[0, 10]``. A neutral
    heuristic leaves the VLM score unchanged (``final == vlm``); below neutral
    penalises, above neutral boosts. The VLM stays the decision-maker; the
    heuristic only nudges.
    """
    return max(0, min(10, vlm + (heur - _HEURISTIC_NEUTRAL)))


def _overlap_fraction(a: Clip, b: Clip) -> float:
    """Fraction of the SHORTER clip covered by its overlap with the other."""
    inter = max(
        0.0,
        min(a.end.total_seconds(), b.end.total_seconds())
        - max(a.start.total_seconds(), b.start.total_seconds()),
    )
    shorter = min(a.duration_sec, b.duration_sec)
    return inter / shorter if shorter > 0 else 0.0


def rank_clips(plan: ClipPlan, config: ScoringConfig) -> list[RankedClip]:
    """Score, filter, deduplicate overlaps, and sort every clip in ``plan``.

    Order: highest final_score first; ties broken by the chronological
    ``start`` timestamp so output naming stays predictable. Walking in that
    order, a clip is dropped when it overlaps an already-kept (higher-scored)
    clip beyond ``_MAX_OVERLAP_FRACTION`` — so the strongest version of a
    moment wins and the final timeline never double-counts it.
    """
    scored: list[RankedClip] = []
    for clip in plan.clips:
        heur = heuristic_score(clip)
        ranked = RankedClip(
            clip=clip,
            vlm_score=clip.score,
            heuristic_score=heur,
            final_score=final_score(clip.score, heur),
        )
        if ranked.final_score < config.min_score:
            continue
        scored.append(ranked)

    scored.sort(key=lambda r: (-r.final_score, r.clip.start))

    kept: list[RankedClip] = []
    for r in scored:
        if any(_overlap_fraction(r.clip, k.clip) > _MAX_OVERLAP_FRACTION for k in kept):
            continue
        kept.append(r)

    return kept[: config.max_clips]
