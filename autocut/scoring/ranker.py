"""Final scoring + ranking layer.

The ranker combines the VLM's subjective ``Clip.score`` with the objective
``heuristic_score`` (0.7 / 0.3 weighted) and produces a sorted list of
``RankedClip`` records. The pipeline uses this list to drive the output
writers.

Two filters happen here:
1. ``min_score`` from ``ScoringConfig.min_score`` — drop low-quality clips.
2. ``max_clips`` from ``ScoringConfig.max_clips`` — keep only the top N.
"""

from __future__ import annotations

from dataclasses import dataclass

from autocut.config import ScoringConfig
from autocut.models import Clip, ClipPlan
from autocut.scoring.heuristics import heuristic_score

# Weights for the final score; must sum to 1.0.
_VLM_WEIGHT: float = 0.7
_HEURISTIC_WEIGHT: float = 0.3


@dataclass(frozen=True, slots=True)
class RankedClip:
    """A clip with both raw scores attached, ready for the output writers."""

    clip: Clip
    vlm_score: int
    heuristic_score: int
    final_score: int


def final_score(vlm: int, heur: int) -> int:
    """Compute the weighted final score, clamped to ``[0, 10]``."""
    raw = _VLM_WEIGHT * float(vlm) + _HEURISTIC_WEIGHT * float(heur)
    return max(0, min(10, round(raw)))


def rank_clips(plan: ClipPlan, config: ScoringConfig) -> list[RankedClip]:
    """Score, filter, and sort every clip in ``plan``.

    Order: highest final_score first; ties broken by the chronological
    ``start`` timestamp so output naming stays predictable.
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
    return scored[: config.max_clips]
