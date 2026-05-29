"""Split long videos into overlapping chunks and merge their ClipPlans.

**Standalone helper, not wired into the default pipeline.** Phase D's
motion sampling already addresses the "too many keyframes for one VLM
call" problem at the source: it samples densely only inside hot windows
and sparsely elsewhere, so a 30-minute match typically produces 300-500
keyframes — well within Gemini Flash's context and budget.

This module exists for callers who want to build a chunked workflow
manually (e.g. a script processing multi-hour broadcasts where smart
sampling still produces too many frames):

    from autocut.chunking import split_into_chunks, merge_plans
    chunks = split_into_chunks(duration, chunk_duration_sec=600, overlap_sec=20)
    plans = [run_pipeline_on(chunk) for chunk in chunks]
    merged = merge_plans(plans, chunks, video_id=..., duration_sec=duration)

Two correctness concerns ``merge_plans`` handles:

1. **Boundary effects** — an action that straddles two chunks could be
   chopped in half and missed by both. Each chunk extends ``overlap_sec``
   into the next so the action falls fully inside at least one chunk.
2. **Duplicate detection** — if the same clip is proposed by both
   chunks (because of the overlap), we keep the one with the higher
   ``score`` and drop the other via temporal IoU.

Known limitations (why this is not the default):

- Score normalisation across chunks is naive; per-chunk score
  distributions are treated as comparable when they often aren't.
- The VLM loses comparative judgement: each chunk is evaluated in
  isolation rather than picking globally best moments.
- Per-chunk prompt boilerplate multiplies the call cost.

The functions here are pure: no I/O, no subprocess, no model calls.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta

from autocut.models import Clip, ClipPlan, ClipPlanMetadata

# Hard lower bound: chunks smaller than this are pointless — every chunk
# pays a per-call provider tax (prompt boilerplate, network round-trip).
_MIN_CHUNK_DURATION_SEC: float = 30.0


@dataclass(frozen=True, slots=True)
class Chunk:
    """One contiguous slice of the source video, with optional overlap."""

    index: int
    start_sec: float
    end_sec: float

    def __post_init__(self) -> None:
        if self.end_sec <= self.start_sec:
            raise ValueError("Chunk end_sec must be strictly greater than start_sec")
        if self.start_sec < 0:
            raise ValueError("Chunk start_sec must be >= 0")

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


def split_into_chunks(
    duration_sec: float,
    *,
    chunk_duration_sec: float,
    overlap_sec: float = 0.0,
) -> list[Chunk]:
    """Split a ``duration_sec`` timeline into overlapping chunks.

    Each chunk has length up to ``chunk_duration_sec + overlap_sec`` and
    advances by ``chunk_duration_sec`` between iterations. The final chunk
    is clipped at the timeline end so we never sample past EOF.

    Returns ``[]`` if ``duration_sec`` is below ``chunk_duration_sec`` —
    the caller should skip chunking and run a single pass instead.
    """
    if duration_sec <= 0:
        raise ValueError("duration_sec must be > 0")
    if chunk_duration_sec < _MIN_CHUNK_DURATION_SEC:
        raise ValueError(
            f"chunk_duration_sec must be >= {_MIN_CHUNK_DURATION_SEC} s (got {chunk_duration_sec})"
        )
    if overlap_sec < 0:
        raise ValueError("overlap_sec must be >= 0")
    if overlap_sec >= chunk_duration_sec:
        raise ValueError("overlap_sec must be < chunk_duration_sec")

    if duration_sec <= chunk_duration_sec:
        return []

    chunks: list[Chunk] = []
    index = 0
    start = 0.0
    while start < duration_sec:
        end = min(duration_sec, start + chunk_duration_sec + overlap_sec)
        chunks.append(Chunk(index=index, start_sec=start, end_sec=end))
        if end >= duration_sec:
            break
        start += chunk_duration_sec
        index += 1
    return chunks


# ---------------------------------------------------------------------------
# Plan merging
# ---------------------------------------------------------------------------


def temporal_iou(
    a_start: float,
    a_end: float,
    b_start: float,
    b_end: float,
) -> float:
    """Return the temporal Intersection-over-Union of two intervals in [0, 1]."""
    intersection = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    if union <= 0:
        return 0.0
    return intersection / union


def merge_plans(
    plans: list[ClipPlan],
    chunks: list[Chunk],
    *,
    video_id: str,
    duration_sec: float,
    iou_threshold: float = 0.5,
) -> ClipPlan:
    """Combine per-chunk ``ClipPlan`` records into one global plan.

    Steps:
    1. Shift every clip's ``start`` / ``end`` by its chunk's ``start_sec``
       so timestamps move from chunk-local to source-global.
    2. Sort by descending VLM ``score`` (higher = stronger signal).
    3. Greedy dedupe: walk the sorted list, keep a clip unless its temporal
       IoU with an already-kept clip exceeds ``iou_threshold``.
    4. Re-sort the survivors chronologically so the output plan reads
       front-to-back over the timeline.

    The returned ``ClipPlan`` inherits ``metadata`` from the first input
    plan (provider/model are identical across chunks because the pipeline
    uses one provider for the whole run).
    """
    if len(plans) != len(chunks):
        raise ValueError(
            f"plans and chunks length mismatch: {len(plans)} plans vs {len(chunks)} chunks"
        )
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")
    if not plans:
        raise ValueError("merge_plans requires at least one ClipPlan")

    # 1. Shift every clip to the global timeline.
    shifted: list[Clip] = []
    for plan, chunk in zip(plans, chunks, strict=True):
        offset = timedelta(seconds=chunk.start_sec)
        for clip in plan.clips:
            # ``Clip.id`` is capped at 64 chars; prepending the chunk index
            # could push it over. Truncate defensively from the right side
            # of the original id so the chunk prefix is always preserved.
            new_id = f"c{chunk.index}_{clip.id}"[:64]
            shifted.append(
                Clip(
                    id=new_id,
                    start=clip.start + offset,
                    end=clip.end + offset,
                    category=clip.category,
                    description=clip.description,
                    score=clip.score,
                    rationale=clip.rationale,
                    tags=list(clip.tags),
                )
            )

    # 2. Sort by descending VLM score, ties broken by chronological start.
    shifted.sort(key=lambda c: (-c.score, c.start))

    # 3. Greedy IoU dedupe.
    kept: list[Clip] = []
    for clip in shifted:
        keep = True
        for survivor in kept:
            iou = temporal_iou(
                clip.start.total_seconds(),
                clip.end.total_seconds(),
                survivor.start.total_seconds(),
                survivor.end.total_seconds(),
            )
            if iou >= iou_threshold:
                keep = False
                break
        if keep:
            kept.append(clip)

    # 4. Chronological order for the final plan.
    kept.sort(key=lambda c: c.start)

    return ClipPlan(
        video_id=video_id,
        duration_sec=duration_sec,
        clips=kept,
        metadata=ClipPlanMetadata(
            vlm_provider=plans[0].metadata.vlm_provider,
            vlm_model=plans[0].metadata.vlm_model,
            prompt_version=plans[0].metadata.prompt_version,
            analysis_time_sec=_sum_optional(plans, lambda m: m.analysis_time_sec),
            cost_usd=_sum_optional(plans, lambda m: m.cost_usd),
        ),
    )


def _sum_optional(
    plans: list[ClipPlan],
    field: Callable[[ClipPlanMetadata], float | None],
) -> float | None:
    """Sum a per-chunk optional metric; return None if any chunk reported None."""
    total = 0.0
    for plan in plans:
        value = field(plan.metadata)
        if value is None:
            return None
        total += value
    return total
