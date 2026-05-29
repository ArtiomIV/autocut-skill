"""L2 — the shared video-analysis engine: compress, batch, loop, merge.

This is the reusable core of the video-input path, consumed by ``run`` (and a
future ``locate``). Given a source video and a provider that can analyse a
video clip, it:

1. compresses the source to analysis grade (small base64-able copy),
2. splits long videos into overlapping ~5-minute batches (the base64 inline
   ceiling), short videos staying a single pass,
3. checks the cost cap ONCE up front, then runs the batch loop autonomously —
   no per-batch confirmation,
4. calls the provider once per batch (timestamps come back relative to the
   batch),
5. merges the per-batch plans back to absolute source time with boundary
   dedup, reusing ``autocut.chunking``.

It is prompt-agnostic: the caller picks sport/talk/hybrid via ``hints``. The
batch/overlap/offset/dedup logic lives here ONCE so ``run`` and ``locate`` do
not each reimplement it.
"""

from __future__ import annotations

import contextlib
import logging
import tempfile
from collections.abc import Callable, Iterator
from datetime import timedelta
from pathlib import Path
from typing import Protocol

from autocut.chunking import Chunk, merge_plans, split_into_chunks
from autocut.models import AnalysisHints, ClipPlan
from autocut.video.compress import compress_for_vlm
from autocut.video.cutter import CutRequest, cut_clip

log = logging.getLogger(__name__)


class VideoAnalysisError(RuntimeError):
    """Raised when the video-analysis engine cannot complete (incl. cost cap)."""


# 5-minute batches: matches the ~20MB base64 inline ceiling for a compressed
# clip. Anything longer is split; anything shorter is a single pass.
_BATCH_DURATION_SEC: float = 300.0
# Rough pre-call estimate for the cap gate only (~$0.0005/s of video, from the
# validated probe). The real billed cost is logged per call via usage.include;
# the precise estimate model is a separate task.
_USD_PER_VIDEO_SECOND: float = 0.0005


class SupportsVideoAnalysis(Protocol):
    """Structural type for any provider L2 can drive (e.g. OpenRouterProvider)."""

    async def analyze_video_clip(
        self,
        clip_path: Path,
        hints: AnalysisHints,
        *,
        video_id: str,
        clip_duration_sec: float,
        timeout_sec: int = 300,
    ) -> ClipPlan: ...


async def analyze_video(
    src: Path,
    hints: AnalysisHints,
    provider: SupportsVideoAnalysis,
    *,
    video_id: str,
    duration_sec: float,
    cost_cap_usd: float,
    confirm_cost: Callable[[float, float], bool] | None = None,
    work_dir: Path | None = None,
) -> ClipPlan:
    """Analyse ``src`` as video and return a merged, absolute-time ``ClipPlan``.

    ``confirm_cost(estimated_usd, cap_usd)`` is called only if the rough
    estimate exceeds ``cost_cap_usd``; returning ``False`` (or passing ``None``)
    aborts with ``VideoAnalysisError``. ``work_dir`` is where the compressed
    copy and batch segments are written; a temp dir is used when omitted.
    """
    if duration_sec <= 0:
        raise VideoAnalysisError("duration_sec must be > 0")

    overlap_sec = min(max(0.0, hints.max_duration_sec / 2.0), _BATCH_DURATION_SEC / 2.0)
    chunks = split_into_chunks(
        duration_sec, chunk_duration_sec=_BATCH_DURATION_SEC, overlap_sec=overlap_sec
    )
    single_pass = not chunks
    if single_pass:
        chunks = [Chunk(index=0, start_sec=0.0, end_sec=duration_sec)]

    _gate_cost(duration_sec, cost_cap_usd, confirm_cost)

    with _scratch_dir(work_dir) as scratch:
        compressed = compress_for_vlm(src, scratch / "compressed.mp4")
        plans: list[ClipPlan] = []
        for chunk in chunks:
            segment = _segment_for(chunk, compressed, scratch, single_pass=single_pass)
            log.info(
                "video analysis: batch %d/%d [%.1f-%.1f s]",
                chunk.index + 1,
                len(chunks),
                chunk.start_sec,
                chunk.end_sec,
            )
            plans.append(
                await provider.analyze_video_clip(
                    segment,
                    hints,
                    video_id=video_id,
                    clip_duration_sec=chunk.duration_sec,
                )
            )

    merged = merge_plans(plans, chunks, video_id=video_id, duration_sec=duration_sec)
    log.info("video analysis: %d batch(es) -> %d merged clip(s)", len(chunks), len(merged.clips))
    return merged


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _gate_cost(
    duration_sec: float,
    cost_cap_usd: float,
    confirm_cost: Callable[[float, float], bool] | None,
) -> None:
    """One upfront cost check; raise unless under cap or explicitly confirmed."""
    estimated = duration_sec * _USD_PER_VIDEO_SECOND
    if estimated <= cost_cap_usd:
        return
    approved = confirm_cost(estimated, cost_cap_usd) if confirm_cost else False
    if not approved:
        raise VideoAnalysisError(
            f"estimated video cost {estimated:.4f} USD exceeds cap {cost_cap_usd:.2f} USD; "
            f"raise the cap or accept the prompt"
        )


def _segment_for(chunk: Chunk, compressed: Path, scratch: Path, *, single_pass: bool) -> Path:
    """Return the clip file for ``chunk``: the whole compressed video (single
    pass) or a stream-copied batch segment cut from it."""
    if single_pass:
        return compressed
    segment = scratch / f"batch_{chunk.index:03d}.mp4"
    cut_clip(
        compressed,
        CutRequest(
            start=timedelta(seconds=chunk.start_sec),
            end=timedelta(seconds=chunk.end_sec),
            output_path=segment,
        ),
    )
    return segment


@contextlib.contextmanager
def _scratch_dir(work_dir: Path | None) -> Iterator[Path]:
    """Yield ``work_dir`` (created) or a self-cleaning temp dir when omitted."""
    if work_dir is not None:
        work_dir.mkdir(parents=True, exist_ok=True)
        yield work_dir
        return
    with tempfile.TemporaryDirectory(prefix="autocut_video_") as tmp:
        yield Path(tmp)
