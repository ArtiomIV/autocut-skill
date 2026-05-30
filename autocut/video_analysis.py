"""L2 — the shared media-analysis engine: prepare, batch, loop, merge.

This is the reusable core of the direct-payload path, consumed by ``run`` for
both the video and the audio routes. Given a source video and a provider that
can analyse a media clip, it:

1. prepares an analysis-grade copy of the source (compress for video, extract a
   small mono MP3 for audio),
2. splits long sources into overlapping batches — ~5 minutes for video (the
   base64 inline ceiling), ~10 minutes for audio (size is not the limit; short
   batches bound the model's timestamp drift on long audio); short sources stay
   one pass,
3. checks the cost cap ONCE up front, then runs the batch loop autonomously —
   no per-batch confirmation,
4. calls the provider once per batch (timestamps come back relative to the
   batch),
5. merges the per-batch plans back to absolute source time with boundary
   dedup, reusing ``autocut.chunking``.

The batch/overlap/offset/dedup logic lives here ONCE so the video and audio
routes (and a future ``locate``) do not each reimplement it; ``analyze_video``
and ``analyze_audio`` are thin wrappers over ``_run_batched`` that differ only
in how they prepare the clip and which provider method they call.
"""

from __future__ import annotations

import contextlib
import logging
import tempfile
from collections.abc import Awaitable, Callable, Iterator
from datetime import timedelta
from pathlib import Path
from typing import Protocol

from autocut.chunking import Chunk, merge_plans, split_into_chunks
from autocut.models import AnalysisHints, ClipPlan
from autocut.video.audio_extract import extract_audio_for_vlm
from autocut.video.compress import compress_for_vlm
from autocut.video.cutter import CutRequest, cut_clip

log = logging.getLogger(__name__)


class VideoAnalysisError(RuntimeError):
    """Raised when the media-analysis engine cannot complete (incl. cost cap)."""


# Video: 5-minute batches matching the ~20MB base64 inline ceiling.
_VIDEO_BATCH_DURATION_SEC: float = 300.0
# Audio: 10-minute batches. Audio is far more compact (~7MB / 30min mono MP3),
# so the driver is NOT size but the model's timestamp accuracy over a long clip:
# on a long audio the model has no reliable internal clock and its timestamps
# drift, so shorter batches bound the within-batch timing error. 10 min was found
# to keep the drift acceptable in practice; if it ever regresses, Whisper
# word-level timestamps are the definitive fix.
_AUDIO_BATCH_DURATION_SEC: float = 600.0
# Rough pre-call estimates for the cap gate only; the real billed cost comes
# back per call via usage.include. Audio is cheaper per second than video.
_USD_PER_VIDEO_SECOND: float = 0.0005
_USD_PER_AUDIO_SECOND: float = 0.00003


class SupportsVideoAnalysis(Protocol):
    """Structural type for any provider with ``analyze_video_clip``."""

    async def analyze_video_clip(
        self,
        clip_path: Path,
        hints: AnalysisHints,
        *,
        video_id: str,
        clip_duration_sec: float,
        timeout_sec: int = 300,
    ) -> ClipPlan: ...


class SupportsAudioAnalysis(Protocol):
    """Structural type for any provider with ``analyze_audio_clip``."""

    async def analyze_audio_clip(
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
    """Analyse ``src`` as VIDEO and return a merged, absolute-time ``ClipPlan``.

    Compresses the source to analysis grade and sends it (batched at ~5 min when
    long) to ``provider.analyze_video_clip``. See ``_run_batched`` for the cost
    gate / scratch-dir semantics.
    """

    async def _analyze(segment: Path, chunk: Chunk) -> ClipPlan:
        return await provider.analyze_video_clip(
            segment, hints, video_id=video_id, clip_duration_sec=chunk.duration_sec
        )

    return await _run_batched(
        src,
        hints,
        video_id=video_id,
        duration_sec=duration_sec,
        cost_cap_usd=cost_cap_usd,
        confirm_cost=confirm_cost,
        work_dir=work_dir,
        batch_duration_sec=_VIDEO_BATCH_DURATION_SEC,
        usd_per_second=_USD_PER_VIDEO_SECOND,
        prepare=lambda s, scratch: compress_for_vlm(s, scratch / "compressed.mp4"),
        segment_suffix=".mp4",
        analyze_chunk=_analyze,
        kind="video",
    )


async def analyze_audio(
    src: Path,
    hints: AnalysisHints,
    provider: SupportsAudioAnalysis,
    *,
    video_id: str,
    duration_sec: float,
    cost_cap_usd: float,
    confirm_cost: Callable[[float, float], bool] | None = None,
    work_dir: Path | None = None,
) -> ClipPlan:
    """Analyse ``src``'s AUDIO and return a merged, absolute-time ``ClipPlan``.

    Extracts a small mono MP3 and sends it (batched at ~10 min when long) to
    ``provider.analyze_audio_clip``. Mirrors ``analyze_video`` exactly except for
    the prepare step and the provider method.
    """

    async def _analyze(segment: Path, chunk: Chunk) -> ClipPlan:
        return await provider.analyze_audio_clip(
            segment, hints, video_id=video_id, clip_duration_sec=chunk.duration_sec
        )

    return await _run_batched(
        src,
        hints,
        video_id=video_id,
        duration_sec=duration_sec,
        cost_cap_usd=cost_cap_usd,
        confirm_cost=confirm_cost,
        work_dir=work_dir,
        batch_duration_sec=_AUDIO_BATCH_DURATION_SEC,
        usd_per_second=_USD_PER_AUDIO_SECOND,
        prepare=lambda s, scratch: extract_audio_for_vlm(s, scratch / "audio.mp3"),
        segment_suffix=".mp3",
        analyze_chunk=_analyze,
        kind="audio",
    )


# ---------------------------------------------------------------------------
# Shared core
# ---------------------------------------------------------------------------


async def _run_batched(
    src: Path,
    hints: AnalysisHints,
    *,
    video_id: str,
    duration_sec: float,
    cost_cap_usd: float,
    confirm_cost: Callable[[float, float], bool] | None,
    work_dir: Path | None,
    batch_duration_sec: float,
    usd_per_second: float,
    prepare: Callable[[Path, Path], Path],
    segment_suffix: str,
    analyze_chunk: Callable[[Path, Chunk], Awaitable[ClipPlan]],
    kind: str,
) -> ClipPlan:
    """Prepare → batch → autonomous loop → merge. Payload-agnostic.

    ``prepare(src, scratch)`` writes the analysis-grade copy and returns its
    path. ``analyze_chunk(segment, chunk)`` calls the provider for one batch.
    ``confirm_cost(estimated_usd, cap_usd)`` is consulted only when the rough
    estimate exceeds ``cost_cap_usd``; ``False``/``None`` aborts with
    ``VideoAnalysisError``.
    """
    if duration_sec <= 0:
        raise VideoAnalysisError("duration_sec must be > 0")

    overlap_sec = min(max(0.0, hints.max_duration_sec / 2.0), batch_duration_sec / 2.0)
    chunks = split_into_chunks(
        duration_sec, chunk_duration_sec=batch_duration_sec, overlap_sec=overlap_sec
    )
    single_pass = not chunks
    if single_pass:
        chunks = [Chunk(index=0, start_sec=0.0, end_sec=duration_sec)]

    _gate_cost(duration_sec, cost_cap_usd, confirm_cost, usd_per_second)

    with _scratch_dir(work_dir) as scratch:
        prepared = prepare(src, scratch)
        plans: list[ClipPlan] = []
        for chunk in chunks:
            segment = _segment_for(
                chunk, prepared, scratch, single_pass=single_pass, suffix=segment_suffix
            )
            log.info(
                "%s analysis: batch %d/%d [%.1f-%.1f s]",
                kind,
                chunk.index + 1,
                len(chunks),
                chunk.start_sec,
                chunk.end_sec,
            )
            plans.append(await analyze_chunk(segment, chunk))

    merged = merge_plans(plans, chunks, video_id=video_id, duration_sec=duration_sec)
    log.info("%s analysis: %d batch(es) -> %d merged clip(s)", kind, len(chunks), len(merged.clips))
    return merged


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _gate_cost(
    duration_sec: float,
    cost_cap_usd: float,
    confirm_cost: Callable[[float, float], bool] | None,
    usd_per_second: float,
) -> None:
    """One upfront cost check; raise unless under cap or explicitly confirmed."""
    estimated = duration_sec * usd_per_second
    if estimated <= cost_cap_usd:
        return
    approved = confirm_cost(estimated, cost_cap_usd) if confirm_cost else False
    if not approved:
        raise VideoAnalysisError(
            f"estimated cost {estimated:.4f} USD exceeds cap {cost_cap_usd:.2f} USD; "
            f"raise the cap or accept the prompt"
        )


def _segment_for(
    chunk: Chunk, prepared: Path, scratch: Path, *, single_pass: bool, suffix: str
) -> Path:
    """Return the clip file for ``chunk``: the whole prepared media (single pass)
    or a stream-copied batch segment cut from it."""
    if single_pass:
        return prepared
    segment = scratch / f"batch_{chunk.index:03d}{suffix}"
    cut_clip(
        prepared,
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
    with tempfile.TemporaryDirectory(prefix="autocut_media_") as tmp:
        yield Path(tmp)
