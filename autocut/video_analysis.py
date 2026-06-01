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
# back per call via usage.include. Calibrated to measured Gemini-Flash billing
# (~$0.27 for a 20-min video = ~0.000228/s; ~$0.32 for a 57-min audio =
# ~0.000094/s), with a small margin. The old 0.0005 video estimate was ~2x too
# high, which made the two-pass cost gate (factor 2) spuriously trip the $1 cap
# on ordinary long videos now that two-pass is the default for them.
_USD_PER_VIDEO_SECOND: float = 0.00025
_USD_PER_AUDIO_SECOND: float = 0.0001

# Two-pass (coarse -> fine). Pass 1 locates candidate regions with high recall
# over the whole timeline (the model is imprecise on long footage); pass 2
# re-analyses each padded candidate in isolation, where the model gives tight,
# accurate boundaries. Only worth it on long sources — short ones are already
# precise in a single pass.
_TWO_PASS_MIN_DURATION_SEC: float = 60.0
# Padding around each coarse candidate before the fine pass: enough for the fine
# pass to see the moment's lead-in and follow-through, but not so wide that
# neighbouring candidates' windows overlap heavily and yield duplicate clips.
_TWO_PASS_PAD_SEC: float = 20.0
_TWO_PASS_MAX_CANDIDATES: int = 12
# Coarse candidate regions may be wider than a final clip (they bracket the
# moment; pass 2 trims it), so the locate pass uses a looser max duration.
_COARSE_MAX_DURATION_SEC: float = 60.0
# Cost-gate multiplier: pass 1 scans the whole source (~1x) and pass 2
# re-analyses the padded candidates (<=~1x more). 2x is a safe upper bound.
_TWO_PASS_COST_FACTOR: float = 2.0

# A provider call for one prepared media segment: (segment, hints, duration) ->
# clip-relative ClipPlan. Both passes use this; only the hints differ.
AnalyzeFn = Callable[[Path, AnalysisHints, float], Awaitable[ClipPlan]]


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
    two_pass: bool = False,
) -> ClipPlan:
    """Analyse ``src`` as VIDEO and return a merged, absolute-time ``ClipPlan``.

    Compresses the source to analysis grade and sends it (batched at ~5 min when
    long) to ``provider.analyze_video_clip``. With ``two_pass`` and a long source
    it runs the coarse→fine locate/analyze pipeline instead. See ``_run`` for the
    cost gate / scratch-dir semantics.
    """

    async def _analyze(segment: Path, h: AnalysisHints, dur: float) -> ClipPlan:
        return await provider.analyze_video_clip(
            segment, h, video_id=video_id, clip_duration_sec=dur
        )

    return await _run(
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
        analyze=_analyze,
        kind="video",
        two_pass=two_pass,
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
    two_pass: bool = False,
) -> ClipPlan:
    """Analyse ``src``'s AUDIO and return a merged, absolute-time ``ClipPlan``.

    Extracts a small mono MP3 and sends it (batched at ~10 min when long) to
    ``provider.analyze_audio_clip``. Mirrors ``analyze_video`` exactly except for
    the prepare step and the provider method.
    """

    async def _analyze(segment: Path, h: AnalysisHints, dur: float) -> ClipPlan:
        return await provider.analyze_audio_clip(
            segment, h, video_id=video_id, clip_duration_sec=dur
        )

    return await _run(
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
        analyze=_analyze,
        kind="audio",
        two_pass=two_pass,
    )


# ---------------------------------------------------------------------------
# Shared core
# ---------------------------------------------------------------------------


async def _run(
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
    analyze: AnalyzeFn,
    kind: str,
    two_pass: bool,
) -> ClipPlan:
    """Dispatch to the two-pass pipeline (long source + opt-in) or single pass."""
    if duration_sec <= 0:
        raise VideoAnalysisError("duration_sec must be > 0")
    if two_pass and duration_sec > _TWO_PASS_MIN_DURATION_SEC:
        return await _run_two_pass(
            src,
            hints,
            video_id=video_id,
            duration_sec=duration_sec,
            cost_cap_usd=cost_cap_usd,
            confirm_cost=confirm_cost,
            work_dir=work_dir,
            batch_duration_sec=batch_duration_sec,
            usd_per_second=usd_per_second,
            prepare=prepare,
            segment_suffix=segment_suffix,
            analyze=analyze,
            kind=kind,
        )
    return await _run_batched(
        src,
        hints,
        video_id=video_id,
        duration_sec=duration_sec,
        cost_cap_usd=cost_cap_usd,
        confirm_cost=confirm_cost,
        work_dir=work_dir,
        batch_duration_sec=batch_duration_sec,
        usd_per_second=usd_per_second,
        prepare=prepare,
        segment_suffix=segment_suffix,
        analyze=analyze,
        kind=kind,
    )


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
    analyze: AnalyzeFn,
    kind: str,
) -> ClipPlan:
    """Prepare → batch → autonomous loop → merge. Payload-agnostic (single pass).

    ``prepare(src, scratch)`` writes the analysis-grade copy and returns its
    path. ``analyze(segment, hints, dur)`` calls the provider for one batch.
    ``confirm_cost(estimated_usd, cap_usd)`` is consulted only when the rough
    estimate exceeds ``cost_cap_usd``; ``False``/``None`` aborts with
    ``VideoAnalysisError``.
    """
    _gate_cost(duration_sec, cost_cap_usd, confirm_cost, usd_per_second)
    with _scratch_dir(work_dir) as scratch:
        prepared = prepare(src, scratch)
        return await _batch_over_prepared(
            prepared,
            hints,
            video_id=video_id,
            duration_sec=duration_sec,
            batch_duration_sec=batch_duration_sec,
            segment_suffix=segment_suffix,
            scratch=scratch,
            analyze=analyze,
            kind=kind,
        )


async def _batch_over_prepared(
    prepared: Path,
    hints: AnalysisHints,
    *,
    video_id: str,
    duration_sec: float,
    batch_duration_sec: float,
    segment_suffix: str,
    scratch: Path,
    analyze: AnalyzeFn,
    kind: str,
) -> ClipPlan:
    """Batch an already-prepared media file, analyse each batch, merge to absolute.

    Split out from ``_run_batched`` so the two-pass locate step can batch the
    same prepared copy without re-preparing it.
    """
    overlap_sec = min(max(0.0, hints.max_duration_sec / 2.0), batch_duration_sec / 2.0)
    chunks = split_into_chunks(
        duration_sec, chunk_duration_sec=batch_duration_sec, overlap_sec=overlap_sec
    )
    single_pass = not chunks
    if single_pass:
        chunks = [Chunk(index=0, start_sec=0.0, end_sec=duration_sec)]

    plans: list[ClipPlan] = []
    ok_chunks: list[Chunk] = []
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
        try:
            plans.append(await analyze(segment, hints, chunk.duration_sec))
            ok_chunks.append(chunk)
        except Exception as exc:
            log.warning(
                "%s analysis: batch %d/%d failed, skipping it: %s",
                kind,
                chunk.index + 1,
                len(chunks),
                exc,
            )

    if not plans:
        raise VideoAnalysisError(
            f"{kind} analysis: all {len(chunks)} batch(es) failed (provider/network error)"
        )
    merged = merge_plans(plans, ok_chunks, video_id=video_id, duration_sec=duration_sec)
    log.info("%s analysis: %d batch(es) -> %d merged clip(s)", kind, len(plans), len(merged.clips))
    return merged


async def _run_two_pass(
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
    analyze: AnalyzeFn,
    kind: str,
) -> ClipPlan:
    """Coarse locate (high recall) → per-candidate fine analyse (tight cut).

    Pass 1 reuses the batch engine with a COARSE prompt to find wide candidate
    regions over the whole timeline. Pass 2 cuts each candidate (± padding) from
    the prepared copy and re-analyses it alone with the FINE prompt — the model
    is precise on a short clip. Per-candidate plans are re-based to absolute time
    and IoU-deduped via ``merge_plans``.
    """
    # Both passes are gated once, up front (pass 1 ~1x duration, pass 2 <=~1x).
    _gate_cost(duration_sec * _TWO_PASS_COST_FACTOR, cost_cap_usd, confirm_cost, usd_per_second)

    coarse_hints = hints.model_copy(
        update={
            "prompt_template": "coarse",
            "max_duration_sec": min(_COARSE_MAX_DURATION_SEC, duration_sec),
        }
    )

    with _scratch_dir(work_dir) as scratch:
        prepared = prepare(src, scratch)

        # PASS 1 — locate candidate regions (absolute time).
        coarse_plan = await _batch_over_prepared(
            prepared,
            coarse_hints,
            video_id=video_id,
            duration_sec=duration_sec,
            batch_duration_sec=batch_duration_sec,
            segment_suffix=segment_suffix,
            scratch=scratch,
            analyze=analyze,
            kind=f"{kind}-coarse",
        )
        candidates = sorted(coarse_plan.clips, key=lambda c: -c.score)[:_TWO_PASS_MAX_CANDIDATES]
        log.info(
            "%s two-pass: %d candidate region(s) located (analysing top %d)",
            kind,
            len(coarse_plan.clips),
            len(candidates),
        )
        if not candidates:
            return coarse_plan

        # PASS 2 — re-analyse each padded candidate in isolation, tight + precise.
        fine_plans: list[ClipPlan] = []
        fine_chunks: list[Chunk] = []
        for i, cand in enumerate(candidates):
            lo = max(0.0, cand.start.total_seconds() - _TWO_PASS_PAD_SEC)
            hi = min(duration_sec, cand.end.total_seconds() + _TWO_PASS_PAD_SEC)
            if hi - lo <= 0:
                continue
            segment = scratch / f"cand_{i:03d}{segment_suffix}"
            cut_clip(
                prepared,
                CutRequest(
                    start=timedelta(seconds=lo),
                    end=timedelta(seconds=hi),
                    output_path=segment,
                ),
            )
            log.info(
                "%s two-pass: fine analyse candidate %d/%d [%.1f-%.1f s]",
                kind,
                i + 1,
                len(candidates),
                lo,
                hi,
            )
            try:
                plan = await analyze(segment, hints, hi - lo)
            except Exception as exc:
                log.warning(
                    "%s two-pass: candidate %d/%d failed, skipping it: %s",
                    kind,
                    i + 1,
                    len(candidates),
                    exc,
                )
                continue
            fine_plans.append(plan)
            fine_chunks.append(Chunk(index=i, start_sec=lo, end_sec=hi))

    if not fine_plans:
        return coarse_plan

    merged = merge_plans(fine_plans, fine_chunks, video_id=video_id, duration_sec=duration_sec)
    merged = _add_cost(merged, coarse_plan.metadata.cost_usd)
    log.info(
        "%s two-pass: %d candidate(s) -> %d final clip(s)", kind, len(fine_plans), len(merged.clips)
    )
    return merged


def _add_cost(plan: ClipPlan, extra_usd: float | None) -> ClipPlan:
    """Fold the coarse pass's billed cost into the merged plan's ``cost_usd``."""
    if extra_usd is None or plan.metadata.cost_usd is None:
        return plan
    new_meta = plan.metadata.model_copy(update={"cost_usd": plan.metadata.cost_usd + extra_usd})
    return plan.model_copy(update={"metadata": new_meta})


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
