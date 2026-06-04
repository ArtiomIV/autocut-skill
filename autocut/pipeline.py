"""Pipeline orchestrator — chains every stage from video file to plan.json.

Cloud analysis path:

    probe -> _select_route -> (video | audio | keyframe) -> rank -> plan.json

``run`` is cloud-only and analysis-only: it talks to the abstract ``VLMProvider``
(openrouter), writes plan.json, and never cuts. The local/host path is no longer
here — a capable agent drives the deterministic ``probe``/``sheet``/``cut``/
``merge`` subcommands itself (see the ``autocut-run`` skill), so there is no
pause/resume sidecar anymore. ``cut --from-json`` then trims the clips 1:1.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from autocut.config import AutoCutConfig
from autocut.content import ContentProfile, profile_for
from autocut.models import (
    AnalysisHints,
    ClipPlan,
    ContentHint,
    Keyframe,
    VideoMetadata,
)
from autocut.output import write_plan_json
from autocut.scoring import RankedClip, rank_clips
from autocut.video import (
    build_sampler,
    detect_scenes,
    extract_keyframes,
    probe_video,
)
from autocut.video_analysis import COST_CAP_ENABLED, analyze_audio, analyze_video
from autocut.vlm import CostEstimate, VLMError, VLMProvider

log = logging.getLogger(__name__)

DEFAULT_KEYFRAME_SUBDIR = "keyframes"

# A confirm hook: returns True to proceed, False to abort. The CLI passes
# its interactive prompt; tests pass a constant.
ConfirmHook = Callable[[CostEstimate, float], bool]


class Route(Enum):
    """How a run delivers the analysis payload to the model (all cloud).

    Payload (keyframe vs video vs audio) over the OpenRouter transport. Keeping
    it as one small table means each new payload is one extra route plus a thin
    runner — the orchestrator does not grow an ad-hoc tree of ``if``s.
    """

    keyframe = "keyframe"  # stills path — openrouter image-only fallback
    openrouter_video = "openrouter_video"  # L2 autonomous batch loop (base64 video)
    openrouter_audio = "openrouter_audio"  # L2 autonomous batch loop (audio, talk)


# Content types whose signal is the spoken word — routed to audio when the model
# can hear (cloud only).
_TALK_HINTS: frozenset[ContentHint] = frozenset({ContentHint.talk})


def _select_route(
    content_hint: ContentHint,
    *,
    supports_video: bool,
    supports_audio: bool,
) -> Route:
    """Pick the analysis route from model capability + content type.

    Pure and side-effect free so it can be unit-tested without a live provider.
    Talk/podcast content with an audio-capable model takes the audio route,
    everything visual takes video, and a model without direct video/audio support
    falls back to keyframes.
    """
    if content_hint in _TALK_HINTS and supports_audio:
        return Route.openrouter_audio
    if supports_video:
        return Route.openrouter_video
    return Route.keyframe


class CostCapExceeded(VLMError):  # noqa: N818
    # ``Error`` suffix is conventional but here we are signalling a deliberate
    # safety gate, not an error condition. Keeping the descriptive name.
    """Raised when the estimated cost exceeds ``config.security.cost_cap_usd``
    and the confirm hook denies the run."""


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Bundle returned by ``run_analysis``: every artefact produced during the run."""

    metadata: VideoMetadata
    keyframes: list[Keyframe]
    plan: ClipPlan
    ranked: list[RankedClip]
    # ``run`` no longer cuts: it writes plan.json (ranked clips with pre/post-roll
    # baked into the timestamps) and the agent then calls ``cut --from-json``.
    plan_path: Path | None


async def run_analysis(
    video_path: str | Path,
    provider: VLMProvider,
    *,
    config: AutoCutConfig,
    hints: AnalysisHints | None = None,
    output_root: Path | None = None,
    sampling_strategy: str = "hybrid",
    write_outputs: bool = True,
    accurate_cuts: bool = False,
    confirm_cost: ConfirmHook | None = None,
    two_pass: bool = False,
    force_two_pass: bool = False,
) -> AnalysisResult:
    """Run the full cloud pipeline. Returns a populated ``AnalysisResult``.

    ``two_pass`` enables the coarse→fine locate/analyse pipeline on the direct
    openrouter media routes for long sources (a no-op on the keyframe fallback,
    which stays single-pass). ``force_two_pass`` additionally lifts the >60s
    duration gate so the refinement also runs on short sources on request.
    """
    video = Path(video_path)
    root = (output_root or config.output.base_dir).resolve()
    keyframe_dir = root / DEFAULT_KEYFRAME_SUBDIR

    effective_hints = _resolve_hints(hints, config)

    log.info("pipeline: probe %s", video)
    metadata = probe_video(video)

    # The orchestrating agent picks the editing MODE up front (from the user's
    # request crossed with the kind of video) and passes it as ``content_hint``
    # / ``query`` — so ``run`` itself does NO content detection. If the agent
    # leaves the mode unset, default to HYBRID (the safe, general profile)
    # rather than spend a VLM call guessing. Explicit auto-classification stays
    # available as the standalone ``autocut detect`` subcommand.
    if effective_hints.content_hint == ContentHint.auto:
        log.info("pipeline: no explicit mode supplied; defaulting to HYBRID")
        effective_hints = effective_hints.model_copy(update={"content_hint": ContentHint.hybrid})

    profile = profile_for(effective_hints.content_hint)
    effective_hints = _apply_profile_to_hints(effective_hints, profile)
    effective_sampling_strategy = (
        sampling_strategy if sampling_strategy != "hybrid" else profile.sampling_strategy
    )

    # Capability gate: pick the payload route once, then dispatch to a thin runner.
    # Video/audio-capable models skip keyframe sampling — the L2 autonomous batch
    # loop ingests the media directly (video, or audio for talk). Everything else
    # takes the keyframe path below. ``supports_audio`` is only queried for talk
    # content (a network call) so non-talk runs don't pay for it.
    supports_video = await provider.supports_video()
    is_talk = effective_hints.content_hint in _TALK_HINTS
    supports_audio = bool(is_talk and await provider.supports_audio())
    route = _select_route(
        effective_hints.content_hint,
        supports_video=supports_video,
        supports_audio=supports_audio,
    )
    if route is Route.openrouter_video:
        return await _run_media_analysis(
            video,
            provider,
            metadata=metadata,
            effective_hints=effective_hints,
            profile=profile,
            config=config,
            root=root,
            accurate_cuts=accurate_cuts,
            write_outputs=write_outputs,
            confirm_cost=confirm_cost,
            kind="video",
            two_pass=two_pass,
            force_two_pass=force_two_pass,
        )
    if route is Route.openrouter_audio:
        return await _run_media_analysis(
            video,
            provider,
            metadata=metadata,
            effective_hints=effective_hints,
            profile=profile,
            config=config,
            root=root,
            accurate_cuts=accurate_cuts,
            write_outputs=write_outputs,
            confirm_cost=confirm_cost,
            kind="audio",
            two_pass=two_pass,
            force_two_pass=force_two_pass,
        )

    log.info("pipeline: scene detect (threshold=%s)", config.advanced.scene_threshold)
    scenes = detect_scenes(video, threshold=config.advanced.scene_threshold)

    log.info(
        "pipeline: sampling (strategy=%s, profile=%s, %d scenes)",
        effective_sampling_strategy,
        profile.name,
        len(scenes),
    )
    specs = build_sampler(
        effective_sampling_strategy,  # type: ignore[arg-type]
        scenes,
        metadata.duration_sec,
        per_scene=config.advanced.keyframes_per_scene,
        min_keyframes=profile.min_keyframes,
    )
    if not specs:
        raise ValueError(
            f"sampler produced 0 keyframes for {video} (duration={metadata.duration_sec}s)"
        )

    log.info("pipeline: extracting %d keyframes -> %s", len(specs), keyframe_dir)
    keyframes = extract_keyframes(
        video,
        specs,
        keyframe_dir,
        long_edge_px=config.advanced.keyframe_resolution,
    )

    # Upfront cost estimate, computed BEFORE the API call when we know exactly how
    # many images we're about to send. The cap is enforced only when
    # ``COST_CAP_ENABLED`` (disabled for now — no spending limit); otherwise this
    # only logs the estimate, which is also recorded in plan.json below.
    estimate = provider.estimate_cost(len(keyframes))
    cap = config.security.cost_cap_usd
    log.info(
        "pipeline: cost estimate %.4f USD (cap %.2f, free=%s, enforced=%s)",
        estimate.estimated_total_usd,
        cap,
        estimate.is_free,
        COST_CAP_ENABLED,
    )
    if COST_CAP_ENABLED and not estimate.is_free and estimate.estimated_total_usd > cap:
        proceed = confirm_cost(estimate, cap) if confirm_cost else False
        if not proceed:
            raise CostCapExceeded(
                f"estimated cost {estimate.estimated_total_usd:.4f} USD exceeds "
                f"cap {cap:.2f} USD; rerun with a higher cost_cap_usd or accept the prompt"
            )

    video_id = video.stem or "video"
    log.info(
        "pipeline: handing %d keyframes to provider=%s model=%s",
        len(keyframes),
        provider.name,
        getattr(provider, "model", "?"),
    )
    plan = await provider.analyze(
        keyframes,
        effective_hints,
        video_id=video_id,
        duration_sec=metadata.duration_sec,
    )
    log.info("pipeline: provider returned %d clip(s)", len(plan.clips))

    return complete_from_plan(
        plan,
        video=video,
        metadata=metadata,
        config=config,
        output_root=root,
        n_scenes=len(scenes),
        n_keyframes=len(keyframes),
        sampling_strategy=effective_sampling_strategy,
        accurate_cuts=accurate_cuts,
        write_outputs=write_outputs,
        cost_estimate_usd=estimate.estimated_total_usd,
        upfront_cost_estimate_usd=estimate.estimated_total_usd,
        keyframes=keyframes,
        profile=profile,
    )


async def _run_media_analysis(
    video: Path,
    provider: VLMProvider,
    *,
    metadata: VideoMetadata,
    effective_hints: AnalysisHints,
    profile: ContentProfile,
    config: AutoCutConfig,
    root: Path,
    accurate_cuts: bool,
    write_outputs: bool,
    confirm_cost: ConfirmHook | None,
    kind: str,
    two_pass: bool = False,
    force_two_pass: bool = False,
) -> AnalysisResult:
    """Direct-payload path (openrouter): prepare + autonomous batch loop via L2.

    ``kind`` selects the payload: ``"video"`` compresses the clip and calls
    ``analyze_video``; ``"audio"`` extracts a mono MP3 and calls
    ``analyze_audio`` (talk content). No keyframe sampling — the provider ingests
    the media directly, so the run completes in one call. Reuses
    ``complete_from_plan`` for ranking + plan.json (with zero scenes/keyframes),
    referencing the ORIGINAL video either way.
    """

    def _confirm(estimated_usd: float, cap_usd: float) -> bool:
        if confirm_cost is None:
            return False
        estimate = CostEstimate(
            provider=provider.name,
            model=getattr(provider, "model", "?"),
            n_input_images=0,
            estimated_input_tokens=0,
            estimated_output_tokens=0,
            estimated_total_usd=estimated_usd,
        )
        return confirm_cost(estimate, cap_usd)

    video_id = video.stem or "video"
    log.info(
        "pipeline: %s path via provider=%s model=%s",
        kind,
        provider.name,
        getattr(provider, "model", "?"),
    )
    # Branch explicitly (not a ternary on the function) so each call type-checks
    # against its own provider protocol — analyze_audio/analyze_video take
    # structurally different providers.
    if kind == "audio":
        plan = await analyze_audio(
            video,
            effective_hints,
            provider,
            video_id=video_id,
            duration_sec=metadata.duration_sec,
            cost_cap_usd=config.security.cost_cap_usd,
            confirm_cost=_confirm,
            two_pass=two_pass,
            force_two_pass=force_two_pass,
        )
    else:
        plan = await analyze_video(
            video,
            effective_hints,
            provider,
            video_id=video_id,
            duration_sec=metadata.duration_sec,
            cost_cap_usd=config.security.cost_cap_usd,
            confirm_cost=_confirm,
            two_pass=two_pass,
            force_two_pass=force_two_pass,
        )
    log.info(
        "pipeline: %s analysis returned %d clip(s), real cost %s USD",
        kind,
        len(plan.clips),
        plan.metadata.cost_usd,
    )
    return complete_from_plan(
        plan,
        video=video,
        metadata=metadata,
        config=config,
        output_root=root,
        n_scenes=0,
        n_keyframes=0,
        sampling_strategy=kind,
        accurate_cuts=accurate_cuts,
        write_outputs=write_outputs,
        # real billed cost from usage.include (summed across batches), not an estimate
        cost_estimate_usd=plan.metadata.cost_usd or 0.0,
        # rough upfront estimate (both passes) attached by the analysis engine
        upfront_cost_estimate_usd=plan.metadata.upfront_cost_estimate_usd,
        keyframes=None,
        profile=profile,
    )


def complete_from_plan(
    plan: ClipPlan,
    *,
    video: Path,
    metadata: VideoMetadata,
    config: AutoCutConfig,
    output_root: Path,
    n_scenes: int,
    n_keyframes: int,
    sampling_strategy: str,
    accurate_cuts: bool,
    write_outputs: bool,
    cost_estimate_usd: float = 0.0,
    upfront_cost_estimate_usd: float | None = None,
    keyframes: list[Keyframe] | None = None,
    profile: ContentProfile | None = None,
) -> AnalysisResult:
    """Run ranking + write plan.json given an already-validated ``ClipPlan``.

    ``profile`` is forwarded so per-content-type padding (currently 0 everywhere)
    and the per-mode keep threshold reach the plan. ``accurate_cuts`` is accepted
    for signature compatibility but unused (cutting moved out to ``cut``).
    """
    # A mode can raise the keep threshold above the global default (highlights
    # uses 7). When it does, overlay it on the scoring config for this run so
    # weak clips are dropped — an empty result is a valid, honest outcome.
    scoring = config.scoring
    if profile is not None and profile.min_score is not None:
        scoring = scoring.model_copy(update={"min_score": profile.min_score})

    ranked = rank_clips(plan, scoring)
    log.info("pipeline: %d clip(s) survived ranking (min_score=%d)", len(ranked), scoring.min_score)

    # ``run`` is analysis-only: it writes plan.json with the pre/post-roll (chosen
    # by the active profile/hint) baked into the timestamps, and does NOT cut any
    # MP4. The orchestrating agent reviews/edits the plan, then calls
    # ``cut --from-json`` to trim the clips 1:1 (no further padding).
    pre_roll = profile.pre_roll_sec if profile else 0.0
    post_roll = profile.post_roll_sec if profile else 0.0

    plan_path: Path | None = None
    if write_outputs and ranked:
        plan_path = write_plan_json(
            output_dir=output_root,
            video_path=video,
            metadata=metadata,
            ranked=ranked,
            pre_roll_sec=pre_roll,
            post_roll_sec=post_roll,
            extra={
                "vlm": {
                    "provider": plan.metadata.vlm_provider,
                    "model": plan.metadata.vlm_model,
                    "prompt_version": plan.metadata.prompt_version,
                    "analysis_time_sec": plan.metadata.analysis_time_sec,
                    "cost_estimate_usd": cost_estimate_usd,
                    "cost_upfront_estimate_usd": upfront_cost_estimate_usd,
                },
                "sampling": {
                    "strategy": sampling_strategy,
                    "n_keyframes": n_keyframes,
                    "n_scenes": n_scenes,
                },
            },
        )

    return AnalysisResult(
        metadata=metadata,
        keyframes=keyframes or [],
        plan=plan,
        ranked=ranked,
        plan_path=plan_path,
    )


def _resolve_hints(
    hints: AnalysisHints | None,
    config: AutoCutConfig,
) -> AnalysisHints:
    """Use caller-provided hints if any; otherwise build them from config defaults."""
    if hints is not None:
        return hints
    defaults = config.content_defaults
    try:
        content_hint = ContentHint(defaults.hint)
    except ValueError:
        content_hint = ContentHint.auto
    return AnalysisHints(
        content_hint=content_hint,
        goal=defaults.goal,
        language=defaults.language,
    )


def _apply_profile_to_hints(hints: AnalysisHints, profile: ContentProfile) -> AnalysisHints:
    """Overlay profile-derived fields (duration bounds, prompt template) on hints.

    The caller's ``content_hint``, ``goal``, ``language``, and
    ``target_clip_count`` are preserved — those reflect intent. Only the
    fields the profile owns are overwritten.
    """
    return hints.model_copy(
        update={
            "min_duration_sec": profile.min_duration_sec,
            "max_duration_sec": profile.max_duration_sec,
            "prompt_template": profile.prompt_template,
        }
    )
