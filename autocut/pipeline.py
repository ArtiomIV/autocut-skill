"""Pipeline orchestrator — chains every stage from video file to written clips.

Full happy path:

    probe -> scene_detect -> frame_sampler -> extract_keyframes
          -> (cost cap check) -> provider.analyze
          -> rank_clips -> dispatch_outputs

The orchestrator is provider-agnostic: it only talks to the abstract
``VLMProvider`` interface. The ``host_agent`` pause flow surfaces here as
``HostAgentPauseRequested`` propagating untouched — the CLI catches it
and tells the user how to resume.

Resume flow: before calling ``provider.analyze`` the orchestrator writes
``RESUME_STATE_FILENAME`` next to the host-agent request file with every
parameter ``complete_from_plan`` needs to finish the pipeline. The CLI's
``resume`` subcommand reads it, calls ``complete_from_plan`` and the user
gets the same output writers and manifest they would have got from a
single-shot run.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from autocut.config import AutoCutConfig
from autocut.content import ContentProfile, profile_for
from autocut.host_analysis import (
    HOST_TWO_PASS_MIN_DURATION_SEC,
    build_coarse_sheets,
    build_fine_sheets,
    dedup_plan,
    empty_plan,
    select_fine_windows,
)
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
from autocut.vlm.host_agent import HostAgentProvider
from autocut.vlm.prompts import PROMPT_VERSION

log = logging.getLogger(__name__)

DEFAULT_KEYFRAME_SUBDIR = "keyframes"
RESUME_STATE_FILENAME = ".autocut_resume.json"

# Sub-dirs the host image two-pass renders its contact sheets into (under the
# work dir, so they persist across the pause/resume).
HOST_COARSE_SHEETS_SUBDIR = "coarse_sheets"
HOST_FINE_SHEETS_SUBDIR = "fine_sheets"
# Coarse candidate regions bracket a moment (the fine pass trims it), so the
# locate pass uses a looser max duration than a final clip.
_HOST_COARSE_MAX_DURATION_SEC: float = 60.0

# A confirm hook: returns True to proceed, False to abort. The CLI passes
# its interactive prompt; tests pass a constant.
ConfirmHook = Callable[[CostEstimate, float], bool]


class Route(Enum):
    """How a run delivers the analysis payload to the model.

    The selection is the payload (keyframe vs video) crossed with the
    transport (host pause/resume vs OpenRouter API). Keeping it as one small
    table here means each new payload (e.g. audio) is one extra route plus a
    thin runner — the orchestrator does not grow an ad-hoc tree of ``if``s.
    """

    keyframe = "keyframe"  # stills path — openrouter images fallback
    openrouter_video = "openrouter_video"  # L2 autonomous batch loop (base64 video)
    openrouter_audio = "openrouter_audio"  # L2 autonomous batch loop (audio, talk)
    host_image = "host_image"  # image-only contact-sheet two-pass via host pause/resume


# Content types whose signal is the spoken word — routed to audio when the model
# can hear and the transport is cloud (the host cannot listen to audio yet).
_TALK_HINTS: frozenset[ContentHint] = frozenset({ContentHint.talk})


def _select_route(
    provider_name: str,
    content_hint: ContentHint,
    *,
    supports_video: bool,
    supports_audio: bool,
) -> Route:
    """Pick the analysis route from provider capability + content type.

    Pure and side-effect free so it can be unit-tested without a live provider.
    Capabilities are the already-resolved booleans (``await
    provider.supports_video()`` / ``supports_audio()``). The host transport
    cannot hear audio (no local transcription yet), so audio is cloud-only;
    talk/podcast content with an audio-capable cloud model takes the audio
    route, everything visual takes video, and anything without direct support
    falls back to keyframes.
    """
    if provider_name == "host":
        # Host is image-only: the contact-sheet two-pass, never video or audio.
        return Route.host_image
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


class ResumeStateError(VLMError):
    """Raised when the resume state sidecar is missing, malformed, or stale."""


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
    """Run the full pipeline. Returns a populated ``AnalysisResult``.

    ``two_pass`` enables the coarse→fine locate/analyse pipeline on the direct
    openrouter media routes for long sources (a no-op on the host and keyframe
    routes, which stay single-pass). ``force_two_pass`` additionally lifts the
    >60s duration gate so the refinement also runs on short sources on request.
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

    # Capability gate: pick the payload x transport route once, then dispatch to
    # a thin runner. Video/audio-capable providers skip keyframe sampling —
    # openrouter via the L2 autonomous batch loop (video, or audio for talk), the
    # host via a single-pass compressed-MP4 pause/resume. Everything else takes
    # the keyframe path below. ``supports_audio`` is only queried for talk
    # content (a network call) so non-talk runs don't pay for it.
    supports_video = await provider.supports_video()
    is_talk = effective_hints.content_hint in _TALK_HINTS
    supports_audio = bool(is_talk and await provider.supports_audio())
    route = _select_route(
        provider.name,
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
    if route is Route.host_image:
        return await _run_host_image_analysis(
            video,
            provider,
            metadata=metadata,
            effective_hints=effective_hints,
            profile=profile,
            config=config,
            root=root,
            accurate_cuts=accurate_cuts,
            write_outputs=write_outputs,
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

    # Persist enough state that ``autocut resume`` can finish the pipeline
    # if the provider pauses (host-agent flow). Cleared on success below.
    _write_resume_state(
        root,
        video=video,
        metadata=metadata,
        n_scenes=len(scenes),
        n_keyframes=len(keyframes),
        sampling_strategy=sampling_strategy,
        accurate_cuts=accurate_cuts,
        write_outputs=write_outputs,
        content_hint=effective_hints.content_hint,
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

    result = complete_from_plan(
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
    _clear_resume_state(root)
    return result


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
    ``analyze_audio`` (talk content). No keyframe sampling and no pause/resume —
    the provider ingests the media directly, so the run completes in one call.
    Reuses ``complete_from_plan`` for ranking + output dispatch (with zero
    scenes/keyframes), cutting the highlights from the ORIGINAL video either way.
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
    result = complete_from_plan(
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
    _clear_resume_state(root)
    return result


async def _run_host_image_analysis(
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
) -> AnalysisResult:
    """Host image-only two-pass — render contact sheets, hand them over, pause.

    No video, no audio, no DSP: the whole analysis is timestamped contact sheets
    the agent reads from disk. For sources over 60s this is the COARSE pass
    (sparse 1f/3s grids over the whole timeline → wide candidate regions); the
    fine pass follows on ``autocut resume``. For short sources it goes straight to
    a single FINE pass (dense 2-FPS grids of the whole clip). Either way the
    provider raises ``HostAgentPauseRequested`` and the resume sidecar (with the
    phase) lets ``autocut resume`` continue. Cutting always uses the ORIGINAL
    video later, via ``cut --from-json``.
    """
    del config, profile  # ranking config/profile are applied on resume, not on the pausing pass
    video_id = video.stem or "video"
    duration = metadata.duration_sec
    long_source = duration > HOST_TWO_PASS_MIN_DURATION_SEC

    if long_source:
        log.info("pipeline: host image two-pass — building COARSE sheets for %s", video)
        sheets, frame_times = build_coarse_sheets(video, root / HOST_COARSE_SHEETS_SUBDIR, duration)
        pass_hints = effective_hints.model_copy(
            update={
                "prompt_template": "coarse",
                "max_duration_sec": min(_HOST_COARSE_MAX_DURATION_SEC, duration),
            }
        )
        phase = "coarse"
    else:
        log.info("pipeline: host image single FINE pass (short source) for %s", video)
        sheets, frame_times = build_fine_sheets(
            video, root / HOST_FINE_SHEETS_SUBDIR, [(0.0, duration)]
        )
        pass_hints = effective_hints
        phase = "fine"

    _write_resume_state(
        root,
        video=video,
        metadata=metadata,
        n_scenes=0,
        n_keyframes=len(frame_times),
        sampling_strategy="host-image",
        accurate_cuts=accurate_cuts,
        write_outputs=write_outputs,
        content_hint=effective_hints.content_hint,
        phase=phase,
        query=effective_hints.query,
        goal=effective_hints.goal,
        language=effective_hints.language,
    )

    # Raises HostAgentPauseRequested (propagates to the CLI), which resumes via
    # ``resume_host_analysis``. The line below is unreachable for the pausing
    # host but kept defensive for a future non-pausing implementation.
    await provider.analyze_contact_sheets(
        sheets, pass_hints, video_id=video_id, duration_sec=duration, frame_times=frame_times
    )
    raise VLMError("host image pass did not pause as expected")  # pragma: no cover


def resume_host_analysis(base: Path, config: AutoCutConfig) -> AnalysisResult:
    """Continue a paused host image two-pass after the agent wrote the response.

    Reads the resume sidecar to learn which phase paused:

    * ``coarse`` → parse the candidate regions, build the dense FINE sheets for
      them, rewrite the sidecar as ``fine`` and PAUSE AGAIN (raises
      ``HostAgentPauseRequested`` for the CLI to surface).
    * ``fine`` → parse the absolute-time clips, dedup, rank and write plan.json.

    Synchronous: the second-pause provider call is driven via ``asyncio.run`` so
    the pause exception propagates out to the CLI like the first pause did.
    """
    import asyncio

    state = load_resume_state(base)
    phase = str(state.get("phase", "fine"))
    provider = HostAgentProvider(work_dir=base, agent_hint=config.vlm.model)
    plan = provider.resume_from_disk()

    video = Path(str(state["video_path"]))
    metadata = VideoMetadata.model_validate(state["video_metadata"])
    try:
        hint = ContentHint(str(state.get("content_hint", ContentHint.hybrid.value)))
    except ValueError:
        hint = ContentHint.hybrid
    profile = profile_for(hint)
    duration = metadata.duration_sec

    if phase == "coarse":
        windows = select_fine_windows(plan, duration)
        log.info("pipeline: host resume (coarse) -> %d candidate region(s)", len(plan.clips))
        if not windows:
            # Coarse found nothing worth refining — finish with an empty plan
            # (a valid, honest outcome) rather than pausing again.
            return _complete_host(
                empty_plan(
                    video.stem or "video",
                    duration,
                    agent_hint=config.vlm.model,
                    prompt_version=PROMPT_VERSION,
                ),
                base,
                video=video,
                metadata=metadata,
                config=config,
                state=state,
                profile=profile,
            )
        sheets, frame_times = build_fine_sheets(video, base / HOST_FINE_SHEETS_SUBDIR, windows)
        fine_hints = _apply_profile_to_hints(
            AnalysisHints(
                content_hint=hint,
                goal=str(state.get("goal", "")),
                language=str(state.get("language", "en")),
                query=_state_query(state),
            ),
            profile,
        )
        _write_resume_state(
            base,
            video=video,
            metadata=metadata,
            n_scenes=0,
            n_keyframes=len(frame_times),
            sampling_strategy="host-image",
            accurate_cuts=bool(state.get("accurate_cuts", False)),
            write_outputs=bool(state.get("write_outputs", True)),
            content_hint=hint,
            phase="fine",
            query=_state_query(state),
            goal=str(state.get("goal", "")),
            language=str(state.get("language", "en")),
        )
        # Raises HostAgentPauseRequested → propagates out of asyncio.run to the CLI.
        asyncio.run(
            provider.analyze_contact_sheets(
                sheets,
                fine_hints,
                video_id=video.stem or "video",
                duration_sec=duration,
                frame_times=frame_times,
            )
        )
        raise VLMError("host fine pass did not pause as expected")  # pragma: no cover

    # phase == "fine": the agent returned absolute-time clips; dedup + complete.
    return _complete_host(
        dedup_plan(plan),
        base,
        video=video,
        metadata=metadata,
        config=config,
        state=state,
        profile=profile,
    )


def _complete_host(
    plan: ClipPlan,
    base: Path,
    *,
    video: Path,
    metadata: VideoMetadata,
    config: AutoCutConfig,
    state: dict[str, object],
    profile: ContentProfile,
) -> AnalysisResult:
    """Rank + write plan.json for a finished host run, then clear the sidecar."""
    result = complete_from_plan(
        plan,
        video=video,
        metadata=metadata,
        config=config,
        output_root=base,
        n_scenes=0,
        n_keyframes=_state_int(state, "n_keyframes"),
        sampling_strategy=str(state.get("sampling_strategy", "host-image")),
        accurate_cuts=bool(state.get("accurate_cuts", False)),
        write_outputs=bool(state.get("write_outputs", True)),
        keyframes=None,
        profile=profile,
    )
    _clear_resume_state(base)
    return result


def _state_int(state: dict[str, object], key: str) -> int:
    """Read an int counter from the resume-state dict, defaulting to 0."""
    value = state.get(key, 0)
    return int(value) if isinstance(value, int | float) else 0


def _state_query(state: dict[str, object]) -> str | None:
    """Read the persisted query (None when absent/empty)."""
    q = state.get("query")
    return q if isinstance(q, str) and q.strip() else None


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
    """Run ranking + dispatch given an already-validated ``ClipPlan``.

    Used both by ``run_analysis`` after a successful ``provider.analyze``
    call, and by the CLI ``resume`` subcommand after the host agent has
    written ``VLM_RESPONSE.json``. ``profile`` is forwarded to the dispatcher
    so per-content-type padding (currently 0 everywhere) reaches the cutter.
    """
    # A mode can raise the keep threshold above the global default (highlights
    # uses 7). When it does, overlay it on the scoring config for this run so
    # weak clips are dropped — an empty result is a valid, honest outcome.
    scoring = config.scoring
    if profile is not None and profile.min_score is not None:
        scoring = scoring.model_copy(update={"min_score": profile.min_score})

    ranked = rank_clips(plan, scoring)
    log.info("pipeline: %d clip(s) survived ranking (min_score=%d)", len(ranked), scoring.min_score)

    # ``run`` is analysis-only now: it writes plan.json with the pre/post-roll
    # (chosen by the active profile/hint) baked into the timestamps, and does NOT
    # cut any MP4. The orchestrating agent reviews/edits the plan, then calls
    # ``cut --from-json`` to trim the clips 1:1 (no further padding). ``accurate_cuts``
    # is accepted for signature compatibility but unused here (cutting moved out).
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


# ---------------------------------------------------------------------------
# Resume state sidecar
# ---------------------------------------------------------------------------


def _write_resume_state(
    output_root: Path,
    *,
    video: Path,
    metadata: VideoMetadata,
    n_scenes: int,
    n_keyframes: int,
    sampling_strategy: str,
    accurate_cuts: bool,
    write_outputs: bool,
    content_hint: ContentHint,
    phase: str = "fine",
    query: str | None = None,
    goal: str = "",
    language: str = "en",
) -> Path:
    """Persist the parameters needed to finish (or advance) a paused run later.

    ``content_hint`` is persisted so the resumed analysis rebuilds the same
    profile and applies the same per-mode keep threshold (e.g. highlights=7)
    that the live path would have. ``phase`` (``coarse``/``fine``) drives the
    host image two-pass resume: a ``coarse`` resume builds the fine sheets and
    pauses again, a ``fine`` resume completes. ``query``/``goal``/``language``
    are persisted so the fine pass rebuilds the same prompt as the coarse one.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    state = {
        "video_path": str(video.resolve()),
        "video_metadata": json.loads(metadata.model_dump_json()),
        "output_root": str(output_root.resolve()),
        "sampling_strategy": sampling_strategy,
        "accurate_cuts": accurate_cuts,
        "write_outputs": write_outputs,
        "n_scenes": n_scenes,
        "n_keyframes": n_keyframes,
        "content_hint": content_hint.value,
        "phase": phase,
        "query": query,
        "goal": goal,
        "language": language,
    }
    path = output_root / RESUME_STATE_FILENAME
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return path


def _clear_resume_state(output_root: Path) -> None:
    """Best-effort delete of the sidecar after a successful run."""
    path = output_root / RESUME_STATE_FILENAME
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        log.debug("could not clear resume state %s: %s", path, exc)


def load_resume_state(output_root: Path) -> dict[str, object]:
    """Load and lightly validate the resume sidecar. Raises ``ResumeStateError`` on issues."""
    path = output_root / RESUME_STATE_FILENAME
    if not path.is_file():
        raise ResumeStateError(
            f"resume state file not found: {path} (no paused run to resume in this output dir)"
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResumeStateError(f"failed to read resume state: {exc}") from exc
    if not isinstance(data, dict):
        raise ResumeStateError("resume state top-level value is not an object")
    required = {"video_path", "video_metadata", "sampling_strategy", "accurate_cuts"}
    missing = required - data.keys()
    if missing:
        raise ResumeStateError(f"resume state missing required keys: {sorted(missing)}")
    return data


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
