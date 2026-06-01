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
from datetime import timedelta
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
from autocut.output import DispatchResult, dispatch_outputs
from autocut.scoring import RankedClip, rank_clips
from autocut.video import (
    build_sampler,
    compute_audio_profile,
    compute_motion_profile,
    detect_scenes,
    extract_keyframes,
    find_hot_windows,
    probe_video,
)
from autocut.video.compress import compress_for_vlm
from autocut.video.refine import refine_start_to_impact
from autocut.video_analysis import analyze_audio, analyze_video
from autocut.vlm import CostEstimate, VLMError, VLMProvider

log = logging.getLogger(__name__)

DEFAULT_KEYFRAME_SUBDIR = "keyframes"
RESUME_STATE_FILENAME = ".autocut_resume.json"

# A confirm hook: returns True to proceed, False to abort. The CLI passes
# its interactive prompt; tests pass a constant.
ConfirmHook = Callable[[CostEstimate, float], bool]

# Name of the compressed analysis copy the host-video path leaves in the work
# dir for the agent to watch (persisted across the pause/resume, unlike the
# L2 engine's self-cleaning temp dir).
HOST_VIDEO_FILENAME = "compressed.mp4"


class Route(Enum):
    """How a run delivers the analysis payload to the model.

    The selection is the payload (keyframe vs video) crossed with the
    transport (host pause/resume vs OpenRouter API). Keeping it as one small
    table here means each new payload (e.g. audio) is one extra route plus a
    thin runner — the orchestrator does not grow an ad-hoc tree of ``if``s.
    """

    keyframe = "keyframe"  # stills path — host pause/resume OR openrouter images
    openrouter_video = "openrouter_video"  # L2 autonomous batch loop (base64 video)
    openrouter_audio = "openrouter_audio"  # L2 autonomous batch loop (audio, talk)
    host_video = "host_video"  # single-pass compressed MP4 via host pause/resume


# Content types whose signal is the spoken word — routed to audio when the model
# can hear and the transport is cloud (the host cannot listen to audio yet).
_TALK_HINTS: frozenset[ContentHint] = frozenset({ContentHint.talk})

# Two-stage impact-snap boundary refine (autocut.video.refine). DISABLED
# 2026-05-30: the v1 heuristic (biggest peak + walk-back) snaps imperfectly and,
# on short clips that the model already placed well, it nudges good boundaries
# without a measurable gain — so it muddies the runs. Re-enable once the v2 tuning
# lands (burst-start + silence guard, audio as co-driver) AND the manifest records
# the pre-refine start so the move is auditable.
_REFINE_BOUNDARIES_ENABLED: bool = False


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
        # Host has no audio path yet (cannot listen); video or stills only.
        return Route.host_video if supports_video else Route.keyframe
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
    dispatch: DispatchResult | None


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
) -> AnalysisResult:
    """Run the full pipeline. Returns a populated ``AnalysisResult``.

    ``two_pass`` enables the coarse→fine locate/analyse pipeline on the direct
    openrouter media routes for long sources (a no-op on the host and keyframe
    routes, which stay single-pass).
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

    # A custom query hunts a SPECIFIC described moment, which may sit in a calm,
    # low-motion stretch (e.g. "when the ring girl enters"). The motion sampler
    # bakes in "interesting == high motion/sound" and would discard such a target
    # before the model ever sees it. So when a query is present we drop the
    # motion pre-filter and show the model the whole timeline (uniform-dense on
    # the keyframe path); the query itself is the smart filter. Transport routing
    # (video/audio/keyframe) is unchanged — it stays capability-driven.
    if effective_hints.query and effective_sampling_strategy == "motion":
        log.info("pipeline: query present — disabling motion pre-filter (uniform sampling)")
        effective_sampling_strategy = "uniform"

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
        )
    if route is Route.host_video:
        return await _run_host_video_analysis(
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

    hot_windows = None
    if effective_sampling_strategy == "motion":
        log.info("pipeline: computing motion + audio profiles for motion sampler")
        motion_profile = compute_motion_profile(video)
        audio_profile = compute_audio_profile(video)
        hot_windows = find_hot_windows(
            motion_profile,
            audio_profile,
            video_duration_sec=metadata.duration_sec,
        )
        log.info(
            "pipeline: motion sampler found %d hot window(s) from %d motion + %d audio samples",
            len(hot_windows),
            len(motion_profile),
            len(audio_profile),
        )
        # Reaching here means the keyframe (stills) path: the video branch
        # returned earlier. A stills VLM cannot perceive motion, so surface the
        # hot windows in the prompt to tell it where to look — for the host
        # agent and for any openrouter keyframe fallback alike.
        if hot_windows:
            effective_hints = effective_hints.model_copy(
                update={
                    "motion_windows_sec": [(w.start_sec, w.end_sec) for w in hot_windows],
                }
            )

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
        hot_windows=hot_windows,
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

    # Cost cap check happens BEFORE the API call, when we know exactly how
    # many images we're about to send.
    estimate = provider.estimate_cost(len(keyframes))
    cap = config.security.cost_cap_usd
    log.info(
        "pipeline: cost estimate %.4f USD (cap %.2f, free=%s)",
        estimate.estimated_total_usd,
        cap,
        estimate.is_free,
    )
    if not estimate.is_free and estimate.estimated_total_usd > cap:
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
        )
    log.info(
        "pipeline: %s analysis returned %d clip(s), real cost %s USD",
        kind,
        len(plan.clips),
        plan.metadata.cost_usd,
    )
    if _REFINE_BOUNDARIES_ENABLED and _should_refine_boundaries(kind, profile):
        plan = _refine_action_boundaries(plan, video=video, duration_sec=metadata.duration_sec)
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
        keyframes=None,
        profile=profile,
    )
    _clear_resume_state(root)
    return result


def _should_refine_boundaries(kind: str, profile: ContentProfile) -> bool:
    """Refine clip starts only for the sport/action profile on the video route.

    Sport moments are physical impacts (a punch, a knockdown) whose precise
    instant the ~1fps video model misses. The refiner is self-gating, so even
    here a non-impact clip (a ring entrance) is left untouched — but we only
    *attempt* it for sport, where impacts are expected. Talk/audio has no impact
    signal (word-boundary refine via Whisper is a separate, future path).
    """
    return kind == "video" and profile.name == "highlights"


def _refine_action_boundaries(plan: ClipPlan, *, video: Path, duration_sec: float) -> ClipPlan:
    """Return ``plan`` with each clip's START snapped to the real impact instant.

    Best-effort and self-gating: a clip whose window has no dominant impact
    spike is returned unchanged. Only the ``start`` moves (always earlier);
    ``end`` and every other field are preserved.
    """
    refined = []
    moved = 0
    for clip in plan.clips:
        result = refine_start_to_impact(
            video,
            clip.start.total_seconds(),
            clip.end.total_seconds(),
            duration_sec,
        )
        if result.moved:
            moved += 1
            refined.append(
                clip.model_copy(update={"start": timedelta(seconds=result.refined_start_sec)})
            )
        else:
            refined.append(clip)
    log.info(
        "pipeline: boundary refine snapped %d/%d clip start(s) to impact", moved, len(plan.clips)
    )
    return plan.model_copy(update={"clips": refined})


async def _run_host_video_analysis(
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
    """Host video-input path: compress once, hand the MP4 to the agent, pause.

    Single-pass by design. Unlike the OpenRouter path there is no base64 size
    ceiling (the agent reads the file from disk), so the whole compressed video
    is sent in one pause/resume — no batch loop, no L2 engine. The host's
    ``analyze_video_clip`` writes ``VLM_REQUEST.md`` and raises
    ``HostAgentPauseRequested``; the resume state written here lets
    ``autocut resume`` finish via ``complete_from_plan`` after the agent fills
    in ``VLM_RESPONSE.json``. Cutting still uses the ORIGINAL video, never the
    compressed analysis copy.
    """
    log.info("pipeline: host video path — compressing %s for the agent", video)
    compressed = compress_for_vlm(video, root / HOST_VIDEO_FILENAME)

    # Persist before the pause so resume can complete (strategy="video",
    # zero keyframes — the agent watched the clip, not stills).
    _write_resume_state(
        root,
        video=video,
        metadata=metadata,
        n_scenes=0,
        n_keyframes=0,
        sampling_strategy="video",
        accurate_cuts=accurate_cuts,
        write_outputs=write_outputs,
        content_hint=effective_hints.content_hint,
    )

    video_id = video.stem or "video"
    # Raises HostAgentPauseRequested (propagates to the CLI). The whole video is
    # one clip, so clip-relative timestamps equal absolute source time. The
    # return path below is unreachable for today's pausing host but kept so a
    # future non-pausing host implementation completes cleanly.
    plan = await provider.analyze_video_clip(
        compressed,
        effective_hints,
        video_id=video_id,
        clip_duration_sec=metadata.duration_sec,
    )
    result = complete_from_plan(
        plan,
        video=video,
        metadata=metadata,
        config=config,
        output_root=root,
        n_scenes=0,
        n_keyframes=0,
        sampling_strategy="video",
        accurate_cuts=accurate_cuts,
        write_outputs=write_outputs,
        cost_estimate_usd=plan.metadata.cost_usd or 0.0,
        keyframes=None,
        profile=profile,
    )
    _clear_resume_state(root)
    return result


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

    pre_roll = profile.pre_roll_sec if profile else 0.0
    post_roll = profile.post_roll_sec if profile else 0.0

    dispatch: DispatchResult | None = None
    if write_outputs and ranked:
        dispatch = dispatch_outputs(
            video,
            ranked,
            metadata,
            output_root,
            modes=config.output.modes,
            merge_order=config.output.merge_order,
            accurate=accurate_cuts,
            pre_roll_sec=pre_roll,
            post_roll_sec=post_roll,
            extra_manifest={
                "vlm": {
                    "provider": plan.metadata.vlm_provider,
                    "model": plan.metadata.vlm_model,
                    "prompt_version": plan.metadata.prompt_version,
                    "analysis_time_sec": plan.metadata.analysis_time_sec,
                    "cost_estimate_usd": cost_estimate_usd,
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
        dispatch=dispatch,
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
) -> Path:
    """Persist the parameters ``complete_from_plan`` needs to finish later.

    ``content_hint`` is persisted so the resumed analysis rebuilds the same
    profile and applies the same per-mode keep threshold (e.g. highlights=7)
    that the live path would have — without it a resumed host run would
    silently fall back to the global default.
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
