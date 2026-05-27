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
from pathlib import Path

from autocut.config import AutoCutConfig
from autocut.content import ContentProfile, detect_content_hint, profile_for
from autocut.models import (
    AnalysisHints,
    ClipPlan,
    ContentHint,
    DetectionResult,
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
from autocut.vlm import CostEstimate, VLMError, VLMProvider
from autocut.vlm.base import HostAgentPauseRequested

log = logging.getLogger(__name__)

DEFAULT_KEYFRAME_SUBDIR = "keyframes"
RESUME_STATE_FILENAME = ".autocut_resume.json"
# Phase E: separate state file so detection-phase resume and analysis-phase
# resume don't collide. The CLI checks the detection file first.
DETECTION_RESUME_STATE_FILENAME = ".autocut_detect_resume.json"

# Detection confidence below this threshold falls back to HYBRID_PROFILE.
# 0.5 is permissive: we trust the model's self-assessment except when it
# explicitly signals uncertainty. Retunable after empirical runs.
_AUTO_DETECT_CONFIDENCE_THRESHOLD: float = 0.5

# A confirm hook: returns True to proceed, False to abort. The CLI passes
# its interactive prompt; tests pass a constant.
ConfirmHook = Callable[[CostEstimate, float], bool]


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
) -> AnalysisResult:
    """Run the full pipeline. Returns a populated ``AnalysisResult``."""
    video = Path(video_path)
    root = (output_root or config.output.base_dir).resolve()
    keyframe_dir = root / DEFAULT_KEYFRAME_SUBDIR

    effective_hints = _resolve_hints(hints, config)

    log.info("pipeline: probe %s", video)
    metadata = probe_video(video)

    # Phase E: if the caller asked for auto-detect (or didn't pass a hint
    # at all), run the detection pre-step before picking the profile.
    if effective_hints.content_hint == ContentHint.auto:
        try:
            detection = await _run_auto_detect(video, provider, metadata=metadata, output_root=root)
        except HostAgentPauseRequested:
            # Host-agent detection pause: persist enough state that the
            # CLI can re-enter the pipeline after the agent fills in
            # DETECTION_RESPONSE.json, then re-raise so the CLI prints
            # the pause message and exits cleanly.
            _write_detection_resume_state(
                root,
                video=video,
                metadata=metadata,
                initial_hints=effective_hints,
                sampling_strategy=sampling_strategy,
                accurate_cuts=accurate_cuts,
                write_outputs=write_outputs,
            )
            raise
        effective_hints = _apply_detection_to_hints(effective_hints, detection)

    profile = profile_for(effective_hints.content_hint)
    effective_hints = _apply_profile_to_hints(effective_hints, profile)
    effective_sampling_strategy = (
        sampling_strategy if sampling_strategy != "hybrid" else profile.sampling_strategy
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
    ranked = rank_clips(plan, config.scoring)
    log.info("pipeline: %d clip(s) survived ranking", len(ranked))

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
) -> Path:
    """Persist the parameters ``complete_from_plan`` needs to finish later."""
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


async def _run_auto_detect(
    video: Path,
    provider: VLMProvider,
    *,
    metadata: VideoMetadata,
    output_root: Path,
) -> DetectionResult:
    """Run Phase E detection and surface a clean DetectionResult.

    Wraps ``detect_content_hint`` in defensive handling for ``VLMError``
    only. ``HostAgentPauseRequested`` is allowed to propagate so the
    caller can persist a detection-phase resume state and re-raise to the
    CLI (which prints the resume instructions).
    """
    try:
        return await detect_content_hint(
            video,
            provider,
            metadata=metadata,
            output_root=output_root,
        )
    except VLMError as exc:
        log.warning("pipeline: auto-detect failed (%s); HYBRID fallback", exc)
        return DetectionResult(
            content_hint=ContentHint.other,
            confidence=0.0,
            reasoning=f"auto-detect failed: {exc}",
        )


def _apply_detection_to_hints(
    hints: AnalysisHints,
    detection: DetectionResult,
) -> AnalysisHints:
    """Update hints with the detected content_hint when confidence is high enough.

    Below the threshold we keep ``content_hint == auto`` so ``profile_for``
    falls back to HYBRID. Returning the hints unchanged in that case keeps
    a single ownership rule (only this helper decides whether to apply).
    """
    if detection.confidence >= _AUTO_DETECT_CONFIDENCE_THRESHOLD:
        log.info(
            "pipeline: auto-detect picked %s (confidence %.2f) — %s",
            detection.content_hint.value,
            detection.confidence,
            detection.reasoning,
        )
        return hints.model_copy(update={"content_hint": detection.content_hint})
    log.warning(
        "pipeline: detection confidence %.2f below %.2f; HYBRID fallback (detected=%s, reason=%r)",
        detection.confidence,
        _AUTO_DETECT_CONFIDENCE_THRESHOLD,
        detection.content_hint.value,
        detection.reasoning,
    )
    return hints


# ---------------------------------------------------------------------------
# Detection-phase resume (Phase E host-agent path)
# ---------------------------------------------------------------------------


def _write_detection_resume_state(
    output_root: Path,
    *,
    video: Path,
    metadata: VideoMetadata,
    initial_hints: AnalysisHints,
    sampling_strategy: str,
    accurate_cuts: bool,
    write_outputs: bool,
) -> Path:
    """Persist enough state to continue the pipeline after a detection pause.

    Stores the AnalysisHints **before** detection updates them, so the
    resume path applies exactly the same logic (threshold + model_copy)
    that the live path would have. Includes the probed metadata so we
    don't re-probe on resume (it's deterministic + cheap, but persistence
    avoids any subtle timing inconsistency).
    """
    output_root.mkdir(parents=True, exist_ok=True)
    state = {
        "phase": "detection",
        "video_path": str(video.resolve()),
        "video_metadata": json.loads(metadata.model_dump_json()),
        "output_root": str(output_root.resolve()),
        "sampling_strategy": sampling_strategy,
        "accurate_cuts": accurate_cuts,
        "write_outputs": write_outputs,
        "initial_hints": json.loads(initial_hints.model_dump_json()),
    }
    path = output_root / DETECTION_RESUME_STATE_FILENAME
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    log.info("pipeline: detection resume state written → %s", path)
    return path


def _clear_detection_resume_state(output_root: Path) -> None:
    """Best-effort delete of the detection sidecar after that phase completes."""
    path = output_root / DETECTION_RESUME_STATE_FILENAME
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        log.debug("could not clear detection resume state %s: %s", path, exc)


def load_detection_resume_state(output_root: Path) -> dict[str, object]:
    """Load the detection sidecar. Raises ``ResumeStateError`` on issues."""
    path = output_root / DETECTION_RESUME_STATE_FILENAME
    if not path.is_file():
        raise ResumeStateError(
            f"detection resume state file not found: {path} "
            f"(no paused detection to resume in this output dir)"
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResumeStateError(f"failed to read detection resume state: {exc}") from exc
    if not isinstance(data, dict):
        raise ResumeStateError("detection resume state top-level value is not an object")
    required = {"video_path", "video_metadata", "initial_hints"}
    missing = required - data.keys()
    if missing:
        raise ResumeStateError(f"detection resume state missing required keys: {sorted(missing)}")
    return data


async def resume_after_detection(
    detection: DetectionResult,
    *,
    state: dict[str, object],
    provider: VLMProvider,
    config: AutoCutConfig,
    confirm_cost: ConfirmHook | None = None,
) -> AnalysisResult:
    """Re-enter the pipeline after detection has been resolved.

    The CLI loads ``DETECTION_RESPONSE.json`` via the provider, calls
    this with the resulting ``DetectionResult``, and the rest of the
    pipeline runs as if the live path had produced the same detection.
    The analysis pause (``HostAgentPauseRequested``) is allowed to
    propagate so the CLI can show its second pause message.
    """
    video = Path(str(state["video_path"]))
    if not video.is_file():
        raise ResumeStateError(f"original video no longer at {video}; cannot resume detection")

    try:
        metadata = VideoMetadata.model_validate(state["video_metadata"])
    except Exception as exc:
        raise ResumeStateError(f"resume state video_metadata is invalid: {exc}") from exc
    try:
        initial_hints = AnalysisHints.model_validate(state["initial_hints"])
    except Exception as exc:
        raise ResumeStateError(f"resume state initial_hints is invalid: {exc}") from exc

    output_root = Path(str(state.get("output_root", str(config.output.base_dir)))).resolve()
    sampling_strategy = str(state.get("sampling_strategy", "hybrid"))
    accurate_cuts = bool(state.get("accurate_cuts", False))
    write_outputs = bool(state.get("write_outputs", True))

    # Apply the detection just like the live path does (threshold etc.).
    effective_hints = _apply_detection_to_hints(initial_hints, detection)

    # Detection step is done; drop the sidecar so a subsequent failure
    # doesn't re-trigger detection resume next time.
    _clear_detection_resume_state(output_root)

    # Re-enter the pipeline with the resolved hint. We pass it explicitly
    # so ``run_analysis`` does NOT enter the auto-detect branch again.
    # The pipeline will re-probe (cheap, ~0.5s, deterministic), so we
    # discard the persisted metadata at this layer.
    del metadata
    return await run_analysis(
        video,
        provider,
        config=config,
        hints=effective_hints,
        output_root=output_root,
        sampling_strategy=sampling_strategy,
        write_outputs=write_outputs,
        accurate_cuts=accurate_cuts,
        confirm_cost=confirm_cost,
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
