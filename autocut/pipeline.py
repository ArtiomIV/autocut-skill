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
from autocut.content import ContentProfile, profile_for
from autocut.models import AnalysisHints, ClipPlan, ContentHint, Keyframe, VideoMetadata
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

log = logging.getLogger(__name__)

DEFAULT_KEYFRAME_SUBDIR = "keyframes"
RESUME_STATE_FILENAME = ".autocut_resume.json"

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
    profile = profile_for(effective_hints.content_hint)
    effective_hints = _apply_profile_to_hints(effective_hints, profile)
    effective_sampling_strategy = (
        sampling_strategy if sampling_strategy != "hybrid" else profile.sampling_strategy
    )

    log.info("pipeline: probe %s", video)
    metadata = probe_video(video)

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
