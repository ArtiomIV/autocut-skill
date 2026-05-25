"""Pipeline orchestrator — chains every stage from video file to written clips.

Full happy path:

    probe -> scene_detect -> frame_sampler -> extract_keyframes
          -> (cost cap check) -> provider.analyze
          -> rank_clips -> dispatch_outputs

The orchestrator is provider-agnostic: it only talks to the abstract
``VLMProvider`` interface. The ``host_agent`` pause flow surfaces here as
``HostAgentPauseRequested`` propagating untouched — the CLI catches it
and tells the user how to resume.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from autocut.config import AutoCutConfig
from autocut.models import AnalysisHints, ClipPlan, ContentHint, Keyframe, VideoMetadata
from autocut.output import DispatchResult, dispatch_outputs
from autocut.scoring import RankedClip, rank_clips
from autocut.video import (
    build_sampler,
    detect_scenes,
    extract_keyframes,
    probe_video,
)
from autocut.vlm import CostEstimate, VLMError, VLMProvider

log = logging.getLogger(__name__)

DEFAULT_KEYFRAME_SUBDIR = "keyframes"

# A confirm hook: returns True to proceed, False to abort. The CLI passes
# its interactive prompt; tests pass a constant.
ConfirmHook = Callable[[CostEstimate, float], bool]


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

    log.info("pipeline: scene detect (threshold=%s)", config.advanced.scene_threshold)
    scenes = detect_scenes(video, threshold=config.advanced.scene_threshold)

    log.info("pipeline: sampling (strategy=%s, %d scenes)", sampling_strategy, len(scenes))
    specs = build_sampler(
        sampling_strategy,  # type: ignore[arg-type]
        scenes,
        metadata.duration_sec,
        per_scene=config.advanced.keyframes_per_scene,
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

    ranked = rank_clips(plan, config.scoring)
    log.info("pipeline: %d clip(s) survived ranking", len(ranked))

    dispatch: DispatchResult | None = None
    if write_outputs and ranked:
        dispatch = dispatch_outputs(
            video,
            ranked,
            metadata,
            root,
            modes=config.output.modes,
            merge_order=config.output.merge_order,
            accurate=accurate_cuts,
            extra_manifest={
                "vlm": {
                    "provider": plan.metadata.vlm_provider,
                    "model": plan.metadata.vlm_model,
                    "prompt_version": plan.metadata.prompt_version,
                    "analysis_time_sec": plan.metadata.analysis_time_sec,
                    "cost_estimate_usd": estimate.estimated_total_usd,
                },
                "sampling": {
                    "strategy": sampling_strategy,
                    "n_keyframes": len(keyframes),
                    "n_scenes": len(scenes),
                },
            },
        )

    return AnalysisResult(
        metadata=metadata,
        keyframes=keyframes,
        plan=plan,
        ranked=ranked,
        dispatch=dispatch,
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
