"""Pipeline orchestrator — chains every stage from video file to clip plan.

This is the single place that knows the full happy path:

    probe -> scene_detect -> frame_sampler -> extract_keyframes
          -> provider.analyze -> ClipPlan

Output writing (separate / merged) is wired in M4; for now ``run_analysis``
returns the validated ``ClipPlan`` and the list of extracted keyframes so
the CLI can preview the result before any cutting happens.

The orchestrator is provider-agnostic: it only talks to the abstract
``VLMProvider`` interface. The ``host_agent`` pause flow surfaces here as
``HostAgentPauseRequested`` propagating untouched — the CLI catches it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from autocut.config import AutoCutConfig
from autocut.models import AnalysisHints, ClipPlan, ContentHint, Keyframe, VideoMetadata
from autocut.video import (
    build_sampler,
    detect_scenes,
    extract_keyframes,
    probe_video,
)
from autocut.vlm import VLMProvider

log = logging.getLogger(__name__)

DEFAULT_KEYFRAME_SUBDIR = "keyframes"


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Bundle returned by ``run_analysis``: metadata + sampled frames + clip plan."""

    metadata: VideoMetadata
    keyframes: list[Keyframe]
    plan: ClipPlan


async def run_analysis(
    video_path: str | Path,
    provider: VLMProvider,
    *,
    config: AutoCutConfig,
    hints: AnalysisHints | None = None,
    output_root: Path | None = None,
    sampling_strategy: str = "hybrid",
) -> AnalysisResult:
    """Run probe -> scenes -> sampler -> keyframes -> provider.analyze.

    ``output_root`` defaults to ``config.output.base_dir``. Keyframes go
    under ``<output_root>/<DEFAULT_KEYFRAME_SUBDIR>/``. Cutting itself
    happens later (CLI / M4) once the user has reviewed the plan.
    """
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
    return AnalysisResult(metadata=metadata, keyframes=keyframes, plan=plan)


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
