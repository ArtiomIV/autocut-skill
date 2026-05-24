"""Choose WHICH timestamps to feed the keyframe extractor.

Decoupling the sampling decision from the JPEG extraction lets us swap
strategies without touching ``keyframes.py``. The three strategies cover
the realistic content types:

- ``scene``: 1..N evenly-spaced timestamps inside each scene. Best for
  cut-heavy material (vlogs, broadcasts, gameplay with cinematics) where
  one shot ≈ one semantic unit.
- ``uniform``: 1 timestamp every ``interval_sec``, ignoring scenes. Best
  for continuous-action material with few visual cuts (boxing, MMA,
  ambient nature, dance). Without this, action inside a long shot can be
  missed entirely.
- ``hybrid``: scene-based, but any scene longer than ``max_gap_sec`` gets
  topped up with additional uniform samples so no action is missed.

Output is a list of ``FrameSpec`` records with the timestamp and the
originating scene index (for traceability). The list is sorted by
timestamp and deduplicated within a small tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Literal

from autocut.models import Scene

SamplingStrategy = Literal["scene", "uniform", "hybrid"]

_DEFAULT_PER_SCENE = 2
_DEFAULT_UNIFORM_INTERVAL_SEC = 2.0
_DEFAULT_HYBRID_MAX_GAP_SEC = 3.0
# Two timestamps closer than this collapse into one.
_DEDUP_TOLERANCE_SEC = 0.25


@dataclass(frozen=True, slots=True)
class FrameSpec:
    """A single frame to extract: when, and which scene it belongs to (or -1)."""

    timestamp: timedelta
    scene_index: int


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


def sample_scene_based(scenes: list[Scene], per_scene: int = _DEFAULT_PER_SCENE) -> list[FrameSpec]:
    """One ``per_scene`` evenly-spaced timestamp inside each scene."""
    if per_scene <= 0:
        raise ValueError("per_scene must be >= 1")
    if not scenes:
        return []
    specs: list[FrameSpec] = []
    for scene in scenes:
        total = scene.duration_sec
        if total <= 0:
            continue
        start = scene.start.total_seconds()
        for i in range(per_scene):
            offset = total * (i + 1) / (per_scene + 1)
            specs.append(
                FrameSpec(timestamp=timedelta(seconds=start + offset), scene_index=scene.index)
            )
    return specs


def sample_uniform(
    duration_sec: float, interval_sec: float = _DEFAULT_UNIFORM_INTERVAL_SEC
) -> list[FrameSpec]:
    """One timestamp every ``interval_sec`` from start to end (scene index = -1)."""
    if duration_sec <= 0:
        raise ValueError("duration_sec must be > 0")
    if interval_sec <= 0:
        raise ValueError("interval_sec must be > 0")
    specs: list[FrameSpec] = []
    # Start at interval/2 so the first sample is in the middle of the first
    # interval, then every ``interval_sec``. This avoids picking the very
    # first frame (often a fade-in) and stays comfortably inside the video.
    t = interval_sec / 2
    while t < duration_sec:
        specs.append(FrameSpec(timestamp=timedelta(seconds=t), scene_index=-1))
        t += interval_sec
    return specs


def sample_hybrid(
    scenes: list[Scene],
    duration_sec: float,
    *,
    per_scene: int = _DEFAULT_PER_SCENE,
    max_gap_sec: float = _DEFAULT_HYBRID_MAX_GAP_SEC,
) -> list[FrameSpec]:
    """Scene-based, supplemented with uniform samples inside long scenes.

    For every scene longer than ``max_gap_sec``, we add extra uniform
    samples so no two consecutive samples are more than ``max_gap_sec``
    apart. This recovers action that happens inside long, uncut shots
    (boxing rounds, podcast monologues, etc.).
    """
    if not scenes:
        return sample_uniform(duration_sec, interval_sec=max_gap_sec)

    base = sample_scene_based(scenes, per_scene=per_scene)
    supplemental: list[FrameSpec] = []
    for scene in scenes:
        if scene.duration_sec <= max_gap_sec:
            continue
        start = scene.start.total_seconds()
        n_extra = max(0, int(scene.duration_sec // max_gap_sec) - per_scene)
        if n_extra <= 0:
            continue
        for i in range(n_extra):
            offset = scene.duration_sec * (i + 1) / (n_extra + 1)
            supplemental.append(
                FrameSpec(
                    timestamp=timedelta(seconds=start + offset),
                    scene_index=scene.index,
                )
            )
    return _dedupe_and_sort(base + supplemental)


def build_sampler(
    strategy: SamplingStrategy,
    scenes: list[Scene],
    duration_sec: float,
    *,
    per_scene: int = _DEFAULT_PER_SCENE,
    interval_sec: float = _DEFAULT_UNIFORM_INTERVAL_SEC,
    max_gap_sec: float = _DEFAULT_HYBRID_MAX_GAP_SEC,
) -> list[FrameSpec]:
    """Single entry point used by the pipeline (M3). Dispatches on strategy."""
    if strategy == "scene":
        return _dedupe_and_sort(sample_scene_based(scenes, per_scene=per_scene))
    if strategy == "uniform":
        return _dedupe_and_sort(sample_uniform(duration_sec, interval_sec=interval_sec))
    if strategy == "hybrid":
        return sample_hybrid(scenes, duration_sec, per_scene=per_scene, max_gap_sec=max_gap_sec)
    # mypy enforces exhaustiveness via the Literal; this guards runtime callers.
    raise ValueError(f"unknown sampling strategy: {strategy!r}")


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _dedupe_and_sort(specs: list[FrameSpec]) -> list[FrameSpec]:
    """Sort by timestamp; drop entries within ``_DEDUP_TOLERANCE_SEC`` of each other."""
    ordered = sorted(specs, key=lambda s: s.timestamp)
    result: list[FrameSpec] = []
    for spec in ordered:
        if (
            result
            and (spec.timestamp - result[-1].timestamp).total_seconds() < _DEDUP_TOLERANCE_SEC
        ):
            continue
        result.append(spec)
    return result
