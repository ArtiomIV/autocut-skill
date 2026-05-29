"""Choose WHICH timestamps to feed the keyframe extractor.

Decoupling the sampling decision from the JPEG extraction lets us swap
strategies without touching ``keyframes.py``. The four strategies cover
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
- ``motion``: when hot windows fire, sample ONLY inside those ``HotWindow``
  intervals (one frame per second by default) and skip the dead time
  entirely — a stills VLM judges the action better when the key moments are
  not diluted by idle frames. With no hot window detected, fall back to a
  sparse uniform baseline so the caller never gets an empty list. Best when
  the pipeline has precomputed motion + audio profiles for the video.

Output is a list of ``FrameSpec`` records with the timestamp and the
originating scene index (for traceability). The list is sorted by
timestamp and deduplicated within a small tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Literal

from autocut.models import Scene
from autocut.video.hot_windows import HotWindow

SamplingStrategy = Literal["scene", "uniform", "hybrid", "motion"]

_DEFAULT_PER_SCENE = 2
_DEFAULT_UNIFORM_INTERVAL_SEC = 2.0
_DEFAULT_HYBRID_MAX_GAP_SEC = 3.0
# Two timestamps closer than this collapse into one.
_DEDUP_TOLERANCE_SEC = 0.25

# Short videos suffer from scene-based sampling: a 30 s boxing clip with one
# uncut shot only yields 2 keyframes, far too sparse for the VLM to reason
# about. Under this threshold, ``hybrid`` is rerouted to a denser uniform pass.
_SHORT_VIDEO_THRESHOLD_SEC = 120.0
_SHORT_VIDEO_INTERVAL_SEC = 1.5

# Final safety net: regardless of strategy, we always feed the VLM at least
# this many keyframes. Below the floor, we densify with uniform sampling.
_DEFAULT_MIN_KEYFRAMES = 3

# Defaults for the ``motion`` strategy. Tunable per-call so the pipeline can
# pass tighter values for sport-style profiles.
_DEFAULT_MOTION_BASE_INTERVAL_SEC = 4.0  # uniform fallback when NO hot window fires
_DEFAULT_MOTION_DENSE_INTERVAL_SEC = 1.0  # one frame per second inside hot windows
_DEFAULT_MOTION_HOT_PADDING_SEC = 1.0  # widen each hot window by this on both sides


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


def sample_motion(
    hot_windows: list[HotWindow],
    duration_sec: float,
    *,
    base_interval_sec: float = _DEFAULT_MOTION_BASE_INTERVAL_SEC,
    dense_interval_sec: float = _DEFAULT_MOTION_DENSE_INTERVAL_SEC,
    hot_padding_sec: float = _DEFAULT_MOTION_HOT_PADDING_SEC,
) -> list[FrameSpec]:
    """Dense sampling ONLY inside ``hot_windows`` — no all-video baseline.

    When at least one hot window fires we sample exclusively inside the
    (padded) windows at ``dense_interval_sec`` and skip every dead second:
    feeding a stills VLM only the key moments stops the decisive action from
    being diluted by idle frames (the failure mode that made a flat 67-frame
    batch miss the climax).

    Empty ``hot_windows`` falls back to uniform sampling at
    ``base_interval_sec`` so the caller never gets an empty list when
    the video has duration but no detected motion.
    """
    if duration_sec <= 0:
        raise ValueError("duration_sec must be > 0")
    if base_interval_sec <= 0 or dense_interval_sec <= 0:
        raise ValueError("intervals must be > 0")
    if hot_padding_sec < 0:
        raise ValueError("hot_padding_sec must be >= 0")

    if not hot_windows:
        # No detected action: a sparse uniform pass keeps the VLM from getting
        # an empty list while staying cheap.
        return _dedupe_and_sort(sample_uniform(duration_sec, interval_sec=base_interval_sec))

    specs: list[FrameSpec] = []
    for window in hot_windows:
        # Pad the window on both sides so we catch the lead-in to the action
        # and the immediate aftermath; clamp to the source bounds.
        start = max(0.0, window.start_sec - hot_padding_sec)
        end = min(duration_sec, window.end_sec + hot_padding_sec)
        # Sample inside this padded window at dense_interval, midpoint-aligned
        # like sample_uniform so the first sample sits comfortably inside.
        t = start + dense_interval_sec / 2.0
        while t < end:
            specs.append(FrameSpec(timestamp=timedelta(seconds=t), scene_index=-1))
            t += dense_interval_sec

    return _dedupe_and_sort(specs)


def build_sampler(
    strategy: SamplingStrategy,
    scenes: list[Scene],
    duration_sec: float,
    *,
    per_scene: int = _DEFAULT_PER_SCENE,
    interval_sec: float = _DEFAULT_UNIFORM_INTERVAL_SEC,
    max_gap_sec: float = _DEFAULT_HYBRID_MAX_GAP_SEC,
    min_keyframes: int = _DEFAULT_MIN_KEYFRAMES,
    hot_windows: list[HotWindow] | None = None,
    motion_base_interval_sec: float = _DEFAULT_MOTION_BASE_INTERVAL_SEC,
    motion_dense_interval_sec: float = _DEFAULT_MOTION_DENSE_INTERVAL_SEC,
    motion_hot_padding_sec: float = _DEFAULT_MOTION_HOT_PADDING_SEC,
) -> list[FrameSpec]:
    """Single entry point used by the pipeline. Dispatches on strategy.

    Two safety nets always apply on top of the chosen strategy:
    1. ``hybrid`` on a short video (< ``_SHORT_VIDEO_THRESHOLD_SEC``) is
       rerouted to dense uniform sampling — scene logic would underproduce.
    2. If the final spec count is below ``min_keyframes``, we densify with
       uniform sampling so the VLM always has enough frames to judge.

    ``motion`` strategy additionally requires the caller to pass
    ``hot_windows`` precomputed via
    ``autocut.video.hot_windows.find_hot_windows`` from motion + audio
    profiles. Passing an empty list is fine — sample_motion will fall
    back to baseline-only uniform sampling.
    """
    if strategy == "motion":
        if hot_windows is None:
            raise ValueError(
                "strategy='motion' requires hot_windows (compute via hot_windows.find_hot_windows)"
            )
        specs = sample_motion(
            hot_windows,
            duration_sec,
            base_interval_sec=motion_base_interval_sec,
            dense_interval_sec=motion_dense_interval_sec,
            hot_padding_sec=motion_hot_padding_sec,
        )
    elif strategy == "hybrid" and duration_sec < _SHORT_VIDEO_THRESHOLD_SEC:
        specs = _dedupe_and_sort(
            sample_uniform(duration_sec, interval_sec=_SHORT_VIDEO_INTERVAL_SEC)
        )
    elif strategy == "scene":
        specs = _dedupe_and_sort(sample_scene_based(scenes, per_scene=per_scene))
    elif strategy == "uniform":
        specs = _dedupe_and_sort(sample_uniform(duration_sec, interval_sec=interval_sec))
    elif strategy == "hybrid":
        specs = sample_hybrid(scenes, duration_sec, per_scene=per_scene, max_gap_sec=max_gap_sec)
    else:
        # mypy enforces exhaustiveness via the Literal; this guards runtime callers.
        raise ValueError(f"unknown sampling strategy: {strategy!r}")

    if len(specs) < min_keyframes and duration_sec > 0:
        target_interval = duration_sec / (min_keyframes + 1)
        if target_interval > 0:
            specs = _dedupe_and_sort(sample_uniform(duration_sec, interval_sec=target_interval))
    return specs


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
