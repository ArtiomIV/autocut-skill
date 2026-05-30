"""Two-stage boundary refine: snap an action clip's START to the real impact.

The VLM video route locates the right moment but its timestamps are coarse
(Gemini ingests video at ~1fps): for a knockdown it anchors the clip on the
long, clearly visible referee count and starts ~8s AFTER the punch that caused
it, so the punch is cut off. This module re-anchors the START of an action clip
to the actual impact, measured LOCALLY on the source at full frame rate —
optical-flow motion (the punch + the opponent dropping is a sharp spike),
confirmed by an audio onset (the thud / crowd surge).

It is **self-gating**: it only moves the boundary when a clearly dominant impact
spike exists in the search window just before the VLM start. A non-impact moment
(a fighter's ring entrance, a talking head) produces no dominant spike, so the
clip is left exactly as the VLM cut it. The caller decides *whether to attempt*
refine (sport profile, video route); this module decides *whether there is an
impact to snap to*.
"""

from __future__ import annotations

import contextlib
import logging
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from autocut.video.audio_peaks import AudioAnalysisError, compute_audio_profile
from autocut.video.cutter import CutRequest, cut_clip
from autocut.video.motion import MotionSample, compute_motion_profile

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RefineConfig:
    """Tunables for the impact-snap. Defaults sized from the boxing E2E."""

    # How far BEFORE the VLM start to look for the impact. The observed KO
    # offset was ~8s; 12s gives margin without reaching into the prior moment.
    search_back_sec: float = 12.0
    # A small look-ahead so an impact landing right on the VLM start is caught.
    search_fwd_sec: float = 1.0
    # Tiny lead-in kept before the detected impact so the punch is not clipped.
    min_lead_sec: float = 0.5
    # Optical-flow sampling rate for the local slice. Denser than the Phase D
    # default (10fps) because we want sub-second impact precision here.
    motion_fps: float = 15.0
    # Dominance gate: the peak must exceed mean + z * std of the window's
    # motion to count as an impact. This is what makes the refine self-gating —
    # a smooth ring entrance has no peak this sharp.
    dominance_z: float = 2.0
    # When walking back from the peak to the spike's rising edge (the punch),
    # stop where motion falls back below mean + this many std.
    onset_floor_z: float = 0.5
    # An audio onset within this many seconds of the motion spike confirms the
    # impact (logged; raises confidence). Not required by default.
    audio_confirm_tol_sec: float = 0.7


@dataclass(frozen=True, slots=True)
class RefineResult:
    """Outcome of one refine attempt, for logging and tests."""

    original_start_sec: float
    refined_start_sec: float
    moved: bool
    impact_sec: float | None
    audio_confirmed: bool
    reason: str


def refine_start_to_impact(
    video_path: str | Path,
    clip_start_sec: float,
    clip_end_sec: float,
    video_duration_sec: float,
    *,
    config: RefineConfig | None = None,
    ffmpeg_path: str | None = None,
) -> RefineResult:
    """Return the impact-snapped start for one action clip.

    Looks in ``[clip_start - search_back, clip_start + search_fwd]`` for a
    dominant optical-flow spike (the punch). When found EARLIER than the VLM
    start, returns a ``RefineResult`` with ``moved=True`` and the new start
    (= impact - ``min_lead_sec``). Otherwise returns the original start
    unchanged — no dominant spike, or the impact is not before the VLM start.
    Never raises on analysis failure: a refine is best-effort, so an ffmpeg or
    decode error degrades to "unchanged" rather than breaking the run.
    """
    cfg = config or RefineConfig()
    video = Path(video_path)

    win_start = max(0.0, clip_start_sec - cfg.search_back_sec)
    win_end = min(video_duration_sec, clip_start_sec + cfg.search_fwd_sec)
    if win_end - win_start < 1.0:
        return _unchanged(clip_start_sec, "search window too small")

    try:
        with _slice(video, win_start, win_end, ffmpeg_path=ffmpeg_path) as clip:
            motion = compute_motion_profile(
                clip, target_fps=cfg.motion_fps, ffmpeg_path=ffmpeg_path
            )
            onsets = _onset_times(clip, ffmpeg_path=ffmpeg_path)
    except (OSError, RuntimeError) as exc:
        # Best-effort: never let a refine failure abort the run.
        log.warning("refine: local analysis failed (%s); leaving boundary as-is", exc)
        return _unchanged(clip_start_sec, f"analysis failed: {exc}")

    return _decide_impact(
        motion,
        onsets,
        win_start=win_start,
        clip_start_sec=clip_start_sec,
        clip_end_sec=clip_end_sec,
        cfg=cfg,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _unchanged(start_sec: float, reason: str) -> RefineResult:
    return RefineResult(
        original_start_sec=start_sec,
        refined_start_sec=start_sec,
        moved=False,
        impact_sec=None,
        audio_confirmed=False,
        reason=reason,
    )


def _decide_impact(
    motion: list[MotionSample],
    onsets: list[float],
    *,
    win_start: float,
    clip_start_sec: float,
    clip_end_sec: float,
    cfg: RefineConfig,
) -> RefineResult:
    """Pure decision: given the window's motion + audio onsets (both relative to
    the slice start), return the snapped start. Side-effect free for testing.

    ``onsets`` are slice-relative timestamps, like ``motion[i].timestamp_sec``.
    """
    if len(motion) < 3:
        return _unchanged(clip_start_sec, "not enough motion samples")

    mags = [s.magnitude for s in motion]
    mean = sum(mags) / len(mags)
    std = _std(mags, mean)
    if std <= 1e-9:
        return _unchanged(clip_start_sec, "flat motion (no spike)")

    peak_idx = max(range(len(mags)), key=mags.__getitem__)
    if mags[peak_idx] < mean + cfg.dominance_z * std:
        # No clearly dominant spike -> not an impact moment. Self-gate: leave it.
        return _unchanged(clip_start_sec, "no dominant impact spike")

    # Walk back from the peak to the spike's rising edge -> the punch instant.
    floor = mean + cfg.onset_floor_z * std
    onset_idx = peak_idx
    while onset_idx > 0 and mags[onset_idx - 1] > floor:
        onset_idx -= 1
    impact_rel = motion[onset_idx].timestamp_sec
    impact_sec = win_start + impact_rel

    audio_confirmed = any(abs(o - impact_rel) <= cfg.audio_confirm_tol_sec for o in onsets)

    new_start = max(win_start, impact_sec - cfg.min_lead_sec)
    # Only ever pull the start EARLIER, and never collapse the clip.
    if new_start >= clip_start_sec - 0.05:
        return _unchanged(clip_start_sec, "impact not before the VLM start")
    if new_start >= clip_end_sec - 1.0:
        return _unchanged(clip_start_sec, "refined start would collapse the clip")

    log.info(
        "refine: start %.2fs -> %.2fs (impact %.2fs, audio_confirmed=%s)",
        clip_start_sec,
        new_start,
        impact_sec,
        audio_confirmed,
    )
    return RefineResult(
        original_start_sec=clip_start_sec,
        refined_start_sec=new_start,
        moved=True,
        impact_sec=impact_sec,
        audio_confirmed=audio_confirmed,
        reason="snapped to dominant impact spike",
    )


@contextlib.contextmanager
def _slice(
    video: Path, start_sec: float, end_sec: float, *, ffmpeg_path: str | None
) -> Iterator[Path]:
    """Yield an accurate-cut temp slice of ``[start, end]`` (frame 0 == start).

    Accurate (re-encode) cut so the slice's timeline starts EXACTLY at
    ``start_sec`` — stream-copy would snap to a keyframe and break the
    absolute-time mapping. Cleaned up on exit.
    """
    tmp = Path(tempfile.mkdtemp(prefix="autocut_refine_"))
    out = tmp / "window.mp4"
    try:
        cut_clip(
            video,
            CutRequest(
                start=timedelta(seconds=start_sec),
                end=timedelta(seconds=end_sec),
                output_path=out,
            ),
            accurate=True,
            ffmpeg_path=ffmpeg_path,
        )
        yield out
    finally:
        with contextlib.suppress(OSError):
            for f in tmp.iterdir():
                f.unlink()
            tmp.rmdir()


def _onset_times(clip: Path, *, ffmpeg_path: str | None) -> list[float]:
    """Audio onset timestamps in the slice, or empty when there is no audio."""
    try:
        profile = compute_audio_profile(clip, ffmpeg_path=ffmpeg_path)
    except AudioAnalysisError:
        return []
    return [s.timestamp_sec for s in profile if s.is_onset]


def _std(values: list[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return float(var**0.5)
