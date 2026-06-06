"""Deterministic ADVISORY signals (audio-energy peaks + scene cuts) for the host.

The host agent calls ``autocut signals`` to LOCATE candidate regions cheaply before
spending context on contact sheets. Two resolution-independent cues:

- **audio peaks** — loud live action shows up as energy peaks in the audio: a short
  sharp transient (``impact``, e.g. a clean landed shot) or a sustained rise
  (``crowd``, e.g. a roar after a knockdown). Computed from the RMS energy envelope.
- **scene cuts** — replays and graphic transitions show up as scene boundaries
  (PySceneDetect). A slow-motion replay is often audio-SILENT, so the scene cut is
  how you still find it.

ADVISORY ONLY, NEVER a gate. These PRIORITISE where to look and CROSS-CHECK a
candidate; they must never be the sole localiser. A quiet-but-real moment (a silent
slow-mo replay, a clean body shot with no crowd) must stay reachable through a visual
coarse pass — so for a ``--query`` or quiet content the agent ignores these and scans
visually. No VLM.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from autocut.video.ffmpeg_path import FFmpegResolveError, ffmpeg_binary
from autocut.video.scene_detect import SceneDetectError, detect_scenes

# Audio analysis constants. 16 kHz mono matches speech/broadcast and is plenty for
# an energy envelope; 100 ms frames give a smooth curve without smearing a fast
# impact across seconds.
_SR_HZ = 16_000
_FRAME_MS = 100
_DECODE_TIMEOUT_SEC = 900
# Local baseline window (rolling median) — long enough to ride loudness drift over a
# fight, short enough to stay local.
_BASELINE_WIN_SEC = 15.0
# A peak's residual (env above the local baseline) must exceed k * global MAD.
_DEFAULT_K = 4.0
# Merge two peaks whose centres fall within this gap (keep the stronger).
_MIN_GAP_SEC = 2.0
# An event narrower than this is a sharp ``impact``; wider is a sustained ``crowd``.
_IMPACT_MAX_WIDTH_SEC = 0.8


class SignalsError(RuntimeError):
    """Raised when audio/scene signal extraction fails."""


@dataclass(frozen=True)
class AudioPeak:
    """A local maximum of the audio energy envelope."""

    t: float  # seconds, centre of the peak
    strength: float  # 0..1, height above the local baseline (normalised)
    kind: str  # "impact" (short, sharp) | "crowd" (sustained)


@dataclass(frozen=True)
class SceneCut:
    """A detected scene boundary (camera cut / graphic / replay transition)."""

    t: float  # seconds, the cut point


# ---------------------------------------------------------------------------
# Audio peaks
# ---------------------------------------------------------------------------


def _decode_audio_mono(video: Path, ffmpeg_path: str | None) -> np.ndarray:
    """Decode ``video``'s audio to a mono float32 array at ``_SR_HZ``.

    Returns an empty array if the source has no audio stream.
    """
    try:
        binary = ffmpeg_binary(ffmpeg_path)
    except FFmpegResolveError as exc:  # pragma: no cover - env-specific
        raise SignalsError(f"ffmpeg not found: {exc}") from exc

    args = [
        binary,
        "-v",
        "error",
        "-i",
        str(video),
        "-vn",  # drop video
        "-ac",
        "1",  # mono
        "-ar",
        str(_SR_HZ),
        "-f",
        "f32le",  # raw 32-bit float PCM on stdout
        "-",
    ]
    try:
        completed = subprocess.run(  # noqa: S603 - fixed arg list, no shell
            args,
            capture_output=True,
            timeout=_DECODE_TIMEOUT_SEC,
            check=False,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        raise SignalsError(f"failed to invoke ffmpeg: {exc}") from exc
    if completed.returncode != 0:
        # No audio stream is a valid outcome (return empty), not an error.
        if b"does not contain any stream" in completed.stderr or not completed.stdout:
            return np.empty(0, dtype=np.float32)
        raise SignalsError(
            f"ffmpeg audio decode failed: {completed.stderr.decode(errors='replace').strip()}"
        )
    return np.frombuffer(completed.stdout, dtype=np.float32)


def _rms_envelope(
    samples: np.ndarray, sr_hz: int = _SR_HZ, frame_ms: int = _FRAME_MS
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(times, env)``: per-frame RMS energy and each frame's centre time."""
    frame = max(1, int(sr_hz * frame_ms / 1000))
    n_frames = len(samples) // frame
    if n_frames == 0:
        return np.empty(0), np.empty(0)
    trimmed = samples[: n_frames * frame].reshape(n_frames, frame).astype(np.float64)
    env = np.sqrt(np.mean(trimmed * trimmed, axis=1))
    times = (np.arange(n_frames) + 0.5) * frame / sr_hz
    return times, env


def _rolling_median(env: np.ndarray, win: int) -> np.ndarray:
    """Edge-padded rolling median of ``env`` over a window of ``win`` frames."""
    if win <= 1 or len(env) <= 1:
        return env.copy()
    win = min(win, len(env))
    half = win // 2
    padded = np.pad(env, (half, win - half - 1), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, win)
    return np.asarray(np.median(windows, axis=1), dtype=float)


def peaks_from_envelope(
    times: np.ndarray,
    env: np.ndarray,
    *,
    frame_ms: int = _FRAME_MS,
    k: float = _DEFAULT_K,
    min_gap_sec: float = _MIN_GAP_SEC,
) -> list[AudioPeak]:
    """Pure peak detector over an energy envelope (no I/O — unit-testable).

    Subtracts a rolling-median baseline (rides loudness drift), thresholds the
    residual at ``k`` times its global MAD, groups supra-threshold runs into
    events, and classifies each by width (sharp ``impact`` vs sustained ``crowd``).
    """
    if len(env) == 0:
        return []
    baseline_win = max(1, int(_BASELINE_WIN_SEC * 1000 / frame_ms))
    baseline = _rolling_median(env, baseline_win)
    resid = env - baseline
    # Robust noise scale from the BULK (IQR), not MAD-about-the-median, which
    # collapses to 0 for sparse spikes over a quiet floor. A relative floor
    # (a fraction of the loudest residual) guarantees a sensible threshold even
    # for near-silent audio with a couple of bursts.
    q1, q3 = (float(v) for v in np.percentile(resid, [25, 75]))
    scale = (q3 - q1) / 1.349
    peak_resid = float(resid.max())
    threshold = max(k * scale, 0.15 * peak_resid)
    if threshold <= 0:
        return []
    above = resid > threshold
    if not above.any():
        return []

    # Group consecutive supra-threshold frames into events.
    edges = np.diff(above.astype(np.int8))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)
    if above[0]:
        starts.insert(0, 0)
    if above[-1]:
        ends.append(len(above))

    denom = float(resid.max()) or 1e-9
    peaks: list[AudioPeak] = []
    for s, e in zip(starts, ends, strict=False):
        idx = s + int(np.argmax(resid[s:e]))
        # Width at HALF the peak's prominence, scanning the full envelope outward
        # (not just the supra-threshold run) so a sustained crowd swell reads wide
        # while a sharp impact reads narrow.
        half = resid[idx] * 0.5
        lo = idx
        while lo > 0 and resid[lo - 1] > half:
            lo -= 1
        hi = idx
        while hi < len(resid) - 1 and resid[hi + 1] > half:
            hi += 1
        width_sec = (hi - lo + 1) * frame_ms / 1000.0
        kind = "impact" if width_sec <= _IMPACT_MAX_WIDTH_SEC else "crowd"
        strength = float(np.clip(resid[idx] / denom, 0.0, 1.0))
        peaks.append(
            AudioPeak(t=round(float(times[idx]), 2), strength=round(strength, 3), kind=kind)
        )

    # Merge peaks closer than min_gap_sec, keeping the stronger.
    peaks.sort(key=lambda p: p.t)
    merged: list[AudioPeak] = []
    for p in peaks:
        if merged and p.t - merged[-1].t < min_gap_sec:
            if p.strength > merged[-1].strength:
                merged[-1] = p
        else:
            merged.append(p)
    return merged


def audio_peaks(
    video: Path, *, ffmpeg_path: str | None = None, k: float = _DEFAULT_K
) -> list[AudioPeak]:
    """Detect audio-energy peaks in ``video`` (empty list if it has no audio)."""
    samples = _decode_audio_mono(video, ffmpeg_path)
    if samples.size == 0:
        return []
    times, env = _rms_envelope(samples)
    return peaks_from_envelope(times, env, k=k)


# ---------------------------------------------------------------------------
# Scene cuts (reuse the existing PySceneDetect wrapper)
# ---------------------------------------------------------------------------


def scene_cuts(video: Path, *, threshold: float = 27.0) -> list[SceneCut]:
    """Return scene-boundary cut points (the start of each scene after the first)."""
    try:
        scenes = detect_scenes(video, threshold=threshold)
    except SceneDetectError as exc:
        raise SignalsError(f"scene detection failed: {exc}") from exc
    cuts: list[SceneCut] = []
    for scene in scenes[1:]:  # scene[0] starts at 0 — not a cut
        cuts.append(SceneCut(t=round(scene.start.total_seconds(), 2)))
    return cuts


# ---------------------------------------------------------------------------
# Combined report (for the CLI)
# ---------------------------------------------------------------------------


def compute_signals(
    video: Path,
    *,
    ffmpeg_path: str | None = None,
    k: float = _DEFAULT_K,
    scene_threshold: float = 27.0,
    with_scenes: bool = True,
) -> dict[str, object]:
    """Run both detectors and return a JSON-serialisable advisory report."""
    peaks = audio_peaks(video, ffmpeg_path=ffmpeg_path, k=k)
    cuts = scene_cuts(video, threshold=scene_threshold) if with_scenes else []
    return {
        "path": str(video.resolve()),
        "audio_peaks": [{"t": p.t, "strength": p.strength, "kind": p.kind} for p in peaks],
        "scene_cuts": [{"t": c.t} for c in cuts],
        "note": (
            "ADVISORY, not a gate. Fine-sheet the audio peaks AND the ~30-60s after each "
            "(for replays) plus scene-cut clusters; for a query or quiet content ignore "
            "this and scan visually."
        ),
    }
