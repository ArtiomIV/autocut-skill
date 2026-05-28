"""Audio energy profile + onset detection for smart sampling (Phase D.2).

Complements ``motion.py`` with the audio channel: where a boxing match
produces a motion spike on a punch, it produces a punchy audio onset
(commentary yell, crowd reaction, impact). Combined in ``hot_windows.py``,
the two signals identify "where is something happening" much more
reliably than either alone.

We chose **ffmpeg pipe + numpy** instead of librosa on 2026-05-26:
- librosa is ~150 MB with numba/scipy/soundfile dependencies and adds
  ~3 s of import cost from numba JIT warm-up.
- For the v0.1.x use case (energy profile + simple onset detection on
  speech/sport) the marginal accuracy of librosa's spectral methods is
  not worth the weight. If a future test proves the simple detector
  misses something important, the public API of this module is shaped
  so the implementation can be swapped to librosa without touching
  callers.

The returned profile is a ``list[AudioSample]`` with one entry per RMS
window (default 100 ms). An ``is_onset`` boolean marks samples where the
energy differential exceeds a threshold — those are the "interesting
moments" hot_windows.py looks for.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from autocut.video.ffmpeg_path import FFmpegResolveError, ffmpeg_binary


class AudioAnalysisError(RuntimeError):
    """Raised when ffmpeg fails to produce a usable audio profile."""


_DEFAULT_SAMPLE_RATE_HZ: int = 16_000
_DEFAULT_WINDOW_MS: float = 100.0
# An onset is a window whose RMS jumps above the running baseline by at
# least this many standard deviations. Tuned for speech+impact audio;
# the value can be overridden per profile in hot_windows.py.
_DEFAULT_ONSET_Z_THRESHOLD: float = 1.5
# Onsets within this window of each other collapse to the highest-energy
# one to avoid double-counting a single event.
_ONSET_MIN_SEPARATION_SEC: float = 0.3
_FFMPEG_TIMEOUT_SEC: int = 600


@dataclass(frozen=True, slots=True)
class AudioSample:
    """One RMS window of mono audio at ``timestamp_sec``."""

    timestamp_sec: float
    rms: float
    is_onset: bool


def compute_audio_profile(
    video_path: str | Path,
    *,
    sample_rate_hz: int = _DEFAULT_SAMPLE_RATE_HZ,
    window_ms: float = _DEFAULT_WINDOW_MS,
    onset_z_threshold: float = _DEFAULT_ONSET_Z_THRESHOLD,
    ffmpeg_path: str | None = None,
) -> list[AudioSample]:
    """Return the audio energy profile with onset flags.

    Pipeline:
    1. ``ffmpeg`` decodes the source audio to mono ``float32`` PCM at
       ``sample_rate_hz`` and streams it to stdout.
    2. We reshape the stream into non-overlapping windows of
       ``window_ms`` and compute the per-window RMS in numpy (vectorised).
    3. Onsets are windows where ``rms[t] - rms[t - 200ms] > threshold *
       std(rms)``. Adjacent onsets within ``_ONSET_MIN_SEPARATION_SEC``
       collapse to the highest-energy one.

    Returns an empty list when the source has no audio stream.
    """
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be > 0")
    if window_ms <= 0:
        raise ValueError("window_ms must be > 0")
    if onset_z_threshold <= 0:
        raise ValueError("onset_z_threshold must be > 0")

    video = Path(video_path)
    if not video.is_file():
        raise AudioAnalysisError(f"input video not found: {video}")

    try:
        binary = ffmpeg_binary(ffmpeg_path)
    except FFmpegResolveError as exc:
        raise AudioAnalysisError(f"ffmpeg not found: {exc}") from exc

    pcm = _decode_to_pcm(binary, video, sample_rate_hz)
    if pcm.size == 0:
        # Source had no audio stream (ffmpeg returns 0 bytes cleanly with -map).
        return []

    rms_values, timestamps = _windowed_rms(pcm, sample_rate_hz=sample_rate_hz, window_ms=window_ms)
    onset_flags = _detect_onsets(
        rms_values,
        timestamps,
        window_ms=window_ms,
        z_threshold=onset_z_threshold,
    )

    return [
        AudioSample(
            timestamp_sec=float(t),
            rms=float(r),
            is_onset=bool(flag),
        )
        for t, r, flag in zip(timestamps, rms_values, onset_flags, strict=True)
    ]


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _decode_to_pcm(ffmpeg_binary: str, video: Path, sample_rate_hz: int) -> np.ndarray:
    """Pipe ffmpeg to stream mono float32 PCM and load it into a numpy array.

    Returns a zero-length array if the source has no audio stream — we
    treat that as a normal "nothing to analyse" case rather than an error.
    """
    args = [
        ffmpeg_binary,
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vn",  # drop video
        "-ac",
        "1",  # mono
        "-ar",
        str(sample_rate_hz),
        "-f",
        "f32le",  # raw little-endian float32
        "-",
    ]
    try:
        completed = subprocess.run(  # noqa: S603 - args is a fixed list, no shell
            args,
            capture_output=True,
            timeout=_FFMPEG_TIMEOUT_SEC,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioAnalysisError("ffmpeg timed out decoding audio") from exc
    except OSError as exc:
        raise AudioAnalysisError(f"failed to invoke ffmpeg: {exc}") from exc

    if completed.returncode != 0:
        stderr_text = completed.stderr.decode("utf-8", errors="replace").strip()
        # "Output file #0 does not contain any stream" is ffmpeg's way of
        # saying "no audio". Treat it as empty rather than as an error.
        if "does not contain any stream" in stderr_text.lower():
            return np.empty(0, dtype=np.float32)
        raise AudioAnalysisError(f"ffmpeg exited {completed.returncode}: {stderr_text}")

    raw = completed.stdout
    if not raw:
        return np.empty(0, dtype=np.float32)
    return np.frombuffer(raw, dtype=np.float32)


def _windowed_rms(
    pcm: np.ndarray,
    *,
    sample_rate_hz: int,
    window_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute RMS over non-overlapping windows. Returns ``(rms, timestamps)``."""
    samples_per_window = max(1, int(sample_rate_hz * window_ms / 1000.0))
    n_windows = pcm.size // samples_per_window
    if n_windows == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    # Drop the trailing partial window: a single straggler can't form a
    # meaningful RMS comparison against its neighbours.
    trimmed = pcm[: n_windows * samples_per_window].reshape((n_windows, samples_per_window))
    rms = np.sqrt(np.mean(trimmed.astype(np.float32) ** 2, axis=1))

    # Window timestamp = midpoint, so the sample lines up with where the
    # energy actually was, not where the window started.
    midpoint_sec = (window_ms / 2.0) / 1000.0
    spacing_sec = window_ms / 1000.0
    timestamps = midpoint_sec + np.arange(n_windows, dtype=np.float32) * spacing_sec
    return rms, timestamps


def _detect_onsets(
    rms: np.ndarray,
    timestamps: np.ndarray,
    *,
    window_ms: float,
    z_threshold: float,
) -> np.ndarray:
    """Return a boolean array: True where the RMS differential is a real onset."""
    if rms.size < 3:
        return np.zeros(rms.size, dtype=bool)

    # Differential against the value ``lookback`` windows ago (~200 ms).
    lookback = max(1, round(200.0 / window_ms))
    diff = np.zeros_like(rms)
    diff[lookback:] = rms[lookback:] - rms[:-lookback]

    # Robust threshold: only positive jumps count, and they must exceed
    # ``z_threshold`` standard deviations of the full diff distribution.
    sigma = float(diff.std())
    if sigma <= 0:
        return np.zeros(rms.size, dtype=bool)
    flag = diff > z_threshold * sigma

    # Greedy non-maximum suppression within the min separation. Walk
    # forward; whenever we see an onset, skip ahead by min_sep windows.
    min_sep_windows = max(1, round(_ONSET_MIN_SEPARATION_SEC * 1000.0 / window_ms))
    suppressed = flag.copy()
    last_kept = -min_sep_windows - 1
    for i in np.where(flag)[0]:
        if i - last_kept < min_sep_windows:
            # Keep the higher-energy onset in this neighbourhood.
            if rms[i] > rms[last_kept]:
                suppressed[last_kept] = False
                last_kept = int(i)
            else:
                suppressed[i] = False
        else:
            last_kept = int(i)
    # Silence the unused timestamps reference to keep the signature
    # forward-compatible (callers may want timestamps in future variants).
    del timestamps
    return suppressed
