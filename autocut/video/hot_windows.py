"""Combine motion + audio profiles into time windows worth dense sampling.

A "hot window" is a chunk of the timeline where *something interesting*
is happening — defined as either:

- Sustained high optical-flow magnitude (a fight exchange, a chase,
  fast camera/subject movement), OR
- A clustered audio onset (impact, commentator yell, applause, laughter).

The sampler in ``frame_sampler.py`` consumes these windows: it samples
sparsely outside them and densely inside, so the VLM sees the bursty
moments at fine granularity without paying for every dead second.

The default thresholds are tuned for sport content. Each ``ContentProfile``
can override them via the ``HotWindowConfig`` passed to ``find_hot_windows``;
talk/podcast profiles use lower motion thresholds and longer windows.

Note: audio onsets often *trail* the visual event in commentary (the
caller yells 1-2 s after the punch). We compensate by widening windows
back in time by ``audio_lookback_sec`` so the punch frame itself ends up
inside the window even when only the audio onset fired.
"""

from __future__ import annotations

from dataclasses import dataclass

from autocut.video.audio_peaks import AudioSample
from autocut.video.motion import MotionSample


@dataclass(frozen=True, slots=True)
class HotWindow:
    """A time interval the sampler should treat as dense-worthy."""

    start_sec: float
    end_sec: float
    score: float  # Higher = stronger signal; for ranking when budget is tight.

    def __post_init__(self) -> None:
        if self.end_sec <= self.start_sec:
            raise ValueError("HotWindow end must be strictly greater than start")

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


@dataclass(frozen=True, slots=True)
class HotWindowConfig:
    """Per-call tuning. Profile-aware callers populate this from ``ContentProfile``."""

    # Motion: a sliding window is "hot" if its mean magnitude exceeds
    # ``motion_z_threshold * overall_std + overall_mean``. Higher = pickier.
    motion_z_threshold: float = 1.0
    # Width of the motion sliding window in seconds. Sport: 1-2 s; talk: 5-10 s.
    motion_window_sec: float = 2.0
    # How much the motion window steps forward each iteration. Smaller =
    # more precision, more compute. 0.5 s is a good default.
    motion_stride_sec: float = 0.5
    # Each audio onset spawns a window of this width centred on the onset
    # minus ``audio_lookback_sec`` (to capture the visual cause of the
    # commentator reaction).
    audio_window_sec: float = 2.0
    audio_lookback_sec: float = 1.0
    # Hot windows whose gap is below this collapse into one. Avoids
    # chopping a continuous exchange into many short fragments.
    merge_gap_sec: float = 1.0


def find_hot_windows(
    motion: list[MotionSample],
    audio: list[AudioSample],
    *,
    video_duration_sec: float,
    config: HotWindowConfig | None = None,
) -> list[HotWindow]:
    """Return the merged, sorted list of hot windows from motion + audio signals.

    Either signal can be empty:
    - No motion samples → only audio-driven windows are produced.
    - No audio samples → only motion-driven windows are produced.
    - Both empty → returns an empty list.

    Windows are clamped to ``[0, video_duration_sec]`` and any with
    non-positive duration after clamping are dropped.
    """
    if video_duration_sec <= 0:
        raise ValueError("video_duration_sec must be > 0")
    cfg = config or HotWindowConfig()

    raw_windows: list[HotWindow] = []
    raw_windows.extend(_motion_windows(motion, cfg))
    raw_windows.extend(_audio_windows(audio, cfg))

    if not raw_windows:
        return []

    # Two-step filter so mypy sees a list[HotWindow] (not list[HotWindow | None]).
    clamped: list[HotWindow] = []
    for w in raw_windows:
        kept = _clamp(w, video_duration_sec)
        if kept is not None:
            clamped.append(kept)
    if not clamped:
        return []

    clamped.sort(key=lambda w: w.start_sec)
    return _merge_adjacent(clamped, merge_gap_sec=cfg.merge_gap_sec)


# ---------------------------------------------------------------------------
# Internals — motion
# ---------------------------------------------------------------------------


def _motion_windows(motion: list[MotionSample], cfg: HotWindowConfig) -> list[HotWindow]:
    if len(motion) < 2:
        return []

    # Build a flat magnitude array; the timestamps come back from the
    # original samples so we don't lose precision when stepping.
    magnitudes = [s.magnitude for s in motion]
    overall_mean = sum(magnitudes) / len(magnitudes)
    overall_std = _std(magnitudes, mean=overall_mean)
    if overall_std <= 0:
        return []  # constant motion → no signal

    threshold = overall_mean + cfg.motion_z_threshold * overall_std

    # Resolve window/stride into sample counts using the inter-sample delta.
    # Motion samples are produced at constant fps so the delta is stable.
    sample_dt = motion[1].timestamp_sec - motion[0].timestamp_sec
    if sample_dt <= 0:
        return []
    window_samples = max(2, round(cfg.motion_window_sec / sample_dt))
    stride_samples = max(1, round(cfg.motion_stride_sec / sample_dt))

    windows: list[HotWindow] = []
    i = 0
    while i + window_samples <= len(motion):
        chunk = magnitudes[i : i + window_samples]
        chunk_mean = sum(chunk) / window_samples
        if chunk_mean > threshold:
            start = motion[i].timestamp_sec
            end = motion[i + window_samples - 1].timestamp_sec
            # Score: how far above threshold we are, normalised. Useful
            # for downstream ranking when the caller has a sample budget.
            score = (chunk_mean - overall_mean) / overall_std
            windows.append(HotWindow(start_sec=start, end_sec=end, score=score))
        i += stride_samples
    return windows


def _std(values: list[float], *, mean: float) -> float:
    if len(values) < 2:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return float(variance**0.5)


# ---------------------------------------------------------------------------
# Internals — audio
# ---------------------------------------------------------------------------


def _audio_windows(audio: list[AudioSample], cfg: HotWindowConfig) -> list[HotWindow]:
    if not audio:
        return []
    half = cfg.audio_window_sec / 2.0
    windows: list[HotWindow] = []
    for sample in audio:
        if not sample.is_onset:
            continue
        # Centre the window on (onset - lookback), so most of it covers
        # the time BEFORE the onset (the visual cause).
        centre = sample.timestamp_sec - cfg.audio_lookback_sec
        start = centre - half
        end = centre + half
        # Score: use the onset's own RMS as a proxy for "loudness".
        windows.append(HotWindow(start_sec=start, end_sec=end, score=sample.rms))
    return windows


# ---------------------------------------------------------------------------
# Internals — clamp + merge
# ---------------------------------------------------------------------------


def _clamp(window: HotWindow, video_duration_sec: float) -> HotWindow | None:
    start = max(0.0, window.start_sec)
    end = min(video_duration_sec, window.end_sec)
    if end <= start:
        return None
    return HotWindow(start_sec=start, end_sec=end, score=window.score)


def _merge_adjacent(windows: list[HotWindow], *, merge_gap_sec: float) -> list[HotWindow]:
    """Merge windows whose gap is below ``merge_gap_sec``. Sums the scores."""
    merged: list[HotWindow] = []
    for current in windows:
        if not merged:
            merged.append(current)
            continue
        last = merged[-1]
        if current.start_sec - last.end_sec <= merge_gap_sec:
            merged[-1] = HotWindow(
                start_sec=last.start_sec,
                end_sec=max(last.end_sec, current.end_sec),
                score=last.score + current.score,
            )
        else:
            merged.append(current)
    return merged
