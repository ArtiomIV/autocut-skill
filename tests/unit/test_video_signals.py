"""Unit tests for ``autocut.video.signals`` — the audio-peak detector logic.

The peak detector is factored into the pure ``peaks_from_envelope`` so it can be
tested on a synthetic energy envelope with no ffmpeg / video fixture. Scene cuts
reuse the already-tested ``detect_scenes`` wrapper, so they are not re-tested here.
"""

from __future__ import annotations

import numpy as np

from autocut.video.signals import AudioPeak, peaks_from_envelope


def _times(n: int, frame_ms: int = 100) -> np.ndarray:
    return (np.arange(n) + 0.5) * frame_ms / 1000.0


def test_peaks_from_envelope_separates_impact_and_crowd() -> None:
    n = 300  # 30 s at 10 fps
    env = np.full(n, 0.02)
    env[49:51] = 0.5  # narrow burst ~5.0 s  -> impact (0.2 s wide)
    env[200:221] = 0.4  # sustained ~20-22 s -> crowd (2.1 s wide)

    peaks = peaks_from_envelope(_times(n), env, frame_ms=100)

    assert len(peaks) == 2
    impact = next(p for p in peaks if p.t < 10)
    crowd = next(p for p in peaks if p.t >= 10)
    assert impact.kind == "impact"
    assert crowd.kind == "crowd"
    assert 4.5 <= impact.t <= 5.5
    assert 19.5 <= crowd.t <= 22.5
    # The narrow burst is louder -> higher normalised strength.
    assert impact.strength >= crowd.strength
    assert all(0.0 <= p.strength <= 1.0 for p in peaks)


def test_peaks_from_envelope_empty_input() -> None:
    assert peaks_from_envelope(np.empty(0), np.empty(0)) == []


def test_peaks_from_envelope_flat_audio_has_no_peaks() -> None:
    n = 200
    env = np.full(n, 0.1)  # perfectly flat -> zero residual -> nothing to flag
    assert peaks_from_envelope(_times(n), env) == []


def test_peaks_merge_close_bursts_keeping_the_stronger() -> None:
    n = 200
    env = np.full(n, 0.02)
    env[100] = 0.3  # t = 10.05 s
    env[108] = 0.6  # t = 10.85 s, within the 2 s merge gap, stronger
    peaks = peaks_from_envelope(_times(n), env, frame_ms=100, min_gap_sec=2.0)
    assert len(peaks) == 1
    assert peaks[0].strength == 1.0  # the stronger burst won


def test_audio_peak_is_immutable() -> None:
    p = AudioPeak(t=1.0, strength=0.5, kind="impact")
    try:
        p.t = 2.0  # type: ignore[misc]
    except AttributeError:
        return
    raise AssertionError("AudioPeak should be frozen")
