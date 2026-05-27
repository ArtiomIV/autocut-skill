"""Auto-detect the content type of a video so the pipeline can pick a profile.

Phase E v0.1.0: the detector samples a small set of keyframes distributed
across the timeline, computes a textual description of the audio waveform,
and asks the VLM to classify the content into one of the ``ContentHint``
categories (``boxing`` / ``sport`` / ``gameplay`` / ``talk`` / ``podcast``
/ ``other``). The pipeline then maps the returned hint to a
``ContentProfile`` and runs the standard highlight extraction with the
right sampling + prompt + duration bounds.

Design choices for v0.1.0:

- **Sampling**: stratified random in N equal segments of the timeline,
  seeded by the SHA-256 of the first 1 MiB of the file. Stratification
  guarantees coverage (no clusters); the file-derived seed guarantees
  determinism (same input → same timestamps) without exposing a
  ``--seed`` knob to the user.
- **Audio signal**: passed to the VLM as a *text description* derived
  from ``compute_audio_profile`` (Phase D). The model does not hear the
  raw audio. This works on every provider (host-agent, openrouter,
  Gemini, GPT-4o, Claude) without forking the code path. When v0.2.0
  ships Whisper-light + Gemini ``input_audio``, the signatures here
  already accept the forward-compat parameters.
- **Confidence threshold**: defined at the call site (pipeline). Below
  threshold → ``ContentHint.other`` → ``HYBRID_PROFILE``. Defensive.

The module exposes a single async entry point ``detect_content_hint``
plus pure helpers for stratified sampling and audio-description rendering
so they can be tested without an ffmpeg run.
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import statistics
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from autocut.models import DetectionResult, VideoMetadata
from autocut.video import AudioSample, compute_audio_profile, extract_keyframes
from autocut.video.audio_peaks import AudioAnalysisError
from autocut.video.frame_sampler import FrameSpec
from autocut.vlm.base import VLMProvider

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunable constants — retunable after empirical E2E on real content
# ---------------------------------------------------------------------------

# 9 keyframes from 9 stratified segments gives a good visual coverage for a
# classification call without bloating the API payload. Sport/talk are
# usually obvious by 3-4 frames; 9 is generous for ambiguous content.
DETECTION_KEYFRAME_COUNT: int = 9

# Voice-activity threshold: an RMS window is "voice-active" when it is in
# the top 50% of the per-clip distribution. Adaptive rather than fixed so
# very quiet sources (interviews in low-noise studios) don't all read as
# silent.
_VOICE_ACTIVITY_QUANTILE: float = 0.5

# Pad inside each segment by this fraction so a frame never lands right
# at the boundary (avoids black/transition frames common at scene cuts).
_STRATIFIED_SEGMENT_PAD_RATIO: float = 0.1

# Bytes of the source file mixed into the random seed. 1 MiB is enough to
# differentiate any two videos in practice and cheap to read.
_SEED_BYTES_FROM_FILE: int = 1 << 20  # 1 MiB

DETECTION_KEYFRAME_SUBDIR: str = "detection_keyframes"


@dataclass(frozen=True, slots=True)
class DetectionContext:
    """Forward-compatible bundle of signals available to the detector.

    Today we populate ``keyframe_specs`` and ``audio_description`` only.
    Future-aware callers can set ``transcript_text`` (Whisper-light in
    v0.2.0) or ``audio_clip_path`` / ``video_clip_paths`` (Gemini
    multimodal input when the model declares the capability). Providers
    receive the bundle's primitives via ``detect_content`` kwargs.
    """

    keyframe_specs: list[FrameSpec]
    audio_description: str
    transcript_text: str | None = None
    audio_clip_path: Path | None = None
    video_clip_paths: list[Path] | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def detect_content_hint(
    video: Path,
    provider: VLMProvider,
    *,
    metadata: VideoMetadata,
    output_root: Path,
    n_keyframes: int = DETECTION_KEYFRAME_COUNT,
    long_edge_px: int = 768,
) -> DetectionResult:
    """Run the detection pre-step and return the classified content hint.

    Side effects:
    - extracts ``n_keyframes`` JPEGs under
      ``output_root / DETECTION_KEYFRAME_SUBDIR``
    - reads the source audio waveform (one ffmpeg pipe)
    - one provider call (real for openrouter, stub for host-agent)

    Raises ``VLMError`` from the underlying provider on cloud failures;
    audio extraction failures are non-fatal (we log and fall back to a
    "no audio available" description).
    """
    if n_keyframes < 3:
        raise ValueError(
            f"n_keyframes must be at least 3 for meaningful coverage; got {n_keyframes}"
        )

    timestamps = _pick_stratified_timestamps(
        duration_sec=metadata.duration_sec,
        n=n_keyframes,
        video_path=video,
    )
    log.info(
        "detector: stratified sampling %d frame(s) across %.1fs",
        n_keyframes,
        metadata.duration_sec,
    )

    specs = [FrameSpec(timestamp=timedelta(seconds=ts), scene_index=-1) for ts in timestamps]
    keyframe_dir = output_root / DETECTION_KEYFRAME_SUBDIR
    keyframes = extract_keyframes(
        video,
        specs,
        keyframe_dir,
        long_edge_px=long_edge_px,
    )

    audio_description = _try_describe_audio(video, duration_sec=metadata.duration_sec)
    log.debug("detector: audio description ready (%d chars)", len(audio_description))

    result = await provider.detect_content(
        keyframes,
        audio_description,
        video_id=video.stem or "video",
        duration_sec=metadata.duration_sec,
    )
    log.info(
        "detector: provider returned hint=%s confidence=%.2f reason=%r",
        result.content_hint.value,
        result.confidence,
        result.reasoning,
    )
    return result


# ---------------------------------------------------------------------------
# Stratified random timestamp picker (pure, testable)
# ---------------------------------------------------------------------------


def _pick_stratified_timestamps(
    *,
    duration_sec: float,
    n: int,
    video_path: Path,
) -> list[float]:
    """Pick ``n`` timestamps using stratified random with a file-derived seed.

    Algorithm:
    1. Split [0, duration] into ``n`` equal segments.
    2. In each segment pick a random offset, padded inward by
       ``_STRATIFIED_SEGMENT_PAD_RATIO`` to avoid landing on segment
       boundaries (often black frames / cuts).
    3. Seed the RNG with the SHA-256 of the file's first MiB so two runs
       on the same file produce the same timestamps (deterministic for
       reproducibility) but different files produce different timestamps.

    Returns timestamps strictly inside ``[0, duration]``. List length is
    exactly ``n``.
    """
    if duration_sec <= 0:
        raise ValueError("duration_sec must be positive")
    if n < 1:
        raise ValueError("n must be at least 1")

    seed = _file_hash_seed(video_path)
    # ``random.Random`` is fine here — the picker is for deterministic
    # sampling reproducibility, never for cryptographic guarantees.
    rng = random.Random(seed)  # noqa: S311 - not security-sensitive

    segment_len = duration_sec / n
    pad = segment_len * _STRATIFIED_SEGMENT_PAD_RATIO
    timestamps: list[float] = []
    for i in range(n):
        low = i * segment_len + pad
        high = (i + 1) * segment_len - pad
        if high <= low:
            # Degenerate case (very short video, very many segments):
            # fall back to the midpoint of the segment.
            timestamps.append(i * segment_len + segment_len / 2.0)
            continue
        timestamps.append(rng.uniform(low, high))
    return timestamps


def _file_hash_seed(video_path: Path) -> int:
    """Derive a 64-bit RNG seed from the first MiB of the file's bytes.

    Reading just the head keeps the seed cheap to compute even for
    multi-GB videos. The first MiB always includes the container
    header + initial frame bytes, which differ between videos.
    Returns 0 if the file cannot be read — the RNG still picks valid
    timestamps, just less varied across runs.
    """
    h = hashlib.sha256()
    try:
        with video_path.open("rb") as f:
            h.update(f.read(_SEED_BYTES_FROM_FILE))
    except OSError as exc:
        log.warning("detector: failed to hash %s for seed: %s", video_path, exc)
        return 0
    return int.from_bytes(h.digest()[:8], "big", signed=False)


# ---------------------------------------------------------------------------
# Audio description renderer (pure, testable)
# ---------------------------------------------------------------------------


def _try_describe_audio(video: Path, *, duration_sec: float) -> str:
    """Compute the audio profile and render the textual description.

    Audio failures are converted to a clear "no audio" description so the
    detector still gets a usable text block. Detection works visually on
    silent footage (gameplay screen recordings, music videos with the
    audio stripped) — losing audio is degraded mode, not a fatal error.
    """
    try:
        samples = compute_audio_profile(video)
    except AudioAnalysisError as exc:
        log.info("detector: audio profile unavailable (%s); falling back to vision-only", exc)
        return "Audio analysis: source has no usable audio stream (vision-only classification)."
    return describe_audio_profile(samples, duration_sec=duration_sec)


def describe_audio_profile(samples: list[AudioSample], *, duration_sec: float) -> str:
    """Render a ``compute_audio_profile`` output as a short text block for the VLM.

    Public so tests can pin formatting and so future callers (CLI debug
    output) can render the same description without re-running ffmpeg.
    """
    if not samples:
        return "Audio analysis: source has no audio stream (vision-only classification)."

    rms_values = [s.rms for s in samples]
    onset_count = sum(1 for s in samples if s.is_onset)
    onset_rate_per_min = onset_count / max(duration_sec, 1e-6) * 60.0

    # Voice-activity ratio: fraction of windows above the median RMS,
    # weighted to suppress tiny background noise. Adaptive so quiet
    # recordings still show non-zero activity if there's any signal.
    if rms_values:
        threshold = statistics.quantiles(rms_values, n=4)[1] if len(rms_values) >= 4 else 0.0
        threshold = max(threshold, _VOICE_ACTIVITY_QUANTILE * max(rms_values))
        voice_active = sum(1 for r in rms_values if r > threshold)
        voice_ratio = voice_active / len(rms_values)
    else:
        voice_ratio = 0.0

    mean_rms = statistics.fmean(rms_values) if rms_values else 0.0
    max_rms = max(rms_values) if rms_values else 0.0
    min_rms_positive = min((r for r in rms_values if r > 0), default=0.0)
    if max_rms > 0 and min_rms_positive > 0:
        dynamic_range_db = 20.0 * math.log10(max_rms / min_rms_positive)
    else:
        dynamic_range_db = 0.0

    # Interpret the metrics into human-readable adjectives so the VLM
    # gets a strong textual signal even when the raw numbers look dry.
    onset_label = _classify_onset_rate(onset_rate_per_min)
    voice_label = _classify_voice_ratio(voice_ratio)

    return (
        "Audio analysis (waveform statistics, not transcribed):\n"
        f"- Voice activity: {voice_label} (ratio {voice_ratio:.2f})\n"
        f"- Onset rate: {onset_label} ({onset_rate_per_min:.1f} events/min, "
        f"{onset_count} total)\n"
        f"- RMS dynamic range: {dynamic_range_db:.1f} dB\n"
        f"- Mean energy: {mean_rms:.4f}, peak energy: {max_rms:.4f}"
    )


def _classify_onset_rate(rate_per_min: float) -> str:
    """Map an onset-per-minute rate to a human label.

    Tuned on real videos in Phase D: boxing match shows ~25-40 onsets/min
    (impacts + crowd reactions + commentator), a podcast shows ~5-8
    onsets/min (turn-taking pauses + laughter), a monologue ~2-4
    onsets/min.
    """
    if rate_per_min >= 20.0:
        return "high (impact-heavy or applause-rich)"
    if rate_per_min >= 8.0:
        return "moderate (conversational with reactions)"
    if rate_per_min >= 2.0:
        return "low (sparse, monologue-like)"
    return "very low (near-silent or sustained tone)"


def _classify_voice_ratio(ratio: float) -> str:
    """Map the voice-activity ratio to a human label."""
    if ratio >= 0.7:
        return "dominant (sustained speech/audio across most of the timeline)"
    if ratio >= 0.4:
        return "frequent (alternating speech and pauses)"
    if ratio >= 0.15:
        return "intermittent (bursts of activity)"
    return "sparse (mostly quiet)"
