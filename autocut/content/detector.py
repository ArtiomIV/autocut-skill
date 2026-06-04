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
- **Vision-only**: classification is purely visual (the stratified
  keyframes). The audio waveform is no longer summarised — sport vs talk
  vs gameplay is obvious from a handful of frames, and dropping the audio
  DSP keeps the detector dependency-light. When v0.2.0 ships Whisper-light
  + Gemini ``input_audio``, audio can re-enter via the forward-compat
  ``DetectionContext`` fields.
- **Confidence threshold**: defined at the call site (pipeline). Below
  threshold → ``ContentHint.other`` → ``HYBRID_PROFILE``. Defensive.

The module exposes a single async entry point ``detect_content_hint``
plus pure helpers for stratified sampling and audio-description rendering
so they can be tested without an ffmpeg run.
"""

from __future__ import annotations

import hashlib
import logging
import random
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from autocut.models import DetectionResult, VideoMetadata
from autocut.video import extract_keyframes
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

# Pad inside each segment by this fraction so a frame never lands right
# at the boundary (avoids black/transition frames common at scene cuts).
_STRATIFIED_SEGMENT_PAD_RATIO: float = 0.1

# Bytes of the source file mixed into the random seed. 1 MiB is enough to
# differentiate any two videos in practice and cheap to read.
_SEED_BYTES_FROM_FILE: int = 1 << 20  # 1 MiB

DETECTION_KEYFRAME_SUBDIR: str = "detection_keyframes"

# Placeholder passed in the (kept) ``audio_description`` slot now that the
# detector is vision-only. Tells the model not to expect an audio summary.
_VISION_ONLY_AUDIO_NOTE: str = "Audio analysis: not provided (vision-only classification)."


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

    # Vision-only classification: no audio waveform summary is sent. The kept
    # ``audio_description`` parameter on ``detect_content`` stays for forward
    # compatibility (Whisper / native audio in v0.2.0) but is empty here.
    result = await provider.detect_content(
        keyframes,
        _VISION_ONLY_AUDIO_NOTE,
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
