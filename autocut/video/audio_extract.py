"""Extract a compact audio track from a video for audio-input VLM analysis.

Audio-capable models (Gemini et al.) hear speech directly, so for talk/podcast
content we send the audio instead of frames — the signal is in the words, not
the pixels. This module pulls the audio out of the source and re-encodes it to a
small mono, low-bitrate MP3: speech is mono and narrowband, so a high-quality
track is wasted bytes. At 32 kbps mono a 30-minute clip is ~7 MB (~9.5 MB
base64), comfortably under the ~20 MB inline request ceiling, so a half-hour fits
in a single API call.

Validated format (probe 2026-05-30): OpenRouter -> Gemini accepts an
``input_audio`` block with ``format: "mp3"`` on the normal route (no Vertex pin,
unlike video).
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from autocut.video.ffmpeg_path import FFmpegResolveError, ffmpeg_binary

log = logging.getLogger(__name__)


class AudioExtractionError(RuntimeError):
    """Raised when ffmpeg fails to produce the extracted audio output."""


# Mono, 16 kHz, 32 kbps MP3: matched to speech. ~7 MB / 30 min.
_DEFAULT_SAMPLE_RATE_HZ = 16_000
_DEFAULT_BITRATE = "32k"
# Extraction is fast (no video re-encode), but a long source still streams the
# whole file; keep a generous ceiling.
_DEFAULT_TIMEOUT_SEC = 900


def extract_audio_for_vlm(
    src: Path,
    output: Path,
    *,
    sample_rate_hz: int = _DEFAULT_SAMPLE_RATE_HZ,
    bitrate: str = _DEFAULT_BITRATE,
    ffmpeg_path: str | None = None,
    timeout_sec: int = _DEFAULT_TIMEOUT_SEC,
) -> Path:
    """Extract ``src``'s audio to a small mono MP3 at ``output``. Returns ``output``.

    Drops the video stream, downmixes to mono, resamples to ``sample_rate_hz``,
    and encodes MP3 at ``bitrate``. Raises ``AudioExtractionError`` if the source
    is missing, has no audio, ffmpeg cannot be resolved, or ffmpeg exits
    non-zero.
    """
    if not src.is_file():
        raise AudioExtractionError(f"source file not found: {src}")
    if sample_rate_hz <= 0:
        raise AudioExtractionError("sample_rate_hz must be > 0")

    try:
        binary = ffmpeg_binary(ffmpeg_path)
    except FFmpegResolveError as exc:
        raise AudioExtractionError(f"ffmpeg not found: {exc}") from exc

    output.parent.mkdir(parents=True, exist_ok=True)

    args = _build_args(binary, src, output, sample_rate_hz, bitrate)
    try:
        completed = subprocess.run(  # noqa: S603 - args is a fixed list, no shell
            args,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        raise AudioExtractionError(f"failed to invoke ffmpeg: {exc}") from exc
    if completed.returncode != 0 or not output.is_file():
        raise AudioExtractionError(
            f"ffmpeg audio extraction failed (source may have no audio stream): "
            f"{completed.stderr.strip()}"
        )

    log.info(
        "audio extract: %s -> %s (mono, %d Hz, %s)",
        src.name,
        output.name,
        sample_rate_hz,
        bitrate,
    )
    return output


def _build_args(
    binary: str,
    src: Path,
    output: Path,
    sample_rate_hz: int,
    bitrate: str,
) -> list[str]:
    """Assemble the ffmpeg argument list (no shell, fixed list)."""
    return [
        binary,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vn",  # drop the video stream
        "-ac",
        "1",  # mono
        "-ar",
        str(sample_rate_hz),
        "-c:a",
        "libmp3lame",
        "-b:a",
        bitrate,
        str(output),
    ]
