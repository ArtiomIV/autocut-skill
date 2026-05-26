"""Optical-flow motion profile for smart sampling (Phase D.1).

Static-image VLMs can't tell a loaded stance from a real strike — they
see the same pose. Measuring how much pixels move between consecutive
frames fixes this at the source: a real punch shows motion magnitude
spike, a study phase doesn't.

This module produces a ``list[MotionSample]`` with one entry per pair of
consecutive sub-sampled frames. Downstream callers (``hot_windows.py``,
``frame_sampler.py``) consume the profile to decide where to look in
the source video.

Implementation choices:
- Frames are pulled via an ``ffmpeg`` pipe with ``-f rawvideo`` so we
  never write JPEGs to disk and ``ffmpeg`` does the fps decimation and
  downscale in one decode pass. The pixel format is single-channel gray
  to keep the buffer small and skip a colour conversion in numpy.
- The flow algorithm is ``cv2.calcOpticalFlowFarneback`` with default
  pyramid params — cheap, deterministic, well-known. Magnitude is the
  mean Euclidean norm of the per-pixel flow vector field.
- We downscale to a 320 px long edge before flow. Bigger frames don't
  improve the average-magnitude signal materially and cost real time.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class MotionAnalysisError(RuntimeError):
    """Raised when ffmpeg or opencv fail to produce a usable motion profile."""


_DEFAULT_TARGET_FPS: float = 10.0
_DEFAULT_DOWNSCALE_LONG_EDGE: int = 320
# Generous: 10 min of source @10fps -> 6000 frames; Farneback is ~50 ms each on
# CPU, so worst case ~5 min. The cap prevents a stuck ffmpeg from hanging us.
_FFMPEG_TIMEOUT_SEC: int = 600


@dataclass(frozen=True, slots=True)
class MotionSample:
    """One sample of mean optical-flow magnitude at ``timestamp_sec``."""

    timestamp_sec: float
    magnitude: float


def compute_motion_profile(
    video_path: str | Path,
    *,
    target_fps: float = _DEFAULT_TARGET_FPS,
    downscale_long_edge: int = _DEFAULT_DOWNSCALE_LONG_EDGE,
    ffmpeg_path: str | None = None,
) -> list[MotionSample]:
    """Return one ``MotionSample`` per pair of consecutive sub-sampled frames.

    The first frame has no predecessor so the returned list starts at
    ``t = 1 / target_fps``. For a 41 s source at 10 fps this yields ~410
    samples covering the full timeline at 100 ms resolution.

    Raises ``MotionAnalysisError`` if ffmpeg fails, the binary is
    missing, or the source has fewer than two decoded frames.
    """
    if target_fps <= 0:
        raise ValueError("target_fps must be > 0")
    if downscale_long_edge < 32:
        raise ValueError("downscale_long_edge must be at least 32 px")

    video = Path(video_path)
    if not video.is_file():
        raise MotionAnalysisError(f"input video not found: {video}")

    binary = ffmpeg_path or shutil.which("ffmpeg")
    if binary is None:
        raise MotionAnalysisError(
            "ffmpeg not found in PATH; install ffmpeg or pass ffmpeg_path explicitly"
        )

    width, height = _pick_downscale_size(video, binary, downscale_long_edge)
    args = _ffmpeg_pipe_args(binary, video, target_fps=target_fps, width=width, height=height)

    samples: list[MotionSample] = []
    prev_gray: np.ndarray | None = None
    frame_bytes = width * height
    frame_idx = 0

    try:
        proc = subprocess.Popen(  # noqa: S603 - args is a fixed list, no shell
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise MotionAnalysisError(f"failed to invoke ffmpeg: {exc}") from exc

    assert proc.stdout is not None  # for mypy: Popen with PIPE always sets these
    assert proc.stderr is not None

    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            gray = np.frombuffer(raw, dtype=np.uint8).reshape((height, width))
            if prev_gray is not None:
                # cv2's stub disallows ``None`` for the optional ``flow`` arg
                # even though OpenCV documents it. Local mypy sees the stub
                # (opencv-python installed) and pre-commit mypy doesn't, so
                # we silence both: ``call-overload`` for local, ``unused-ignore``
                # for pre-commit.
                flow = cv2.calcOpticalFlowFarneback(  # type: ignore[call-overload, unused-ignore]
                    prev_gray,
                    gray,
                    None,
                    0.5,  # pyr_scale
                    3,  # levels
                    15,  # winsize
                    3,  # iterations
                    5,  # poly_n
                    1.2,  # poly_sigma
                    0,  # flags
                )
                magnitude = float(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean())
                # ``ffmpeg fps=N`` produces a constant-rate output, so the
                # timestamp of frame i is exactly i / N.
                samples.append(
                    MotionSample(timestamp_sec=frame_idx / target_fps, magnitude=magnitude)
                )
            prev_gray = gray
            frame_idx += 1
        # Drain stderr only after stdout is exhausted; otherwise a chatty
        # ffmpeg can deadlock with both pipes full.
        try:
            proc.wait(timeout=_FFMPEG_TIMEOUT_SEC)
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            raise MotionAnalysisError("ffmpeg timed out producing motion frames") from exc
    finally:
        # Best-effort cleanup if we exit via exception mid-loop.
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)

    if proc.returncode != 0:
        stderr_text = proc.stderr.read().decode("utf-8", errors="replace").strip()
        raise MotionAnalysisError(f"ffmpeg exited {proc.returncode}: {stderr_text}")

    if not samples:
        raise MotionAnalysisError(
            f"motion profile empty: fewer than 2 frames decoded from {video.name}"
        )
    return samples


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _pick_downscale_size(
    video: Path,
    ffmpeg_binary: str,
    long_edge: int,
) -> tuple[int, int]:
    """Return the (width, height) we'll downscale to, preserving aspect ratio.

    Uses ``ffprobe`` to read the source dimensions. Dimensions are rounded
    down to the nearest even number — some ffmpeg filters reject odd
    dimensions on yuv4*p pixel formats.
    """
    # ffprobe sits next to ffmpeg under the same WinGet/brew install.
    ffprobe = ffmpeg_binary.replace("ffmpeg", "ffprobe", 1)
    if not Path(ffprobe).is_file():
        # Fallback: try PATH lookup.
        located = shutil.which("ffprobe")
        if located is None:
            raise MotionAnalysisError("ffprobe not found alongside ffmpeg nor in PATH")
        ffprobe = located

    args = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0:s=x",
        str(video),
    ]
    try:
        completed = subprocess.run(  # noqa: S603 - args is a fixed list, no shell
            args,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        raise MotionAnalysisError(f"ffprobe failed to read dimensions: {exc}") from exc
    if completed.returncode != 0:
        raise MotionAnalysisError(
            f"ffprobe failed reading {video.name}: {completed.stderr.strip()}"
        )

    raw = completed.stdout.strip().splitlines()
    if not raw:
        raise MotionAnalysisError(f"ffprobe returned no stream info for {video.name}")
    try:
        src_w_str, src_h_str = raw[0].split("x")
        src_w, src_h = int(src_w_str), int(src_h_str)
    except (ValueError, IndexError) as exc:
        raise MotionAnalysisError(f"unexpected ffprobe output: {raw[0]!r}") from exc

    longer = max(src_w, src_h)
    if longer <= long_edge:
        # Source is already smaller than the target; just round to even.
        return (src_w & ~1, src_h & ~1)
    scale = long_edge / longer
    new_w = max(2, round(src_w * scale) & ~1)
    new_h = max(2, round(src_h * scale) & ~1)
    return new_w, new_h


def _ffmpeg_pipe_args(
    ffmpeg_binary: str,
    video: Path,
    *,
    target_fps: float,
    width: int,
    height: int,
) -> list[str]:
    """Build the ffmpeg command that streams downscaled gray frames to stdout."""
    return [
        ffmpeg_binary,
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vf",
        f"fps={target_fps},scale={width}:{height}",
        "-pix_fmt",
        "gray",
        "-f",
        "rawvideo",
        "-",
    ]
