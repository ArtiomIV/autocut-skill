"""End-to-end integration tests for the M2 video pipeline.

These tests invoke real ffmpeg + PySceneDetect, so they're marked as
``integration`` and skipped automatically when ffmpeg is missing.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from autocut.video import (
    CutRequest,
    build_sampler,
    cut_clips,
    detect_scenes,
    extract_keyframes,
    probe_video,
)

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------


def test_probe_reports_real_metadata(medium_video: Path) -> None:
    meta = probe_video(medium_video)
    assert meta.width == 640
    assert meta.height == 480
    assert meta.duration_sec == pytest.approx(10.0, abs=0.5)
    assert meta.video_codec == "h264"
    assert meta.fps == pytest.approx(30.0, abs=0.1)


# ---------------------------------------------------------------------------
# scene detection
# ---------------------------------------------------------------------------


def test_scene_detection_falls_back_to_single_scene_on_uniform_video(
    short_video: Path,
) -> None:
    # testsrc is visually uniform -> ContentDetector finds nothing -> fallback.
    scenes = detect_scenes(short_video)
    assert len(scenes) == 1
    assert scenes[0].index == 0
    assert scenes[0].start == timedelta()
    assert scenes[0].duration_sec == pytest.approx(4.0, abs=0.5)


# ---------------------------------------------------------------------------
# Full chain: probe -> scenes -> sampler -> keyframes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["scene", "uniform", "hybrid"])
def test_full_chain_produces_jpeg_keyframes(
    medium_video: Path, tmp_path: Path, strategy: str
) -> None:
    meta = probe_video(medium_video)
    scenes = detect_scenes(medium_video)
    specs = build_sampler(
        strategy,  # type: ignore[arg-type]
        scenes,
        meta.duration_sec,
        per_scene=2,
        interval_sec=2.0,
        max_gap_sec=3.0,
    )
    assert len(specs) > 0

    kf_dir = tmp_path / "kfs"
    keyframes = extract_keyframes(medium_video, specs, kf_dir, long_edge_px=256)

    assert len(keyframes) == len(specs)
    for kf in keyframes:
        assert kf.path.is_file()
        assert kf.path.suffix == ".jpg"
        # Each JPEG is non-empty and not absurdly large for 256px.
        size = kf.path.stat().st_size
        assert 500 < size < 200_000


def test_uniform_sampling_produces_expected_count(medium_video: Path, tmp_path: Path) -> None:
    # 10s video, interval 2s, first sample at 1s -> 5 samples expected.
    meta = probe_video(medium_video)
    specs = build_sampler("uniform", [], meta.duration_sec, interval_sec=2.0)
    assert len(specs) == 5
    kfs = extract_keyframes(medium_video, specs, tmp_path / "uniform", long_edge_px=256)
    assert len(kfs) == 5
    assert all(kf.scene_index is None for kf in kfs)


def test_hybrid_on_short_video_produces_enough_keyframes(short_video: Path, tmp_path: Path) -> None:
    # 4s video with hybrid: the short-video reroute must give us enough
    # keyframes for the VLM. Pre-A.2 this could degenerate to 1-2 samples.
    meta = probe_video(short_video)
    scenes = detect_scenes(short_video)
    specs = build_sampler("hybrid", scenes, meta.duration_sec)
    assert len(specs) >= 3
    kfs = extract_keyframes(short_video, specs, tmp_path / "short", long_edge_px=256)
    assert len(kfs) >= 3


# ---------------------------------------------------------------------------
# cutter
# ---------------------------------------------------------------------------


def test_cut_clips_accurate_mode_produces_exact_durations(
    medium_video: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "clips"
    out_dir.mkdir()
    requests = [
        CutRequest(
            start=timedelta(seconds=1),
            end=timedelta(seconds=3),
            output_path=out_dir / "clip_1.mp4",
        ),
        CutRequest(
            start=timedelta(seconds=4),
            end=timedelta(seconds=7),
            output_path=out_dir / "clip_2.mp4",
        ),
    ]
    paths = cut_clips(medium_video, requests, accurate=True)
    assert len(paths) == 2

    meta1 = probe_video(paths[0])
    meta2 = probe_video(paths[1])
    # Accurate mode is frame-accurate; allow ±100 ms tolerance for muxer overhead.
    assert meta1.duration_sec == pytest.approx(2.0, abs=0.1)
    assert meta2.duration_sec == pytest.approx(3.0, abs=0.1)


def test_cut_clips_stream_copy_mode_runs(medium_video: Path, tmp_path: Path) -> None:
    out = tmp_path / "stream_copy.mp4"
    requests = [CutRequest(start=timedelta(seconds=2), end=timedelta(seconds=5), output_path=out)]
    paths = cut_clips(medium_video, requests, accurate=False)
    assert paths[0].is_file()
    # Stream-copy snaps to keyframes; we built the fixture with -g 30 so
    # cuts should land within 1 second of the requested boundary.
    meta = probe_video(paths[0])
    assert meta.duration_sec == pytest.approx(3.0, abs=1.1)
