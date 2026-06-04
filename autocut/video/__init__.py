"""Video pipeline: probe, scene detection, frame sampling, keyframe extraction, cutting."""

from autocut.video.compress import CompressionError, compress_for_vlm
from autocut.video.concat import ConcatError, concat_videos
from autocut.video.cutter import CutRequest, CutterError, cut_clip, cut_clips
from autocut.video.frame_sampler import (
    FrameSpec,
    SamplingStrategy,
    build_sampler,
    sample_hybrid,
    sample_scene_based,
    sample_uniform,
)
from autocut.video.keyframes import KeyframeExtractionError, extract_keyframes
from autocut.video.probe import FFprobeError, probe_video
from autocut.video.scene_detect import SceneDetectError, detect_scenes

__all__ = [
    "CompressionError",
    "ConcatError",
    "CutRequest",
    "CutterError",
    "FFprobeError",
    "FrameSpec",
    "KeyframeExtractionError",
    "SamplingStrategy",
    "SceneDetectError",
    "build_sampler",
    "compress_for_vlm",
    "concat_videos",
    "cut_clip",
    "cut_clips",
    "detect_scenes",
    "extract_keyframes",
    "probe_video",
    "sample_hybrid",
    "sample_scene_based",
    "sample_uniform",
]
