"""Video pipeline: probe, scene detection, frame sampling, keyframe extraction, cutting."""

from autocut.video.audio_peaks import (
    AudioAnalysisError,
    AudioSample,
    compute_audio_profile,
)
from autocut.video.compress import CompressionError, compress_for_vlm
from autocut.video.concat import ConcatError, concat_videos
from autocut.video.cutter import CutRequest, CutterError, cut_clip, cut_clips
from autocut.video.frame_sampler import (
    FrameSpec,
    SamplingStrategy,
    build_sampler,
    sample_hybrid,
    sample_motion,
    sample_scene_based,
    sample_uniform,
)
from autocut.video.hot_windows import HotWindow, HotWindowConfig, find_hot_windows
from autocut.video.keyframes import KeyframeExtractionError, extract_keyframes
from autocut.video.motion import MotionAnalysisError, MotionSample, compute_motion_profile
from autocut.video.probe import FFprobeError, probe_video
from autocut.video.scene_detect import SceneDetectError, detect_scenes

__all__ = [
    "AudioAnalysisError",
    "AudioSample",
    "CompressionError",
    "ConcatError",
    "CutRequest",
    "CutterError",
    "FFprobeError",
    "FrameSpec",
    "HotWindow",
    "HotWindowConfig",
    "KeyframeExtractionError",
    "MotionAnalysisError",
    "MotionSample",
    "SamplingStrategy",
    "SceneDetectError",
    "build_sampler",
    "compress_for_vlm",
    "compute_audio_profile",
    "compute_motion_profile",
    "concat_videos",
    "cut_clip",
    "cut_clips",
    "detect_scenes",
    "extract_keyframes",
    "find_hot_windows",
    "probe_video",
    "sample_hybrid",
    "sample_motion",
    "sample_scene_based",
    "sample_uniform",
]
