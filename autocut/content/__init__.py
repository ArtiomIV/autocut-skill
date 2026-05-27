"""Content-type profiles + auto-detection (Phase E)."""

from autocut.content.detector import (
    DETECTION_KEYFRAME_COUNT,
    DETECTION_KEYFRAME_SUBDIR,
    DetectionContext,
    describe_audio_profile,
    detect_content_hint,
)
from autocut.content.profiles import (
    HYBRID_PROFILE,
    SPORT_PROFILE,
    TALK_PROFILE,
    ContentProfile,
    profile_for,
)

__all__ = [
    "DETECTION_KEYFRAME_COUNT",
    "DETECTION_KEYFRAME_SUBDIR",
    "HYBRID_PROFILE",
    "SPORT_PROFILE",
    "TALK_PROFILE",
    "ContentProfile",
    "DetectionContext",
    "describe_audio_profile",
    "detect_content_hint",
    "profile_for",
]
