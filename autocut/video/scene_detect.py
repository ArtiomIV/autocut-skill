"""Scene detection backed by PySceneDetect.

We use ``ContentDetector``, which compares consecutive frames and flags a
scene boundary when the visual difference exceeds ``threshold`` (default
27.0, recommended by the library for general content).

Cutting the video into scene-bounded chunks BEFORE keyframe extraction is
how we keep VLM token cost down: a 30-min video typically has ~150 scenes;
sending one keyframe per scene means ~150 frames at the VLM instead of
54000 frames at 30 fps.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from scenedetect import ContentDetector, SceneManager, open_video

from autocut.models import Scene


class SceneDetectError(RuntimeError):
    """Raised when PySceneDetect cannot open or process the video."""


_DEFAULT_THRESHOLD = 27.0
_DEFAULT_MIN_SCENE_LEN_FRAMES = 15  # ~0.5s at 30 fps; filters spurious cuts


def detect_scenes(
    video_path: str | Path,
    *,
    threshold: float = _DEFAULT_THRESHOLD,
    min_scene_len_frames: int = _DEFAULT_MIN_SCENE_LEN_FRAMES,
) -> list[Scene]:
    """Detect scene boundaries in ``video_path`` and return them as ``Scene`` objects.

    If the detector finds no boundaries (very short or visually uniform video),
    a single ``Scene`` covering the full duration is returned instead — so
    downstream stages always receive at least one segment.
    """
    path = Path(video_path)
    if not path.is_file():
        raise SceneDetectError(f"input file does not exist: {path}")

    try:
        video_stream = open_video(str(path))
    except Exception as exc:
        raise SceneDetectError(f"failed to open video for scene detection: {exc}") from exc

    detector = ContentDetector(threshold=threshold, min_scene_len=min_scene_len_frames)
    manager = SceneManager()
    manager.add_detector(detector)

    try:
        manager.detect_scenes(video=video_stream, show_progress=False)
        raw_scenes = manager.get_scene_list()
    except Exception as exc:
        raise SceneDetectError(f"scene detection failed: {exc}") from exc

    if not raw_scenes:
        full_duration = video_stream.duration.seconds
        if full_duration <= 0:
            raise SceneDetectError(
                "scene detection returned no scenes and video duration is unknown"
            )
        return [
            Scene(
                index=0,
                start=timedelta(),
                end=timedelta(seconds=full_duration),
            )
        ]

    scenes: list[Scene] = []
    for index, (start_tc, end_tc) in enumerate(raw_scenes):
        scenes.append(
            Scene(
                index=index,
                start=timedelta(seconds=start_tc.seconds),
                end=timedelta(seconds=end_tc.seconds),
            )
        )
    return scenes
