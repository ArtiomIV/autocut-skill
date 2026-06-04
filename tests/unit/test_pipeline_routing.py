"""Unit tests for the cloud payload route selection in the pipeline."""

from __future__ import annotations

from autocut.models import ContentHint
from autocut.pipeline import Route, _select_route


def _route(
    hint: ContentHint = ContentHint.hybrid,
    *,
    video: bool = False,
    audio: bool = False,
) -> Route:
    return _select_route(hint, supports_video=video, supports_audio=audio)


# ---------------------------------------------------------------------------
# OpenRouter (cloud) transport: audio for talk, else video, else keyframe
# ---------------------------------------------------------------------------


def test_talk_with_audio_takes_audio_route() -> None:
    assert _route(ContentHint.talk, video=True, audio=True) is Route.openrouter_audio


def test_talk_without_audio_falls_back_to_video() -> None:
    assert _route(ContentHint.talk, video=True, audio=False) is Route.openrouter_video


def test_non_talk_takes_video_even_if_audio_capable() -> None:
    # Audio routing is reserved for talk; a highlights clip uses video.
    assert _route(ContentHint.highlights, video=True, audio=True) is Route.openrouter_video


def test_no_direct_support_takes_keyframe() -> None:
    assert _route(ContentHint.hybrid, video=False, audio=False) is Route.keyframe
    # Audio-capable but talk-less + no video still ends up on keyframes.
    assert _route(ContentHint.highlights, video=False, audio=True) is Route.keyframe
