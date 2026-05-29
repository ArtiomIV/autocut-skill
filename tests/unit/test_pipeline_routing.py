"""Unit tests for the payload-x-transport route selection in the pipeline."""

from __future__ import annotations

from autocut.pipeline import Route, _select_route


def test_no_video_support_takes_keyframe_route() -> None:
    # Both host and openrouter fall back to stills when video is unsupported.
    assert _select_route("host", supports_video=False) is Route.keyframe
    assert _select_route("openrouter", supports_video=False) is Route.keyframe


def test_video_capable_host_takes_host_video_route() -> None:
    assert _select_route("host", supports_video=True) is Route.host_video


def test_video_capable_api_takes_openrouter_video_route() -> None:
    # Any non-host provider that reports video support uses the L2 batch loop.
    assert _select_route("openrouter", supports_video=True) is Route.openrouter_video
