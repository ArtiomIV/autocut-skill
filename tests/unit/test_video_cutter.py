"""Unit tests for ``autocut.video.cutter`` — request validation and arg generation."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from autocut.video.cutter import CutRequest, _build_args, _format_ts, expand_request


def _req(start: float, end: float, path: Path) -> CutRequest:
    return CutRequest(start=timedelta(seconds=start), end=timedelta(seconds=end), output_path=path)


# ---------------------------------------------------------------------------
# CutRequest validation
# ---------------------------------------------------------------------------


def test_cutrequest_rejects_end_before_start(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="end must be strictly greater than start"):
        _req(5, 3, tmp_path / "out.mp4")


def test_cutrequest_rejects_zero_length_clip(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _req(5, 5, tmp_path / "out.mp4")


def test_cutrequest_is_immutable(tmp_path: Path) -> None:
    req = _req(0, 5, tmp_path / "out.mp4")
    with pytest.raises(AttributeError):
        req.start = timedelta(seconds=99)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _format_ts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ts", "expected"),
    [
        (timedelta(), "00:00:00.000"),
        (timedelta(seconds=5), "00:00:05.000"),
        (timedelta(seconds=12.345), "00:00:12.345"),
        (timedelta(hours=1, minutes=2, seconds=3), "01:02:03.000"),
        (timedelta(seconds=-1), "00:00:00.000"),  # clamped to zero
    ],
)
def test_format_ts(ts: timedelta, expected: str) -> None:
    assert _format_ts(ts) == expected


# ---------------------------------------------------------------------------
# _build_args (no subprocess invocation)
# ---------------------------------------------------------------------------


def test_streamcopy_args_use_ss_before_input(tmp_path: Path) -> None:
    req = _req(10, 15, tmp_path / "out.mp4")
    args = _build_args("/fake/ffmpeg", Path("in.mp4"), req, accurate=False)
    # In stream-copy mode -ss must come BEFORE -i for fast seek.
    ss_pos = args.index("-ss")
    i_pos = args.index("-i")
    assert ss_pos < i_pos
    assert "-c" in args and args[args.index("-c") + 1] == "copy"


def test_accurate_args_use_ss_after_input(tmp_path: Path) -> None:
    req = _req(10, 15, tmp_path / "out.mp4")
    args = _build_args("/fake/ffmpeg", Path("in.mp4"), req, accurate=True)
    # In accurate mode -ss must come AFTER -i so ffmpeg decodes from the start.
    ss_pos = args.index("-ss")
    i_pos = args.index("-i")
    assert ss_pos > i_pos
    assert "libx264" in args


def test_args_include_avoid_negative_ts(tmp_path: Path) -> None:
    req = _req(0, 5, tmp_path / "out.mp4")
    args = _build_args("/fake/ffmpeg", Path("in.mp4"), req, accurate=False)
    assert "-avoid_negative_ts" in args


def test_args_never_use_shell_metacharacters(tmp_path: Path) -> None:
    # Sanity check: every element must be a string with no shell pipes/redirects.
    # (Defence in depth — we use args=[list] and shell=False, but verify.)
    req = _req(0, 5, tmp_path / "out file with spaces.mp4")
    args = _build_args("/fake/ffmpeg", Path("in put.mp4"), req, accurate=False)
    forbidden = {"|", ";", "&", ">", "<", "$(", "`"}
    for arg in args:
        assert not any(token in arg for token in forbidden), f"shell metacharacter in {arg!r}"


# ---------------------------------------------------------------------------
# expand_request — pre/post-roll plumbing (A.5.3)
# ---------------------------------------------------------------------------


def test_expand_request_is_noop_when_padding_is_zero(tmp_path: Path) -> None:
    original = _req(5, 10, tmp_path / "out.mp4")
    expanded = expand_request(
        original, pre_roll_sec=0.0, post_roll_sec=0.0, video_duration_sec=60.0
    )
    assert expanded is original  # identity, not just equal


def test_expand_request_widens_both_bounds(tmp_path: Path) -> None:
    original = _req(10, 20, tmp_path / "out.mp4")
    expanded = expand_request(
        original, pre_roll_sec=1.5, post_roll_sec=0.5, video_duration_sec=60.0
    )
    assert expanded.start == timedelta(seconds=8.5)
    assert expanded.end == timedelta(seconds=20.5)
    assert expanded.output_path == original.output_path


def test_expand_request_clamps_start_to_zero(tmp_path: Path) -> None:
    original = _req(0.5, 5, tmp_path / "out.mp4")
    expanded = expand_request(
        original, pre_roll_sec=2.0, post_roll_sec=0.0, video_duration_sec=60.0
    )
    assert expanded.start == timedelta(0)
    assert expanded.end == timedelta(seconds=5)


def test_expand_request_clamps_end_to_duration(tmp_path: Path) -> None:
    original = _req(50, 58, tmp_path / "out.mp4")
    expanded = expand_request(
        original, pre_roll_sec=0.0, post_roll_sec=5.0, video_duration_sec=60.0
    )
    assert expanded.start == timedelta(seconds=50)
    assert expanded.end == timedelta(seconds=60)


def test_expand_request_rejects_negative_padding(tmp_path: Path) -> None:
    original = _req(10, 20, tmp_path / "out.mp4")
    with pytest.raises(ValueError, match="non-negative"):
        expand_request(
            original, pre_roll_sec=-1.0, post_roll_sec=0.0, video_duration_sec=60.0
        )
    with pytest.raises(ValueError, match="non-negative"):
        expand_request(
            original, pre_roll_sec=0.0, post_roll_sec=-1.0, video_duration_sec=60.0
        )
