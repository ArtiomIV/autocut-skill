"""Unit tests for the file-readiness gate (``autocut wait-ready``)."""

from __future__ import annotations

import pytest

from autocut.video.readiness import (
    ReadinessTimeout,
    ReadyResult,
    file_signature,
    wait_until_ready,
)


class FakeClock:
    """Monotonic clock whose ``sleep`` deterministically advances ``time``."""

    def __init__(self) -> None:
        self.time = 0.0

    def monotonic(self) -> float:
        return self.time

    def sleep(self, seconds: float) -> None:
        self.time += seconds


def _make_signature_sequence(values):
    """Return a ``signature`` callable that yields each value once per call.

    The final value repeats forever (so a "stable" tail can be polled many times).
    """
    seq = list(values)

    def _sig(_path):
        return seq.pop(0) if len(seq) > 1 else seq[0]

    return _sig


def test_ready_immediately_when_already_stable():
    clock = FakeClock()
    # Same signature on every poll → stable.
    result = wait_until_ready(
        "video.mp4",
        stable_for=2.0,
        timeout=30.0,
        poll=1.0,
        signature=lambda _p: (1000, 111),
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    assert isinstance(result, ReadyResult)
    assert result.size_bytes == 1000
    # First poll records the signature; it takes >= stable_for to confirm.
    assert result.waited_sec >= 2.0


def test_waits_while_growing_then_returns_when_size_settles():
    clock = FakeClock()
    # Size grows for three polls, then holds steady.
    sig = _make_signature_sequence([(100, 1), (500, 2), (900, 3), (1000, 4), (1000, 4)])
    result = wait_until_ready(
        "video.mp4",
        stable_for=2.0,
        timeout=60.0,
        poll=1.0,
        signature=sig,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    assert result.size_bytes == 1000


def test_missing_file_times_out():
    clock = FakeClock()
    with pytest.raises(ReadinessTimeout):
        wait_until_ready(
            "nope.mp4",
            stable_for=2.0,
            timeout=10.0,
            poll=1.0,
            signature=lambda _p: None,  # never appears
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )


def test_empty_file_is_not_ready():
    clock = FakeClock()
    with pytest.raises(ReadinessTimeout):
        wait_until_ready(
            "zero.mp4",
            stable_for=2.0,
            timeout=10.0,
            poll=1.0,
            signature=lambda _p: (0, 1),  # exists but 0 bytes → still copying
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )


def test_never_settling_times_out():
    clock = FakeClock()
    # Size changes on every single poll → never stable.
    counter = {"n": 0}

    def _sig(_path):
        counter["n"] += 1
        return (counter["n"] * 100, counter["n"])

    with pytest.raises(ReadinessTimeout):
        wait_until_ready(
            "growing.mp4",
            stable_for=2.0,
            timeout=10.0,
            poll=1.0,
            signature=_sig,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )


def test_file_signature_on_real_file(tmp_path):
    f = tmp_path / "clip.bin"
    f.write_bytes(b"hello world")
    sig = file_signature(f)
    assert sig is not None
    assert sig[0] == 11  # size in bytes


def test_file_signature_missing_returns_none(tmp_path):
    assert file_signature(tmp_path / "absent.bin") is None


def test_file_signature_directory_returns_none(tmp_path):
    # A directory is not a regular file → not "ready".
    assert file_signature(tmp_path) is None


def test_wait_until_ready_on_real_stable_file(tmp_path):
    f = tmp_path / "clip.bin"
    f.write_bytes(b"x" * 2048)
    # Real clock but tiny windows: already stable, returns fast.
    result = wait_until_ready(f, stable_for=0.0, timeout=5.0, poll=0.01)
    assert result.size_bytes == 2048
