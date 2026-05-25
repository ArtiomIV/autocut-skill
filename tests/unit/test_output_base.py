"""Unit tests for ``autocut.output.base`` — the shared slugify helper."""

from __future__ import annotations

import pytest

from autocut.output.base import slugify


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("simple", "simple"),
        ("Two Words", "Two-Words"),
        ("MiXeD Case 123", "MiXeD-Case-123"),
        ("trailing-dashes---", "trailing-dashes"),
        ("---leading-dashes", "leading-dashes"),
        ("multiple   spaces", "multiple-spaces"),
    ],
)
def test_slugify_basic(text: str, expected: str) -> None:
    assert slugify(text) == expected


def test_slugify_strips_path_separators() -> None:
    # The VLM is untrusted: any "../" or "/etc/passwd" must collapse to dashes.
    assert ".." not in slugify("../../etc/passwd")
    assert "/" not in slugify("foo/bar")
    assert "\\" not in slugify(r"foo\bar")


def test_slugify_strips_unicode() -> None:
    # Non-ASCII (emoji, accents, arrows) is replaced with dashes — keeping the
    # filename predictable on every OS.
    result = slugify("café → 🎬 boxing")
    assert all(c.isascii() for c in result)


def test_slugify_handles_empty_input() -> None:
    assert slugify("") == "clip"
    assert slugify("   ") == "clip"
    assert slugify("///") == "clip"


def test_slugify_caps_length() -> None:
    long = "x" * 200
    out = slugify(long, max_length=30)
    assert len(out) <= 30


def test_slugify_squeezes_runs_of_dashes() -> None:
    # Multiple bad chars in a row should not produce "----" in the output.
    assert "--" not in slugify("a !! @@ b")
