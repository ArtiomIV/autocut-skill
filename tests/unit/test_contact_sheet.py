"""Unit tests for the contact-sheet builder (ffmpeg not invoked)."""

from __future__ import annotations

from pathlib import Path

import pytest

from autocut.video.contact_sheet import (
    ContactSheetError,
    _build_filter,
    _escape_fontpath,
    _resolve_font,
    build_contact_sheets,
)


def test_escape_fontpath_windows() -> None:
    # Backslashes -> forward slashes, drive-letter colon escaped for filtergraph.
    assert _escape_fontpath(r"C:\Windows\Fonts\arial.ttf") == r"C\:/Windows/Fonts/arial.ttf"


def test_escape_fontpath_posix() -> None:
    # A POSIX path has no colon/backslash, so it is unchanged.
    p = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    assert _escape_fontpath(p) == p


def test_build_filter_has_all_stages() -> None:
    vf = _build_filter(fps=2.0, cell_px=192, cols=6, rows=8, font="/f.ttf")
    assert "fps=2" in vf
    assert "scale=" in vf and "192" in vf
    # Each cell is labelled with its 0-based frame index (looked up in the map).
    assert r"drawtext=" in vf
    assert r"%{n}" in vf
    assert "tile=6x8" in vf


def test_resolve_font_explicit_ok(tmp_path: Path) -> None:
    font = tmp_path / "f.ttf"
    font.write_bytes(b"not a real font")
    assert _resolve_font(str(font)) == str(font)


def test_resolve_font_explicit_missing(tmp_path: Path) -> None:
    with pytest.raises(ContactSheetError, match="font file not found"):
        _resolve_font(str(tmp_path / "missing.ttf"))


def test_build_contact_sheets_missing_segment(tmp_path: Path) -> None:
    with pytest.raises(ContactSheetError, match="segment not found"):
        build_contact_sheets(tmp_path / "nope.mp4", tmp_path / "out")


def test_build_contact_sheets_rejects_bad_grid(tmp_path: Path) -> None:
    seg = tmp_path / "seg.mp4"
    seg.write_bytes(b"fake")
    with pytest.raises(ContactSheetError, match="must all be > 0"):
        build_contact_sheets(seg, tmp_path / "out", cols=0)
