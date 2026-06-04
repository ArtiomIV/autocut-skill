"""Unit tests for the contact-sheet builder (ffmpeg not invoked)."""

from __future__ import annotations

from pathlib import Path

import pytest

from autocut.video import contact_sheet
from autocut.video.contact_sheet import (
    ContactSheetError,
    _build_filter,
    _escape_fontpath,
    _resolve_font,
    build_contact_sheets,
    build_timestamped_sheets,
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


def test_build_filter_index_offset_shifts_burned_index() -> None:
    # With an offset the cell burns ``n + offset`` (continuous across batches).
    vf = _build_filter(fps=2.0, cell_px=192, cols=6, rows=8, font="/f.ttf", index_offset=48)
    assert r"%{eif\:n+48\:d}" in vf
    assert r"%{n}'" not in vf  # the plain form is replaced when offset is set


def test_build_contact_sheets_rejects_negative_offset(tmp_path: Path) -> None:
    seg = tmp_path / "seg.mp4"
    seg.write_bytes(b"fake")
    with pytest.raises(ContactSheetError, match="index_offset must be >= 0"):
        build_contact_sheets(seg, tmp_path / "out", index_offset=-1)


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


# ---------------------------------------------------------------------------
# build_timestamped_sheets — frame-time math (ffmpeg stubbed out)
# ---------------------------------------------------------------------------


def _stub_build_contact_sheets(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Replace ``build_contact_sheets`` with a no-op recorder; return captured kwargs."""
    captured: dict[str, object] = {}

    def _fake(video: Path, out_dir: Path, **kwargs: object) -> list[Path]:
        captured.update(kwargs)
        return [out_dir / "sheet_001.jpg"]

    monkeypatch.setattr(contact_sheet, "build_contact_sheets", _fake)
    return captured


def test_timestamped_sheets_frame_times_are_index_over_fps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _stub_build_contact_sheets(monkeypatch)
    _, frame_times = build_timestamped_sheets(
        tmp_path / "v.mp4", tmp_path / "out", fps=2.0, start_sec=0.0, end_sec=5.0
    )
    # 5s / 0.5s interval = 10 frames at 0.0, 0.5, ... 4.5.
    assert frame_times == [j * 0.5 for j in range(10)]
    # Window is trimmed to an EXACT multiple of the interval (10 * 0.5 = 5.0).
    assert captured["trim_sec"] == pytest.approx(5.0)
    assert captured["start_sec"] == pytest.approx(0.0)


def test_timestamped_sheets_offsets_times_by_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_build_contact_sheets(monkeypatch)
    _, frame_times = build_timestamped_sheets(
        tmp_path / "v.mp4", tmp_path / "out", fps=1.0, start_sec=10.0, end_sec=13.0
    )
    # Absolute times start at the window start, not at zero.
    assert frame_times == [10.0, 11.0, 12.0]


def test_timestamped_sheets_rejects_inverted_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_build_contact_sheets(monkeypatch)
    with pytest.raises(ContactSheetError, match="end_sec must be greater"):
        build_timestamped_sheets(
            tmp_path / "v.mp4", tmp_path / "out", fps=2.0, start_sec=5.0, end_sec=5.0
        )


def test_timestamped_sheets_rejects_bad_fps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_build_contact_sheets(monkeypatch)
    with pytest.raises(ContactSheetError, match="fps must be > 0"):
        build_timestamped_sheets(
            tmp_path / "v.mp4", tmp_path / "out", fps=0.0, start_sec=0.0, end_sec=5.0
        )
