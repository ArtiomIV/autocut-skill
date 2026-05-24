"""Tests for ``autocut.security.paths`` — safe path resolution and parent guard."""

from __future__ import annotations

from pathlib import Path

import pytest

from autocut.security.paths import PathValidationError, ensure_inside, safe_resolve


def test_safe_resolve_expands_user(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows
    resolved = safe_resolve("~/foo.txt")
    assert str(resolved).startswith(str(tmp_path))


def test_safe_resolve_returns_absolute_even_for_missing_paths(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist_yet.mp4"
    resolved = safe_resolve(missing)
    assert resolved.is_absolute()


def test_ensure_inside_accepts_valid_child(tmp_path: Path) -> None:
    child = tmp_path / "sub" / "clip.mp4"
    out = ensure_inside(child, tmp_path)
    assert out == child.resolve()


def test_ensure_inside_rejects_parent_escape(tmp_path: Path) -> None:
    with pytest.raises(PathValidationError):
        ensure_inside(tmp_path / ".." / ".." / "escape.txt", tmp_path)


def test_ensure_inside_rejects_absolute_path_outside_parent(tmp_path: Path) -> None:
    other = tmp_path.parent / "sibling.txt"
    with pytest.raises(PathValidationError):
        ensure_inside(other, tmp_path)


def test_ensure_inside_treats_parent_itself_as_valid(tmp_path: Path) -> None:
    # Resolving ``parent`` against itself should not raise (it's "inside" itself).
    out = ensure_inside(tmp_path, tmp_path)
    assert out == tmp_path.resolve()
