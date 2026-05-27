"""Unit tests for the ``autocut merge --from-manifest`` selection helper.

The CLI subcommand delegates path resolution to ``_select_from_manifest``,
a pure function over a JSON-shaped dict. We exercise filtering, ordering
and the defensive error paths here so a broken manifest fails loud at the
right boundary instead of producing a silent miscut.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cli.__main__ import ManifestSelectionError, _select_from_manifest


def _write_manifest(tmp_path: Path, clips: list[dict[str, object]], separate: list[str]) -> Path:
    """Build a minimal manifest.json fixture covering only the fields used."""
    payload = {
        "video": {"path": str(tmp_path / "src.mp4"), "duration_sec": 60.0},
        "clips": clips,
        "outputs": {"separate": separate},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_selects_only_clips_above_min_score(tmp_path: Path) -> None:
    clips: list[dict[str, object]] = [
        {"rank": 1, "final_score": 9, "start": "00:00:05.000"},
        {"rank": 2, "final_score": 6, "start": "00:00:30.000"},
        {"rank": 3, "final_score": 8, "start": "00:00:15.000"},
    ]
    separate = [
        str(tmp_path / "clip_001.mp4"),
        str(tmp_path / "clip_002.mp4"),
        str(tmp_path / "clip_003.mp4"),
    ]
    manifest = _write_manifest(tmp_path, clips, separate)

    selected = _select_from_manifest(manifest, min_score=7, order="manifest")
    # min_score=7 keeps clip 1 (score 9) and clip 3 (score 8) only.
    assert [p.name for p in selected] == ["clip_001.mp4", "clip_003.mp4"]


def test_chronological_order_sorts_by_clip_start(tmp_path: Path) -> None:
    clips: list[dict[str, object]] = [
        {"rank": 1, "final_score": 9, "start": "00:00:50.000"},
        {"rank": 2, "final_score": 8, "start": "00:00:10.000"},
        {"rank": 3, "final_score": 7, "start": "00:00:30.000"},
    ]
    separate = [
        str(tmp_path / "late.mp4"),
        str(tmp_path / "early.mp4"),
        str(tmp_path / "mid.mp4"),
    ]
    manifest = _write_manifest(tmp_path, clips, separate)

    selected = _select_from_manifest(manifest, min_score=0, order="chronological")
    assert [p.name for p in selected] == ["early.mp4", "mid.mp4", "late.mp4"]


def test_score_desc_order_sorts_high_to_low(tmp_path: Path) -> None:
    clips: list[dict[str, object]] = [
        {"rank": 1, "final_score": 6, "start": "00:00:05.000"},
        {"rank": 2, "final_score": 9, "start": "00:00:30.000"},
        {"rank": 3, "final_score": 8, "start": "00:00:15.000"},
    ]
    separate = [
        str(tmp_path / "low.mp4"),
        str(tmp_path / "high.mp4"),
        str(tmp_path / "mid.mp4"),
    ]
    manifest = _write_manifest(tmp_path, clips, separate)

    selected = _select_from_manifest(manifest, min_score=0, order="score-desc")
    assert [p.name for p in selected] == ["high.mp4", "mid.mp4", "low.mp4"]


def test_score_path_correspondence_preserved_across_sorts(tmp_path: Path) -> None:
    # Regression guard: the sort must keep score↔path bound together, not
    # reorder paths independently of their clips.
    clips: list[dict[str, object]] = [
        {"rank": 1, "final_score": 5, "start": "00:00:00.000"},
        {"rank": 2, "final_score": 10, "start": "00:00:30.000"},
    ]
    separate = [
        str(tmp_path / "score5.mp4"),
        str(tmp_path / "score10.mp4"),
    ]
    manifest = _write_manifest(tmp_path, clips, separate)

    selected = _select_from_manifest(manifest, min_score=7, order="score-desc")
    # min_score=7 drops the score-5 clip; the kept path must be score10.mp4.
    assert [p.name for p in selected] == ["score10.mp4"]


def test_raises_on_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(ManifestSelectionError, match="manifest not found"):
        _select_from_manifest(tmp_path / "nope.json", min_score=0, order="manifest")


def test_raises_on_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("not json at all", encoding="utf-8")
    with pytest.raises(ManifestSelectionError, match="failed to read manifest"):
        _select_from_manifest(path, min_score=0, order="manifest")


def test_raises_when_separate_outputs_missing(tmp_path: Path) -> None:
    # Manifest exists but the user ran ``--output merged`` only → no
    # ``outputs.separate`` array. We need it to look up per-clip paths.
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "video": {"path": "x.mp4", "duration_sec": 60.0},
                "clips": [{"rank": 1, "final_score": 9, "start": "00:00:05.000"}],
                "outputs": {"merged": ["highlights.mp4"]},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ManifestSelectionError, match=r"no 'outputs\.separate' paths"):
        _select_from_manifest(path, min_score=0, order="manifest")


def test_raises_on_length_mismatch(tmp_path: Path) -> None:
    # Defensive: dispatcher writes index-aligned today; flag drift loudly.
    clips: list[dict[str, object]] = [
        {"rank": 1, "final_score": 9, "start": "00:00:05.000"},
        {"rank": 2, "final_score": 8, "start": "00:00:30.000"},
    ]
    separate = [str(tmp_path / "only_one.mp4")]
    manifest = _write_manifest(tmp_path, clips, separate)
    with pytest.raises(ManifestSelectionError, match="inconsistent"):
        _select_from_manifest(manifest, min_score=0, order="manifest")


def test_raises_on_unknown_order(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, [], [])
    with pytest.raises(ManifestSelectionError, match="unknown --order"):
        _select_from_manifest(manifest, min_score=0, order="random")
