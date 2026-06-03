"""Unit tests for the host-agent provider (pause/resume flow, no network)."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from autocut.models import AnalysisHints, Keyframe
from autocut.vlm import HostAgentPauseRequested, VLMError
from autocut.vlm.host_agent import HostAgentProvider


def _kf(idx: int, secs: float, path: Path) -> Keyframe:
    return Keyframe(scene_index=idx, timestamp=timedelta(seconds=secs), path=path)


@pytest.mark.asyncio
async def test_analyze_writes_request_and_raises_pause(tmp_path: Path) -> None:
    provider = HostAgentProvider(work_dir=tmp_path)
    kfs = [_kf(0, 1.0, tmp_path / "kf_0.jpg"), _kf(1, 5.0, tmp_path / "kf_1.jpg")]
    with pytest.raises(HostAgentPauseRequested) as caught:
        await provider.analyze(
            kfs,
            AnalysisHints(),
            video_id="video_001",
            duration_sec=30.0,
        )
    assert caught.value.request_path == tmp_path / "VLM_REQUEST.md"
    assert caught.value.response_path == tmp_path / "VLM_RESPONSE.json"
    written = caught.value.request_path.read_text(encoding="utf-8")
    assert "kf_0.jpg" in written
    assert "kf_1.jpg" in written
    assert "video_001" in written
    assert "00:00:01.000" in written  # first keyframe timestamp
    assert "00:00:05.000" in written


@pytest.mark.asyncio
async def test_analyze_rejects_empty_keyframes(tmp_path: Path) -> None:
    provider = HostAgentProvider(work_dir=tmp_path)
    with pytest.raises(VLMError, match="no keyframes"):
        await provider.analyze(
            [],
            AnalysisHints(),
            video_id="x",
            duration_sec=1.0,
        )


def test_resume_loads_valid_response(tmp_path: Path) -> None:
    response_path = tmp_path / "VLM_RESPONSE.json"
    response_path.write_text(
        json.dumps(
            {
                "video_id": "video_001",
                "duration_sec": 30.0,
                "clips": [
                    {
                        "id": "c1",
                        "start": "00:00:01",
                        "end": "00:00:05",
                        "category": "highlight",
                        "description": "d",
                        "score": 8,
                        "rationale": "r",
                        "tags": [],
                    }
                ],
                "metadata": {
                    "vlm_provider": "host",
                    "vlm_model": "claude-opus",
                },
            }
        ),
        encoding="utf-8",
    )
    provider = HostAgentProvider(work_dir=tmp_path)
    plan = provider.resume_from_disk()
    assert len(plan.clips) == 1
    assert plan.clips[0].score == 8


def test_resume_errors_when_response_missing(tmp_path: Path) -> None:
    provider = HostAgentProvider(work_dir=tmp_path)
    with pytest.raises(VLMError, match="not found"):
        provider.resume_from_disk()


def test_resume_errors_when_response_is_not_json(tmp_path: Path) -> None:
    (tmp_path / "VLM_RESPONSE.json").write_text("not json", encoding="utf-8")
    provider = HostAgentProvider(work_dir=tmp_path)
    with pytest.raises(VLMError, match="failed to read"):
        provider.resume_from_disk()


def test_resume_errors_on_schema_violation(tmp_path: Path) -> None:
    # ``duration_sec`` must be > 0 — this triggers ClipPlan validation. The
    # provider's ``setdefault`` only fills metadata, it can't rescue a bad
    # required field.
    (tmp_path / "VLM_RESPONSE.json").write_text(
        json.dumps({"video_id": "x", "duration_sec": -1.0, "clips": [], "metadata": {}}),
        encoding="utf-8",
    )
    provider = HostAgentProvider(work_dir=tmp_path)
    with pytest.raises(VLMError, match="failed ClipPlan validation"):
        provider.resume_from_disk()


def test_estimate_cost_is_zero() -> None:
    provider = HostAgentProvider(work_dir=Path("."), agent_hint="claude")
    estimate = provider.estimate_cost(n_keyframes=100)
    assert estimate.is_free
    assert estimate.estimated_total_usd == 0.0


# ---------------------------------------------------------------------------
# Image-only: contact-sheet pause (host two-pass) — no video capability
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_supports_video_always_false(tmp_path: Path) -> None:
    # The host is image-only now; there is no video opt-in.
    assert await HostAgentProvider(work_dir=tmp_path).supports_video() is False


@pytest.mark.asyncio
async def test_analyze_contact_sheets_writes_request_and_pauses(tmp_path: Path) -> None:
    sheets = [tmp_path / "sheet_000.jpg", tmp_path / "sheet_001.jpg"]
    for s in sheets:
        s.write_bytes(b"\xff\xd8\xff")  # presence is enough; bytes are not read
    provider = HostAgentProvider(work_dir=tmp_path, agent_hint="claude")

    with pytest.raises(HostAgentPauseRequested) as caught:
        await provider.analyze_contact_sheets(
            sheets,
            AnalysisHints(),
            video_id="video_001",
            duration_sec=44.0,
            frame_times=[0.0, 0.5, 1.0],
        )

    assert caught.value.request_path == tmp_path / "VLM_REQUEST.md"
    assert caught.value.response_path == tmp_path / "VLM_RESPONSE.json"
    written = caught.value.request_path.read_text(encoding="utf-8")
    assert "sheet_000.jpg" in written  # references each sheet by path
    assert "sheet_001.jpg" in written
    assert "CONTACT SHEETS" in written
    assert "video_001" in written
    # The index -> absolute time sidecar is written and referenced.
    index_path = tmp_path / "VLM_SHEET_INDEX.json"
    assert index_path.is_file()
    assert "VLM_SHEET_INDEX.json" in written
    index_map = json.loads(index_path.read_text(encoding="utf-8"))
    assert index_map["0"] == "00:00:00.000"
    assert index_map["1"] == "00:00:00.500"


@pytest.mark.asyncio
async def test_analyze_contact_sheets_rejects_empty(tmp_path: Path) -> None:
    provider = HostAgentProvider(work_dir=tmp_path)
    with pytest.raises(VLMError, match="no sheets"):
        await provider.analyze_contact_sheets(
            [],
            AnalysisHints(),
            video_id="x",
            duration_sec=10.0,
            frame_times=[],
        )
