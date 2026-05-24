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
