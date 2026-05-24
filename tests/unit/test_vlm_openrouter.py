"""Unit tests for the OpenRouter provider (HTTP mocked with respx)."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import httpx
import pytest
import respx

from autocut.models import AnalysisHints, Keyframe
from autocut.vlm import VLMError
from autocut.vlm.openrouter import OpenRouterProvider


def _kf(idx: int, secs: float, path: Path) -> Keyframe:
    return Keyframe(scene_index=idx, timestamp=timedelta(seconds=secs), path=path)


@pytest.fixture
def fake_keyframes(tmp_path: Path) -> list[Keyframe]:
    paths: list[Path] = []
    for i in range(2):
        p = tmp_path / f"kf_{i}.jpg"
        # Minimal valid JPEG header is fine — we only base64-encode the bytes.
        p.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg-bytes")
        paths.append(p)
    return [
        _kf(0, 1.0, paths[0]),
        _kf(1, 4.5, paths[1]),
    ]


def _chat_completion_payload(content: str) -> dict[str, object]:
    return {
        "id": "cmpl_test",
        "object": "chat.completion",
        "created": 0,
        "model": "google/gemini-2.5-flash",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


def _valid_clipplan_json() -> str:
    return json.dumps(
        {
            "video_id": "video_001",
            "duration_sec": 60.0,
            "clips": [
                {
                    "id": "c1",
                    "start": "00:00:01",
                    "end": "00:00:05",
                    "category": "highlight",
                    "description": "punch lands",
                    "score": 9,
                    "rationale": "decisive moment",
                    "tags": ["combo"],
                }
            ],
            "metadata": {
                "vlm_provider": "openrouter",
                "vlm_model": "google/gemini-2.5-flash",
            },
        }
    )


def test_constructor_rejects_empty_key() -> None:
    with pytest.raises(VLMError, match="non-empty API key"):
        OpenRouterProvider(api_key="", model="x")


def test_constructor_rejects_whitespace_key() -> None:
    with pytest.raises(VLMError, match="non-empty API key"):
        OpenRouterProvider(api_key="   ", model="x")


def test_estimate_cost_scales_with_keyframes() -> None:
    provider = OpenRouterProvider(api_key="sk-or-test", model="x")
    a = provider.estimate_cost(n_keyframes=10)
    b = provider.estimate_cost(n_keyframes=100)
    assert b.estimated_input_tokens > a.estimated_input_tokens
    assert b.estimated_total_usd > a.estimated_total_usd
    assert not a.is_free


@pytest.mark.asyncio
async def test_analyze_happy_path(fake_keyframes: list[Keyframe]) -> None:
    with respx.mock(base_url="https://openrouter.ai/api/v1", assert_all_called=True) as router:
        router.post("/chat/completions").mock(
            return_value=httpx.Response(200, json=_chat_completion_payload(_valid_clipplan_json()))
        )
        provider = OpenRouterProvider(api_key="sk-or-test", model="google/gemini-2.5-flash")
        plan = await provider.analyze(
            fake_keyframes,
            AnalysisHints(),
            video_id="video_001",
            duration_sec=60.0,
        )
    assert len(plan.clips) == 1
    assert plan.clips[0].score == 9
    # Provider injects metadata even if the model omitted it.
    assert plan.metadata.prompt_version == "v1"


@pytest.mark.asyncio
async def test_analyze_rejects_empty_keyframes() -> None:
    provider = OpenRouterProvider(api_key="sk-or-test", model="x")
    with pytest.raises(VLMError, match="no keyframes"):
        await provider.analyze(
            [],
            AnalysisHints(),
            video_id="v",
            duration_sec=1.0,
        )


@pytest.mark.asyncio
async def test_analyze_wraps_invalid_json(fake_keyframes: list[Keyframe]) -> None:
    with respx.mock(base_url="https://openrouter.ai/api/v1") as router:
        router.post("/chat/completions").mock(
            return_value=httpx.Response(200, json=_chat_completion_payload("not json at all"))
        )
        provider = OpenRouterProvider(api_key="sk-or-test", model="x")
        with pytest.raises(VLMError, match="not valid JSON"):
            await provider.analyze(
                fake_keyframes,
                AnalysisHints(),
                video_id="v",
                duration_sec=10.0,
            )


@pytest.mark.asyncio
async def test_analyze_wraps_schema_violation(fake_keyframes: list[Keyframe]) -> None:
    bad_json = json.dumps(
        {
            "video_id": "v",
            "duration_sec": 10.0,
            # Missing required ``rationale`` field on the clip -> schema violation.
            "clips": [
                {
                    "id": "c1",
                    "start": "00:00:00",
                    "end": "00:00:05",
                    "category": "highlight",
                    "description": "d",
                    "score": 5,
                }
            ],
            "metadata": {"vlm_provider": "openrouter", "vlm_model": "x"},
        }
    )
    with respx.mock(base_url="https://openrouter.ai/api/v1") as router:
        router.post("/chat/completions").mock(
            return_value=httpx.Response(200, json=_chat_completion_payload(bad_json))
        )
        provider = OpenRouterProvider(api_key="sk-or-test", model="x")
        with pytest.raises(VLMError, match="ClipPlan validation"):
            await provider.analyze(
                fake_keyframes,
                AnalysisHints(),
                video_id="v",
                duration_sec=10.0,
            )


@pytest.mark.asyncio
async def test_analyze_wraps_empty_response(fake_keyframes: list[Keyframe]) -> None:
    with respx.mock(base_url="https://openrouter.ai/api/v1") as router:
        router.post("/chat/completions").mock(
            return_value=httpx.Response(200, json=_chat_completion_payload(""))
        )
        provider = OpenRouterProvider(api_key="sk-or-test", model="x")
        with pytest.raises(VLMError, match="empty response"):
            await provider.analyze(
                fake_keyframes,
                AnalysisHints(),
                video_id="v",
                duration_sec=10.0,
            )
