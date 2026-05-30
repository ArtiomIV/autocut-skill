"""Unit tests for the OpenRouter discovery client (HTTP mocked with respx)."""

from __future__ import annotations

import httpx
import pytest
import respx

from autocut.vlm import VLMError
from autocut.vlm.discovery import (
    OPENROUTER_MODELS_URL,
    list_openrouter_models,
    validate_openrouter_model,
)


def _models_payload() -> dict[str, object]:
    return {
        "data": [
            {
                "id": "google/gemini-2.5-flash",
                "name": "Gemini 2.5 Flash",
                "context_length": 1_000_000,
                # audio + video -> usable by AutoCut
                "architecture": {"input_modalities": ["text", "image", "audio", "video"]},
                "pricing": {"prompt": "0.000000075", "completion": "0.0000003"},
            },
            {
                "id": "openai/gpt-4o",
                "name": "GPT-4o",
                "context_length": 128000,
                # image only -> filtered out (no audio, no video)
                "architecture": {"input_modalities": ["text", "image"]},
                "pricing": {"prompt": "0.0000025", "completion": "0.00001"},
            },
            {
                "id": "video-only-model",
                "context_length": 64000,
                # video but no audio -> filtered out
                "architecture": {"input_modalities": ["text", "image", "video"]},
                "pricing": {"prompt": "0.000001", "completion": "0.000002"},
            },
            {
                "id": "audio-only-model",
                "context_length": 64000,
                # audio but no video -> filtered out
                "architecture": {"input_modalities": ["text", "audio"]},
                "pricing": {"prompt": "0.000001", "completion": "0.000002"},
            },
            {
                "id": "missing-pricing",
                "context_length": 32000,
                # audio + video, no pricing -> usable, sorts last
                "architecture": {"modality": "text+image+audio+video->text"},
            },
        ]
    }


@pytest.mark.asyncio
async def test_filters_to_audio_and_video_models() -> None:
    with respx.mock() as router:
        router.get(OPENROUTER_MODELS_URL).mock(
            return_value=httpx.Response(200, json=_models_payload())
        )
        models = await list_openrouter_models()
    ids = [m.id for m in models]
    assert "google/gemini-2.5-flash" in ids
    assert "missing-pricing" in ids  # audio+video via legacy modality string
    # Anything missing audio OR video is filtered out.
    assert "openai/gpt-4o" not in ids  # image only
    assert "video-only-model" not in ids  # no audio
    assert "audio-only-model" not in ids  # no video


@pytest.mark.asyncio
async def test_normalises_pricing_to_per_million_tokens() -> None:
    with respx.mock() as router:
        router.get(OPENROUTER_MODELS_URL).mock(
            return_value=httpx.Response(200, json=_models_payload())
        )
        models = await list_openrouter_models()
    by_id = {m.id: m for m in models}
    assert by_id["google/gemini-2.5-flash"].usd_per_1m_input == pytest.approx(0.075)
    assert by_id["google/gemini-2.5-flash"].usd_per_1m_output == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_sorts_cheapest_input_first() -> None:
    with respx.mock() as router:
        router.get(OPENROUTER_MODELS_URL).mock(
            return_value=httpx.Response(200, json=_models_payload())
        )
        models = await list_openrouter_models()
    # gemini-flash is the cheapest with a real price; missing-pricing has None
    # and should land last.
    assert models[0].id == "google/gemini-2.5-flash"
    assert models[-1].id == "missing-pricing"


@pytest.mark.asyncio
async def test_passes_authorization_header_when_key_present() -> None:
    seen_headers: dict[str, str] = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        seen_headers.update(request.headers)
        return httpx.Response(200, json={"data": []})

    with respx.mock() as router:
        router.get(OPENROUTER_MODELS_URL).mock(side_effect=_capture)
        await list_openrouter_models(api_key="sk-or-secret-key-1234567890")
    assert seen_headers.get("authorization", "").startswith("Bearer ")


@pytest.mark.asyncio
async def test_validate_model_passes_for_audio_video_model() -> None:
    with respx.mock() as router:
        router.get(OPENROUTER_MODELS_URL).mock(
            return_value=httpx.Response(200, json=_models_payload())
        )
        # Should not raise: gemini-2.5-flash declares both audio and video.
        await validate_openrouter_model("google/gemini-2.5-flash")


@pytest.mark.asyncio
async def test_validate_model_rejects_video_only_model() -> None:
    with respx.mock() as router:
        router.get(OPENROUTER_MODELS_URL).mock(
            return_value=httpx.Response(200, json=_models_payload())
        )
        with pytest.raises(VLMError, match="does not declare audio"):
            await validate_openrouter_model("video-only-model")


@pytest.mark.asyncio
async def test_validate_model_rejects_unknown_model() -> None:
    with respx.mock() as router:
        router.get(OPENROUTER_MODELS_URL).mock(
            return_value=httpx.Response(200, json=_models_payload())
        )
        with pytest.raises(VLMError, match="not in the OpenRouter catalogue"):
            await validate_openrouter_model("no/such-model")


@pytest.mark.asyncio
async def test_wraps_http_error() -> None:
    with respx.mock() as router:
        router.get(OPENROUTER_MODELS_URL).mock(return_value=httpx.Response(500, text="oops"))
        with pytest.raises(VLMError, match="HTTP 500"):
            await list_openrouter_models()


@pytest.mark.asyncio
async def test_wraps_non_json_response() -> None:
    with respx.mock() as router:
        router.get(OPENROUTER_MODELS_URL).mock(return_value=httpx.Response(200, text="not json"))
        with pytest.raises(VLMError, match="not JSON"):
            await list_openrouter_models()


@pytest.mark.asyncio
async def test_wraps_missing_data_field() -> None:
    with respx.mock() as router:
        router.get(OPENROUTER_MODELS_URL).mock(
            return_value=httpx.Response(200, json={"unexpected": "shape"})
        )
        with pytest.raises(VLMError, match="missing 'data'"):
            await list_openrouter_models()
