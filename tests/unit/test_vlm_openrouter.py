"""Unit tests for the OpenRouter provider (HTTP mocked with respx)."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import httpx
import pytest
import respx

from autocut.models import AnalysisHints, ContentHint, Keyframe
from autocut.vlm import VLMError
from autocut.vlm.openrouter import OpenRouterProvider, _extract_json, _parse_response


def _clip_json(cid: str, start: str, end: str, *, tags: object = None) -> dict[str, object]:
    clip: dict[str, object] = {
        "id": cid,
        "start": start,
        "end": end,
        "category": "highlight",
        "description": "d",
        "score": 8,
        "rationale": "r",
    }
    if tags is not None:
        clip["tags"] = tags
    return clip


def test_parse_response_drops_malformed_clip_keeps_valid() -> None:
    # One good clip + one with end <= start (invalid). The bad clip is dropped,
    # the run survives with the good one — a single malformed clip must not sink
    # the whole (paid) batch.
    payload = {
        "video_id": "v",
        "duration_sec": 60.0,
        "clips": [
            _clip_json("good", "00:00:05.000", "00:00:10.000"),
            _clip_json("bad", "00:00:20.000", "00:00:18.000"),  # end < start
        ],
        "metadata": {},
    }
    plan = _parse_response(json.dumps(payload), provider="openrouter", model="m", elapsed_sec=1.0)
    assert [c.id for c in plan.clips] == ["good"]


def test_parse_response_coerces_null_tags() -> None:
    # A clip with ``tags: null`` must not fail validation (the model emits this).
    payload = {
        "video_id": "v",
        "duration_sec": 60.0,
        "clips": [_clip_json("c", "00:00:05.000", "00:00:10.000", tags=None)],
        "metadata": {},
    }
    # Force the null through (the helper drops None, so set it explicitly).
    payload["clips"][0]["tags"] = None  # type: ignore[index]
    plan = _parse_response(json.dumps(payload), provider="openrouter", model="m", elapsed_sec=1.0)
    assert plan.clips[0].tags == []


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
    assert plan.metadata.prompt_version == "v12"


@pytest.mark.asyncio
async def test_analyze_video_clip_happy_path(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42fake-mp4-bytes")
    with respx.mock(base_url="https://openrouter.ai/api/v1", assert_all_called=True) as router:
        router.post("/chat/completions").mock(
            return_value=httpx.Response(200, json=_chat_completion_payload(_valid_clipplan_json()))
        )
        provider = OpenRouterProvider(api_key="sk-or-test", model="google/gemini-3.5-flash")
        plan = await provider.analyze_video_clip(
            clip,
            AnalysisHints(),
            video_id="video_001",
            clip_duration_sec=44.0,
        )
    assert len(plan.clips) == 1
    # Provenance is overwritten authoritatively with our configured model.
    assert plan.metadata.vlm_model == "google/gemini-3.5-flash"
    assert plan.metadata.vlm_provider == "openrouter"


@pytest.mark.asyncio
async def test_analyze_video_clip_missing_file_raises(tmp_path: Path) -> None:
    provider = OpenRouterProvider(api_key="sk-or-test", model="x")
    with pytest.raises(VLMError, match="video clip not found"):
        await provider.analyze_video_clip(
            tmp_path / "nope.mp4",
            AnalysisHints(),
            video_id="v",
            clip_duration_sec=10.0,
        )


@pytest.mark.asyncio
async def test_analyze_audio_clip_happy_path(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp3"
    clip.write_bytes(b"ID3\x04\x00fake-mp3-bytes")
    captured: dict[str, object] = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content.decode("utf-8", "replace")
        return httpx.Response(200, json=_chat_completion_payload(_valid_clipplan_json()))

    with respx.mock(base_url="https://openrouter.ai/api/v1", assert_all_called=True) as router:
        router.post("/chat/completions").mock(side_effect=_capture)
        provider = OpenRouterProvider(api_key="sk-or-test", model="google/gemini-2.5-flash")
        plan = await provider.analyze_audio_clip(
            clip,
            AnalysisHints(),
            video_id="video_001",
            clip_duration_sec=120.0,
        )
    assert len(plan.clips) == 1
    assert plan.metadata.vlm_model == "google/gemini-2.5-flash"
    assert plan.metadata.vlm_provider == "openrouter"
    # The request must carry an input_audio block with format mp3 (the verified
    # OpenRouter->Gemini shape), NOT an image/video block.
    body = str(captured["body"])
    assert "input_audio" in body
    assert '"format"' in body and "mp3" in body


@pytest.mark.asyncio
async def test_analyze_audio_clip_missing_file_raises(tmp_path: Path) -> None:
    provider = OpenRouterProvider(api_key="sk-or-test", model="x")
    with pytest.raises(VLMError, match="audio clip not found"):
        await provider.analyze_audio_clip(
            tmp_path / "nope.mp3",
            AnalysisHints(),
            video_id="v",
            clip_duration_sec=10.0,
        )


@pytest.mark.asyncio
async def test_supports_audio_true_when_model_declares_it() -> None:
    payload = {
        "data": [
            {
                "id": "google/gemini-2.5-flash",
                "architecture": {"input_modalities": ["text", "image", "audio", "video"]},
            }
        ]
    }
    with respx.mock(base_url="https://openrouter.ai/api/v1", assert_all_called=True) as router:
        router.get("/models").mock(return_value=httpx.Response(200, json=payload))
        provider = OpenRouterProvider(api_key="sk-or-test", model="google/gemini-2.5-flash")
        assert await provider.supports_audio() is True


@pytest.mark.asyncio
async def test_supports_video_true_when_model_declares_it() -> None:
    payload = {
        "data": [
            {
                "id": "google/gemini-3.5-flash",
                "architecture": {"input_modalities": ["text", "image", "video"]},
            }
        ]
    }
    with respx.mock(base_url="https://openrouter.ai/api/v1", assert_all_called=True) as router:
        router.get("/models").mock(return_value=httpx.Response(200, json=payload))
        provider = OpenRouterProvider(api_key="sk-or-test", model="google/gemini-3.5-flash")
        assert await provider.supports_video() is True


@pytest.mark.asyncio
async def test_supports_video_false_on_fetch_error() -> None:
    with respx.mock(base_url="https://openrouter.ai/api/v1") as router:
        router.get("/models").mock(return_value=httpx.Response(500, text="boom"))
        provider = OpenRouterProvider(api_key="sk-or-test", model="google/gemini-3.5-flash")
        assert await provider.supports_video() is False


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
async def test_analyze_drops_malformed_clip(fake_keyframes: list[Keyframe]) -> None:
    # A clip missing a required field is DROPPED (not fatal): the run survives
    # with the well-formed clips. One good clip + one missing ``rationale``.
    json_body = json.dumps(
        {
            "video_id": "v",
            "duration_sec": 30.0,
            "clips": [
                {
                    "id": "good",
                    "start": "00:00:01",
                    "end": "00:00:06",
                    "category": "highlight",
                    "description": "d",
                    "score": 8,
                    "rationale": "r",
                },
                {
                    "id": "bad",
                    "start": "00:00:10",
                    "end": "00:00:15",
                    "category": "highlight",
                    "description": "d",
                    "score": 5,
                    # missing ``rationale`` -> this clip is dropped
                },
            ],
            "metadata": {"vlm_provider": "openrouter", "vlm_model": "x"},
        }
    )
    with respx.mock(base_url="https://openrouter.ai/api/v1") as router:
        router.post("/chat/completions").mock(
            return_value=httpx.Response(200, json=_chat_completion_payload(json_body))
        )
        provider = OpenRouterProvider(api_key="sk-or-test", model="x")
        plan = await provider.analyze(
            fake_keyframes, AnalysisHints(), video_id="v", duration_sec=30.0
        )
    assert [c.id for c in plan.clips] == ["good"]


@pytest.mark.asyncio
async def test_analyze_wraps_non_object_response(fake_keyframes: list[Keyframe]) -> None:
    # A genuinely broken response (top-level not an object) still raises VLMError.
    with respx.mock(base_url="https://openrouter.ai/api/v1") as router:
        router.post("/chat/completions").mock(
            return_value=httpx.Response(200, json=_chat_completion_payload("[1, 2, 3]"))
        )
        provider = OpenRouterProvider(api_key="sk-or-test", model="x")
        with pytest.raises(VLMError):
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


@pytest.mark.asyncio
async def test_analyze_retries_then_succeeds_on_bad_json(fake_keyframes: list[Keyframe]) -> None:
    # First reply is unparseable, second is valid: the request is RE-ISSUED and
    # the run succeeds — a malformed JSON must never crash the pipeline.
    responses = [
        httpx.Response(200, json=_chat_completion_payload("not json at all")),
        httpx.Response(200, json=_chat_completion_payload(_valid_clipplan_json())),
    ]
    with respx.mock(base_url="https://openrouter.ai/api/v1") as router:
        route = router.post("/chat/completions").mock(side_effect=responses)
        provider = OpenRouterProvider(api_key="sk-or-test", model="x")
        plan = await provider.analyze(
            fake_keyframes, AnalysisHints(), video_id="v", duration_sec=10.0
        )
    assert len(plan.clips) == 1
    assert route.call_count == 2  # one bad attempt, one good retry


@pytest.mark.asyncio
async def test_analyze_retries_on_transient_5xx(fake_keyframes: list[Keyframe]) -> None:
    # A transient 503 is retried (with backoff); the second attempt succeeds.
    responses = [
        httpx.Response(503, text="upstream temporarily unavailable"),
        httpx.Response(200, json=_chat_completion_payload(_valid_clipplan_json())),
    ]
    with respx.mock(base_url="https://openrouter.ai/api/v1") as router:
        route = router.post("/chat/completions").mock(side_effect=responses)
        provider = OpenRouterProvider(api_key="sk-or-test", model="x")
        plan = await provider.analyze(
            fake_keyframes, AnalysisHints(), video_id="v", duration_sec=10.0
        )
    assert len(plan.clips) == 1
    assert route.call_count == 2


@pytest.mark.asyncio
async def test_analyze_does_not_retry_client_error(fake_keyframes: list[Keyframe]) -> None:
    # A permanent 400 (bad request) is NOT retried — re-issuing the same request
    # cannot help, so it raises at once after a single call.
    with respx.mock(base_url="https://openrouter.ai/api/v1") as router:
        route = router.post("/chat/completions").mock(
            return_value=httpx.Response(400, text="bad request")
        )
        provider = OpenRouterProvider(api_key="sk-or-test", model="x")
        with pytest.raises(VLMError, match="API call failed"):
            await provider.analyze(fake_keyframes, AnalysisHints(), video_id="v", duration_sec=10.0)
        assert route.call_count == 1


@pytest.mark.asyncio
async def test_analyze_raises_after_exhausting_retries(fake_keyframes: list[Keyframe]) -> None:
    # Every attempt returns junk: after _MAX_REQUEST_ATTEMPTS the last error is
    # surfaced (it carries the diagnostic), and the call count is bounded.
    from autocut.vlm.openrouter import _MAX_REQUEST_ATTEMPTS

    with respx.mock(base_url="https://openrouter.ai/api/v1") as router:
        route = router.post("/chat/completions").mock(
            return_value=httpx.Response(200, json=_chat_completion_payload("still not json"))
        )
        provider = OpenRouterProvider(api_key="sk-or-test", model="x")
        with pytest.raises(VLMError, match="not valid JSON"):
            await provider.analyze(fake_keyframes, AnalysisHints(), video_id="v", duration_sec=10.0)
        assert route.call_count == _MAX_REQUEST_ATTEMPTS


def test_extract_json_passes_clean_object_through() -> None:
    clean = '{"content_hint": "highlights", "confidence": 1.0}'
    assert _extract_json(clean) == clean


def test_extract_json_strips_markdown_fence() -> None:
    fenced = '```json\n{"content_hint": "highlights", "confidence": 1.0}\n```'
    assert json.loads(_extract_json(fenced)) == {"content_hint": "highlights", "confidence": 1.0}


def test_extract_json_isolates_object_from_prose() -> None:
    prose = 'Here is the analysis: {"content_hint": "talk", "confidence": 0.7}. Done.'
    assert json.loads(_extract_json(prose)) == {"content_hint": "talk", "confidence": 0.7}


@pytest.mark.asyncio
async def test_detect_content_parses_fenced_response(fake_keyframes: list[Keyframe]) -> None:
    # A model that wraps its JSON in a markdown fence must still be parsed.
    fenced = (
        '```json\n{"content_hint": "highlights", "confidence": 0.95,'
        ' "reasoning": "ring + gloves"}\n```'
    )
    with respx.mock(base_url="https://openrouter.ai/api/v1", assert_all_called=True) as router:
        router.post("/chat/completions").mock(
            return_value=httpx.Response(200, json=_chat_completion_payload(fenced))
        )
        provider = OpenRouterProvider(api_key="sk-or-test", model="google/gemini-3.5-flash")
        result = await provider.detect_content(
            fake_keyframes,
            "onset rate high; voice activity sparse",
            video_id="v",
            duration_sec=24.0,
        )
    assert result.content_hint == ContentHint.highlights
    assert result.confidence == 0.95


@pytest.mark.asyncio
async def test_detect_content_error_includes_raw_snippet(fake_keyframes: list[Keyframe]) -> None:
    # A truncated response (the real Gemini 3.x failure mode) cannot be
    # recovered, but the error must surface the raw text for debugging.
    truncated = '{"content_hint": "boxing", "confidence": 0.98, "'
    with respx.mock(base_url="https://openrouter.ai/api/v1") as router:
        router.post("/chat/completions").mock(
            return_value=httpx.Response(200, json=_chat_completion_payload(truncated))
        )
        provider = OpenRouterProvider(api_key="sk-or-test", model="x")
        with pytest.raises(VLMError, match="raw response began with"):
            await provider.detect_content(
                fake_keyframes,
                "audio",
                video_id="v",
                duration_sec=10.0,
            )
