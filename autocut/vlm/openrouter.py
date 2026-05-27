"""OpenRouter VLM provider.

OpenRouter (https://openrouter.ai) exposes 200+ LLM/VLM models behind one
OpenAI-compatible HTTP API. We use the official ``openai`` SDK pointed at
``https://openrouter.ai/api/v1`` so:

- a single API key unlocks Claude/GPT/Gemini/Llama/Qwen/Mistral
- pricing and model list are queryable live (see ``discovery.py``)
- failover between providers is built-in (OpenRouter handles it)

This module sends keyframes as base64-encoded JPEGs inside a multimodal
``messages`` payload, parses the JSON response, and validates it against
``ClipPlan``. Network errors and schema violations are wrapped as
``VLMError`` so the caller has a single exception type to catch.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from pathlib import Path
from typing import Any, ClassVar, cast

from openai import APIError, AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import ValidationError

from autocut.models import AnalysisHints, ClipPlan, DetectionResult, Keyframe
from autocut.video.frame_sampler import FrameSpec
from autocut.vlm.base import CostEstimate, VLMError, VLMProvider
from autocut.vlm.prompts import (
    PROMPT_VERSION,
    build_detection_system_prompt,
    build_detection_user_prompt,
    build_system_prompt,
    build_user_prompt,
)

log = logging.getLogger(__name__)

# Default per-image token budget (Anthropic / OpenAI both bill 768px images
# in this range). Used only for the pre-call cost estimate.
_TOKENS_PER_IMAGE: int = 1100
_TOKENS_PER_PROMPT_OVERHEAD: int = 800
_TOKENS_OUTPUT_PER_CLIP: int = 120

# Fallback pricing when discovery did not feed us a live rate. These numbers
# are rough order-of-magnitude; the security guard (``cost_cap_usd``) is the
# real safety, not this estimate.
_FALLBACK_USD_PER_1M_INPUT: float = 3.0
_FALLBACK_USD_PER_1M_OUTPUT: float = 15.0


class OpenRouterProvider(VLMProvider):
    """OpenAI-SDK shim that points at OpenRouter."""

    name: ClassVar[str] = "openrouter"

    BASE_URL: ClassVar[str] = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        attribution_url: str = "https://github.com/ArtiomIV/autocut-skill",
        attribution_title: str = "AutoCut Skill",
        usd_per_1m_input: float | None = None,
        usd_per_1m_output: float | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise VLMError("openrouter requires a non-empty API key")
        self.model = model
        self._usd_per_1m_input = usd_per_1m_input or _FALLBACK_USD_PER_1M_INPUT
        self._usd_per_1m_output = usd_per_1m_output or _FALLBACK_USD_PER_1M_OUTPUT
        self._client = client or AsyncOpenAI(
            base_url=self.BASE_URL,
            api_key=api_key,
            default_headers={
                # OpenRouter uses these headers for attribution analytics.
                "HTTP-Referer": attribution_url,
                "X-Title": attribution_title,
            },
        )

    # ------------------------------------------------------------------
    # VLMProvider interface
    # ------------------------------------------------------------------

    async def analyze(
        self,
        keyframes: list[Keyframe],
        hints: AnalysisHints,
        *,
        video_id: str,
        duration_sec: float,
        timeout_sec: int = 300,
    ) -> ClipPlan:
        if not keyframes:
            raise VLMError("openrouter.analyze called with no keyframes")

        specs = [
            FrameSpec(timestamp=kf.timestamp, scene_index=kf.scene_index or -1) for kf in keyframes
        ]
        system_prompt = build_system_prompt(hints)
        user_prompt = build_user_prompt(
            video_id=video_id,
            duration_sec=duration_sec,
            hints=hints,
            specs=specs,
        )

        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for kf in keyframes:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _encode_image_to_data_url(kf.path)},
                }
            )

        # Local mypy sees openai's strict TypedDict union and rejects the
        # dict-literal; pre-commit mypy has no openai stubs and treats the
        # cast as redundant. Silence both with the combined ignore.
        messages = cast(  # type: ignore[redundant-cast, unused-ignore]
            list[ChatCompletionMessageParam],
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
        )
        start = time.monotonic()
        try:
            completion = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    max_tokens=4096,
                ),
                timeout=timeout_sec,
            )
        except TimeoutError as exc:
            raise VLMError(f"openrouter request timed out after {timeout_sec}s") from exc
        except APIError as exc:
            raise VLMError(f"openrouter API call failed: {exc}") from exc

        elapsed = time.monotonic() - start
        raw = (completion.choices[0].message.content or "").strip()
        if not raw:
            raise VLMError("openrouter returned an empty response body")

        plan = _parse_response(
            raw,
            provider=self.name,
            model=self.model,
            elapsed_sec=elapsed,
        )
        log.info(
            "openrouter analyse complete: model=%s clips=%d elapsed=%.2fs",
            self.model,
            len(plan.clips),
            elapsed,
        )
        return plan

    async def detect_content(
        self,
        keyframes: list[Keyframe],
        audio_description: str,
        *,
        video_id: str,
        duration_sec: float,
        timeout_sec: int = 120,
        transcript_text: str | None = None,
        audio_clip_path: Path | None = None,
        video_clip_paths: list[Path] | None = None,
    ) -> DetectionResult:
        """Classify the video content type via a small OpenRouter call.

        v0.1.0 sends only image content blocks + the textual audio
        description. ``transcript_text`` (Whisper) and ``audio_clip_path``
        (Gemini ``input_audio``) hooks are accepted for forward-compat but
        not yet wired into the payload — they will be in v0.2.0 once the
        capability detection picks Gemini specifically.
        """
        # Silence the forward-compat parameters until v0.2.0 plumbs them in.
        del transcript_text, audio_clip_path, video_clip_paths

        if not keyframes:
            raise VLMError("openrouter.detect_content called with no keyframes")

        system_prompt = build_detection_system_prompt()
        user_prompt = build_detection_user_prompt(
            duration_sec=duration_sec,
            keyframe_timestamps=[kf.timestamp for kf in keyframes],
            audio_description=audio_description,
        )

        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for kf in keyframes:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _encode_image_to_data_url(kf.path)},
                }
            )

        messages = cast(  # type: ignore[redundant-cast, unused-ignore]
            list[ChatCompletionMessageParam],
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
        )
        start = time.monotonic()
        try:
            completion = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    # Detection JSON is tiny; cap output low to limit cost.
                    max_tokens=256,
                ),
                timeout=timeout_sec,
            )
        except TimeoutError as exc:
            raise VLMError(f"openrouter detection timed out after {timeout_sec}s") from exc
        except APIError as exc:
            raise VLMError(f"openrouter detection API call failed: {exc}") from exc

        elapsed = time.monotonic() - start
        # Suppress unused-variable warning when timing is not logged.
        del elapsed
        raw = (completion.choices[0].message.content or "").strip()
        if not raw:
            raise VLMError("openrouter detection returned an empty response body")

        return _parse_detection_response(raw, model=self.model, video_id=video_id)

    def estimate_cost(self, n_keyframes: int) -> CostEstimate:
        input_tokens = _TOKENS_PER_PROMPT_OVERHEAD + n_keyframes * _TOKENS_PER_IMAGE
        # Conservative output guess: assume ~10 clips at 120 tokens each.
        output_tokens = _TOKENS_OUTPUT_PER_CLIP * 10
        total_usd = (
            input_tokens / 1_000_000 * self._usd_per_1m_input
            + output_tokens / 1_000_000 * self._usd_per_1m_output
        )
        return CostEstimate(
            provider=self.name,
            model=self.model,
            n_input_images=n_keyframes,
            estimated_input_tokens=input_tokens,
            estimated_output_tokens=output_tokens,
            estimated_total_usd=round(total_usd, 4),
        )

    def health_check(self) -> bool:
        """Cheapest reachable probe: ask the API for the model list."""
        try:
            asyncio.run(_async_health_probe(self._client))
        except Exception as exc:
            # Any failure here means the provider is not reachable / configured.
            log.debug("openrouter health check failed: %s", exc)
            return False
        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _async_health_probe(client: AsyncOpenAI) -> None:
    await client.models.list()


def _encode_image_to_data_url(path: Path) -> str:
    """Read an image file and return a ``data:image/jpeg;base64,...`` URL."""
    with path.open("rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    suffix = path.suffix.lower().lstrip(".") or "jpeg"
    mime = "jpeg" if suffix == "jpg" else suffix
    return f"data:image/{mime};base64,{encoded}"


def _parse_detection_response(raw: str, *, model: str, video_id: str) -> DetectionResult:
    """Validate the detection JSON. Lenient on field-naming variations.

    Wraps both invalid JSON and pydantic validation failures into ``VLMError``
    with a message that identifies the model — so the user can tell which
    backend misbehaved when one fails.
    """
    # ``video_id`` is currently logged via the surrounding error context;
    # ``del`` silences unused-arg complaints without removing the parameter
    # (we want it in the signature for future structured-logging hooks).
    del video_id
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise VLMError(
            f"openrouter detection ({model}) response was not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise VLMError(f"openrouter detection ({model}) response top-level value is not an object")

    # Some models prefer ``category`` or ``content_type`` — normalise to
    # the schema field ``content_hint`` so DetectionResult validates.
    if "content_hint" not in payload:
        for alias in ("category", "content_type", "type", "label"):
            if alias in payload:
                payload["content_hint"] = payload.pop(alias)
                break
    # Confidence might come as a string ("0.9") or percentage ("90%").
    if isinstance(payload.get("confidence"), str):
        raw_conf = payload["confidence"].strip().rstrip("%")
        try:
            value = float(raw_conf)
        except ValueError as exc:
            raise VLMError(
                f"openrouter detection ({model}) returned non-numeric confidence: {payload['confidence']!r}"
            ) from exc
        # If the model wrote "90", read as percent → 0.9.
        payload["confidence"] = value / 100.0 if value > 1.0 else value

    try:
        # Lowercase the hint so "Boxing" / "BOXING" still match the enum.
        if isinstance(payload.get("content_hint"), str):
            payload["content_hint"] = payload["content_hint"].lower().strip()
        return DetectionResult.model_validate(payload)
    except ValidationError as exc:
        raise VLMError(
            f"openrouter detection ({model}) response failed DetectionResult validation: {exc}"
        ) from exc


def _parse_response(
    raw: str,
    *,
    provider: str,
    model: str,
    elapsed_sec: float,
) -> ClipPlan:
    """Parse a VLM response string and validate it as ``ClipPlan``.

    The provider injects metadata (provider name, model, prompt version,
    timing) if the model forgot, so the downstream pipeline always sees a
    fully-populated record.
    """
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise VLMError(f"openrouter response was not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise VLMError("openrouter response top-level value is not an object")

    metadata = payload.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise VLMError("openrouter response metadata field is not an object")
    metadata.setdefault("vlm_provider", provider)
    metadata.setdefault("vlm_model", model)
    metadata.setdefault("prompt_version", PROMPT_VERSION)
    metadata.setdefault("analysis_time_sec", round(elapsed_sec, 2))

    try:
        return ClipPlan.model_validate(payload)
    except ValidationError as exc:
        raise VLMError(f"openrouter response failed ClipPlan validation: {exc}") from exc
