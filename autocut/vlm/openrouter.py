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

from autocut.models import AnalysisHints, ClipPlan, Keyframe
from autocut.video.frame_sampler import FrameSpec
from autocut.vlm.base import CostEstimate, VLMError, VLMProvider
from autocut.vlm.prompts import (
    PROMPT_VERSION,
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
