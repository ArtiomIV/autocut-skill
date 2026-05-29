"""Live model discovery for OpenRouter.

The ``GET /api/v1/models`` endpoint returns the full catalogue with prices,
context lengths, and modality flags. We filter to vision-capable models and
expose a typed view (``ModelInfo``) the CLI can render as a table.

This module deliberately uses ``httpx`` directly instead of the ``openai``
SDK: the discovery endpoint is simple JSON, and going through ``httpx``
avoids constructing an SDK client just to read a single GET.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from autocut.vlm.base import VLMError

log = logging.getLogger(__name__)

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
_DISCOVERY_TIMEOUT_SEC = 15


class ModelInfo(BaseModel):
    """A single VLM-capable model row, ready for the CLI table."""

    model_config = ConfigDict(extra="ignore")

    id: str
    name: str | None = None
    context_length: int | None = Field(default=None, ge=0)
    usd_per_1m_input: float | None = Field(default=None, ge=0)
    usd_per_1m_output: float | None = Field(default=None, ge=0)


async def list_openrouter_models(
    *,
    api_key: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> list[ModelInfo]:
    """Return every vision-capable model exposed by OpenRouter.

    The endpoint is public; an ``api_key`` is optional and only used to
    surface account-specific availability (e.g. region restrictions).
    """
    raw_models = await _fetch_models_data(api_key=api_key, client=client)

    vision_models: list[ModelInfo] = []
    for raw in raw_models:
        if not _is_vision_capable(raw):
            continue
        info = _to_model_info(raw)
        if info is not None:
            vision_models.append(info)

    # Cheapest input first — most users care about that ranking.
    vision_models.sort(key=lambda m: (m.usd_per_1m_input is None, m.usd_per_1m_input or 0.0, m.id))
    return vision_models


async def model_supports_video(
    model_id: str,
    *,
    api_key: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> bool:
    """Live check: does ``model_id`` declare ``video`` as an input modality?

    Fetches the OpenRouter catalogue and inspects the model's modality flags.
    Raises ``VLMError`` if the catalogue cannot be fetched — callers that want
    a graceful fallback (e.g. the pipeline's capability gate) should catch it
    and treat failure as "no video support".
    """
    models = await _fetch_models_data(api_key=api_key, client=client)
    return model_supports_video_input(model_id, models)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


async def _fetch_models_data(
    *,
    api_key: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> list[dict[str, Any]]:
    """GET ``/models`` and return the raw ``data`` array (dict rows only)."""
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    own_client = client is None
    http = client or httpx.AsyncClient(timeout=_DISCOVERY_TIMEOUT_SEC)
    try:
        try:
            response = await http.get(OPENROUTER_MODELS_URL, headers=headers)
        except httpx.HTTPError as exc:
            raise VLMError(f"failed to fetch OpenRouter model list: {exc}") from exc
        if response.status_code != 200:
            raise VLMError(
                f"OpenRouter /models returned HTTP {response.status_code}: {response.text[:200]}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise VLMError(f"OpenRouter /models response is not JSON: {exc}") from exc
    finally:
        if own_client:
            await http.aclose()

    raw_models = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(raw_models, list):
        raise VLMError("OpenRouter /models payload missing 'data' array")
    return [m for m in raw_models if isinstance(m, dict)]


def _is_vision_capable(raw: dict[str, Any]) -> bool:
    """OpenRouter encodes modality flags under ``architecture.modality``."""
    return _has_modality(raw, "image")


def _has_modality(raw: dict[str, Any], needle: str) -> bool:
    """Return True if the model's input modalities contain ``needle`` (case-insensitive).

    Reads both the legacy ``architecture.modality`` string ("text+image->text"
    style) and the newer ``architecture.input_modalities`` list, so the
    helper works against any OpenRouter payload shape we have seen.
    """
    needle_lc = needle.lower()
    arch = raw.get("architecture") or {}
    if not isinstance(arch, dict):
        return False
    modality = arch.get("modality", "")
    if isinstance(modality, str) and needle_lc in modality.lower():
        return True
    inputs = arch.get("input_modalities")
    return isinstance(inputs, list) and any(
        isinstance(m, str) and needle_lc in m.lower() for m in inputs
    )


def model_supports_audio_input(model_id: str, models: list[dict[str, Any]]) -> bool:
    """Return True if ``model_id`` declares ``audio`` as an input modality.

    Phase E v0.1.0 does not yet ship the audio block path — this helper
    exists so the detector and any future caller can branch on capability
    when we plumb raw audio uploads in v0.2.0 (Whisper-light + Gemini
    ``input_audio``). For now it reads the OpenRouter ``/models`` payload
    directly and ignores any provider-side caching: detection is rare.
    """
    raw = _find_model(model_id, models)
    return raw is not None and _has_modality(raw, "audio")


def model_supports_video_input(model_id: str, models: list[dict[str, Any]]) -> bool:
    """Return True if ``model_id`` declares ``video`` as an input modality.

    See ``model_supports_audio_input`` for the rationale; video upload is
    Gemini-family only today (Sonnet/GPT-4o do not yet expose it through
    OpenRouter's OpenAI-compatible schema).
    """
    raw = _find_model(model_id, models)
    return raw is not None and _has_modality(raw, "video")


def _find_model(model_id: str, models: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Linear scan; the OpenRouter catalogue is a few hundred rows so this is fine."""
    for entry in models:
        if isinstance(entry, dict) and entry.get("id") == model_id:
            return entry
    return None


def _to_model_info(raw: dict[str, Any]) -> ModelInfo | None:
    """Extract the fields we care about. Returns ``None`` if ``id`` is missing."""
    model_id = raw.get("id")
    if not isinstance(model_id, str) or not model_id:
        return None
    pricing = raw.get("pricing") or {}
    return ModelInfo(
        id=model_id,
        name=raw.get("name") if isinstance(raw.get("name"), str) else None,
        context_length=_safe_int(raw.get("context_length")),
        usd_per_1m_input=_price_per_million(pricing.get("prompt")),
        usd_per_1m_output=_price_per_million(pricing.get("completion")),
    )


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _price_per_million(value: Any) -> float | None:
    """OpenRouter quotes prices as USD per token; we normalise to USD per 1M tokens."""
    if value is None:
        return None
    try:
        per_token = float(value)
    except (TypeError, ValueError):
        return None
    if per_token < 0:
        return None
    return round(per_token * 1_000_000, 4)
