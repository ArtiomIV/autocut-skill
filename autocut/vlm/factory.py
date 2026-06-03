"""Provider factory — turns a config-shaped request into a ``VLMProvider`` instance.

This is the single dispatch point between the layered config (provider name
+ model + optional API key) and the concrete provider classes. Keeping it
in one place means the rest of the pipeline never imports ``OpenRouterProvider``
or ``HostAgentProvider`` directly.

In v0.1.0 only ``host`` and ``openrouter`` are wired up. Asking for any
other provider raises ``UnavailableProviderError`` with a friendly message
that points the user to the OpenRouter alternative (e.g.
``anthropic/claude-opus-4-6``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

from autocut.security import KeyringError, SupportedProvider, get_key
from autocut.vlm.base import VLMError, VLMProvider
from autocut.vlm.host_agent import HostAgentProvider
from autocut.vlm.openrouter import OpenRouterProvider

log = logging.getLogger(__name__)


class UnavailableProviderError(VLMError):
    """Raised when the caller asks for a provider not implemented in this version."""


# v0.2.0+ providers — listed here so the friendly error can suggest the
# OpenRouter route for the most common destinations.
_DEFERRED_TO_OPENROUTER: Final[dict[str, str]] = {
    "anthropic": "anthropic/claude-opus-4-6",
    "openai": "openai/gpt-4o",
    "gemini": "google/gemini-2.5-flash",
}
_DEFERRED_LOCAL: Final[tuple[str, ...]] = ("ollama", "lmstudio")


def make_provider(
    provider_name: str,
    *,
    model: str,
    work_dir: Path | None = None,
    api_key: str | None = None,
) -> VLMProvider:
    """Return a ready-to-use ``VLMProvider`` for ``provider_name``.

    ``model`` is required even for ``host`` (it becomes the agent hint).
    ``work_dir`` is required for ``host`` — it is where ``VLM_REQUEST.md``
    and ``VLM_RESPONSE.json`` live during the pause/resume handshake.
    ``api_key`` is optional: if omitted we read it from the OS keyring. The host
    is image-only (the contact-sheet two-pass), so there is no video-capability
    flag; non-host providers discover video support from their model catalogue.
    """
    name = provider_name.strip().lower()

    if name == "host":
        if work_dir is None:
            raise VLMError("host provider requires work_dir for the request/response files")
        return HostAgentProvider(work_dir=work_dir, agent_hint=model)

    if name == "openrouter":
        key = api_key or _resolve_keyring("openrouter")
        if not key:
            raise VLMError(
                "openrouter provider requires an API key. "
                "Run `autocut keys set openrouter` to store one."
            )
        return OpenRouterProvider(api_key=key, model=model)

    if name in _DEFERRED_TO_OPENROUTER:
        suggested = _DEFERRED_TO_OPENROUTER[name]
        raise UnavailableProviderError(
            f"provider {name!r} ships in v0.2.0. "
            f"For now use OpenRouter with the equivalent model: "
            f"--vlm openrouter --vlm-model {suggested}"
        )

    if name in _DEFERRED_LOCAL:
        raise UnavailableProviderError(
            f"provider {name!r} (local) ships in v0.3.0. "
            f"Use --vlm openrouter for cloud inference until then."
        )

    raise UnavailableProviderError(
        f"unknown provider {name!r}. Available in v0.1.0: host, openrouter."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_keyring(provider: SupportedProvider) -> str | None:
    """Look up an API key in the OS keyring; swallow backend errors as None."""
    try:
        return get_key(provider)
    except KeyringError as exc:
        log.warning("keyring lookup for %s failed: %s", provider, exc)
        return None
