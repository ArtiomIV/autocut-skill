"""Provider factory — turns a config-shaped request into a ``VLMProvider`` instance.

This is the single dispatch point between the layered config (provider name
+ model + optional API key) and the concrete provider classes. Keeping it
in one place means the rest of the pipeline never imports ``OpenRouterProvider``
directly.

``run`` is cloud-only: the only wired provider is ``openrouter``. The local/host
path is no longer a provider — the orchestrating agent drives the deterministic
``probe``/``sheet``/``cut``/``merge`` subcommands itself (see the ``autocut-run``
skill), so ``--vlm host`` points the user there. Any other provider raises
``UnavailableProviderError`` suggesting the OpenRouter equivalent.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

from autocut.security import KeyringError, SupportedProvider, get_key
from autocut.vlm.base import VLMError, VLMProvider
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

    Only ``openrouter`` is wired: ``run`` is the cloud-only analysis pipeline.
    ``api_key`` is optional — if omitted we read it from the OS keyring. The host
    path is not a provider: ``--vlm host`` raises with a pointer to the agent-driven
    ``sheet``/``cut`` recipe (the ``autocut-run`` skill). ``work_dir`` is accepted
    for signature stability but unused.
    """
    del work_dir  # host pause/resume removed; kept for call-site compatibility
    name = provider_name.strip().lower()

    if name == "host":
        raise UnavailableProviderError(
            "`--vlm host` is no longer a pipeline provider. On a capable agent, run "
            "the local recipe instead: `autocut probe` + `autocut sheet` (read the "
            "grids) + `autocut cut`/`autocut merge` — no API, no cost. For an "
            "autonomous cloud run use `--vlm openrouter`."
        )

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
        f"unknown provider {name!r}. The only pipeline provider is openrouter "
        "(the local/host path runs as an agent recipe, not via --vlm)."
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
