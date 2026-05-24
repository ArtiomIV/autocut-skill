"""OS keyring wrapper for VLM provider API keys.

Backends used automatically by the ``keyring`` library:
- Windows: Credential Manager (DPAPI-protected)
- macOS: Keychain
- Linux: Secret Service (GNOME Keyring, KWallet, etc.)

Storage policy:
- Service name is constant: ``autocut``
- Username is the provider name (e.g. ``openrouter``)
- The secret value is the raw API key string

API keys are NEVER written to plaintext files anywhere in the codebase. If a
function in this module returns a key, the caller is responsible for never
logging it or echoing it to stdout. The redaction filter in
``autocut.security.redaction`` is a defence in depth, not the primary safeguard.
"""

from __future__ import annotations

from typing import Literal

import keyring
import keyring.errors

SERVICE_NAME = "autocut"

# v0.1.0 only stores a key for OpenRouter (the ``host`` provider does not use
# any API key — it delegates to the host agent). Future providers will extend
# this Literal.
SupportedProvider = Literal["openrouter"]


class KeyringError(RuntimeError):
    """Raised when the underlying OS keyring backend fails."""


def set_key(provider: SupportedProvider, key: str) -> None:
    """Store ``key`` for ``provider`` in the OS keyring, overwriting any previous value."""
    if not key or not key.strip():
        raise ValueError("API key cannot be empty")
    try:
        keyring.set_password(SERVICE_NAME, provider, key.strip())
    except keyring.errors.KeyringError as exc:
        raise KeyringError(f"failed to write key for {provider!r}: {exc}") from exc


def get_key(provider: SupportedProvider) -> str | None:
    """Return the stored key for ``provider``, or ``None`` if absent."""
    try:
        return keyring.get_password(SERVICE_NAME, provider)
    except keyring.errors.KeyringError as exc:
        raise KeyringError(f"failed to read key for {provider!r}: {exc}") from exc


def delete_key(provider: SupportedProvider) -> bool:
    """Remove the key for ``provider``. Returns ``True`` if a key was present."""
    try:
        existing = keyring.get_password(SERVICE_NAME, provider)
        if existing is None:
            return False
        keyring.delete_password(SERVICE_NAME, provider)
        return True
    except keyring.errors.PasswordDeleteError:
        # Some backends raise even though we just verified the entry exists.
        return False
    except keyring.errors.KeyringError as exc:
        raise KeyringError(f"failed to delete key for {provider!r}: {exc}") from exc


def list_providers_with_key() -> list[SupportedProvider]:
    """Return the providers that currently have a key set.

    The OS keyring does not offer a portable enumeration API, so we probe each
    known provider individually. This is fine for the small set we support.
    """
    known: tuple[SupportedProvider, ...] = ("openrouter",)
    return [p for p in known if get_key(p) is not None]
