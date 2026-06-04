"""Unit tests for ``autocut.vlm.factory`` — dispatch + friendly error path."""

from __future__ import annotations

import pytest

from autocut.vlm import (
    OpenRouterProvider,
    UnavailableProviderError,
    VLMError,
    make_provider,
)


def test_host_points_to_local_recipe() -> None:
    # `--vlm host` is no longer a provider: it redirects to the sheet/cut recipe.
    with pytest.raises(UnavailableProviderError, match="sheet"):
        make_provider("host", model="claude-opus")


def test_dispatches_to_openrouter() -> None:
    provider = make_provider(
        "openrouter",
        model="google/gemini-2.5-flash",
        api_key="sk-or-v1-test-1234567890abcdef",
    )
    assert isinstance(provider, OpenRouterProvider)
    assert provider.model == "google/gemini-2.5-flash"


def test_openrouter_without_key_in_keyring_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("autocut.vlm.factory.get_key", lambda _: None)
    with pytest.raises(VLMError, match="API key"):
        make_provider("openrouter", model="x")


@pytest.mark.parametrize(
    ("provider", "expected_suggestion"),
    [
        ("anthropic", "anthropic/claude"),
        ("openai", "openai/gpt-4o"),
        ("gemini", "google/gemini"),
    ],
)
def test_deferred_cloud_providers_suggest_openrouter(
    provider: str, expected_suggestion: str
) -> None:
    with pytest.raises(UnavailableProviderError, match=expected_suggestion):
        make_provider(provider, model="x")


@pytest.mark.parametrize("provider", ["ollama", "lmstudio"])
def test_deferred_local_providers_point_to_openrouter(provider: str) -> None:
    with pytest.raises(UnavailableProviderError, match="openrouter"):
        make_provider(provider, model="x")


def test_unknown_provider_errors_with_known_list() -> None:
    with pytest.raises(UnavailableProviderError, match="openrouter"):
        make_provider("bogus", model="x")


def test_provider_name_is_normalised() -> None:
    # "  HOST  " still resolves to the host branch (stripped + lowered).
    with pytest.raises(UnavailableProviderError, match="sheet"):
        make_provider("  HOST  ", model="x")
