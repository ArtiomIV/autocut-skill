"""Unit tests for ``autocut.vlm.factory`` — dispatch + friendly error path."""

from __future__ import annotations

from pathlib import Path

import pytest

from autocut.vlm import (
    HostAgentProvider,
    OpenRouterProvider,
    UnavailableProviderError,
    VLMError,
    make_provider,
)


def test_dispatches_to_host(tmp_path: Path) -> None:
    provider = make_provider("host", model="claude-opus", work_dir=tmp_path)
    assert isinstance(provider, HostAgentProvider)
    assert provider.name == "host"


def test_host_requires_work_dir() -> None:
    with pytest.raises(VLMError, match="work_dir"):
        make_provider("host", model="claude-opus")


def test_dispatches_to_openrouter(tmp_path: Path) -> None:
    provider = make_provider(
        "openrouter",
        model="google/gemini-2.5-flash",
        api_key="sk-or-v1-test-1234567890abcdef",
        work_dir=tmp_path,
    )
    assert isinstance(provider, OpenRouterProvider)
    assert provider.model == "google/gemini-2.5-flash"


def test_openrouter_without_key_in_keyring_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("autocut.vlm.factory.get_key", lambda _: None)
    with pytest.raises(VLMError, match="API key"):
        make_provider("openrouter", model="x", work_dir=tmp_path)


@pytest.mark.parametrize(
    ("provider", "expected_suggestion"),
    [
        ("anthropic", "anthropic/claude"),
        ("openai", "openai/gpt-4o"),
        ("gemini", "google/gemini"),
    ],
)
def test_deferred_cloud_providers_suggest_openrouter(
    provider: str, expected_suggestion: str, tmp_path: Path
) -> None:
    with pytest.raises(UnavailableProviderError, match=expected_suggestion):
        make_provider(provider, model="x", work_dir=tmp_path)


@pytest.mark.parametrize("provider", ["ollama", "lmstudio"])
def test_deferred_local_providers_point_to_openrouter(provider: str, tmp_path: Path) -> None:
    with pytest.raises(UnavailableProviderError, match="openrouter"):
        make_provider(provider, model="x", work_dir=tmp_path)


def test_unknown_provider_errors_with_known_list(tmp_path: Path) -> None:
    with pytest.raises(UnavailableProviderError, match="host, openrouter"):
        make_provider("bogus", model="x", work_dir=tmp_path)


def test_provider_name_is_normalised(tmp_path: Path) -> None:
    provider = make_provider("  HOST  ", model="x", work_dir=tmp_path)
    assert isinstance(provider, HostAgentProvider)
