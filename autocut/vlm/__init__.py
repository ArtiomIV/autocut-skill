"""Vision Language Model providers and prompt templates."""

from autocut.vlm.base import (
    CostEstimate,
    HostAgentPauseRequested,
    VLMError,
    VLMProvider,
)
from autocut.vlm.discovery import ModelInfo, list_openrouter_models
from autocut.vlm.factory import UnavailableProviderError, make_provider
from autocut.vlm.host_agent import HostAgentProvider
from autocut.vlm.openrouter import OpenRouterProvider

__all__ = [
    "CostEstimate",
    "HostAgentPauseRequested",
    "HostAgentProvider",
    "ModelInfo",
    "OpenRouterProvider",
    "UnavailableProviderError",
    "VLMError",
    "VLMProvider",
    "list_openrouter_models",
    "make_provider",
]
