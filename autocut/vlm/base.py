"""Shared VLM types: the provider interface and the exceptions it can raise.

Every concrete provider (``openrouter``, ``host_agent``, future
``anthropic``/``openai``/``gemini``/``ollama``/``lmstudio``) implements
``VLMProvider`` so the pipeline can stay provider-agnostic.

Three return shapes are possible from ``analyze()``:

1. A validated ``ClipPlan`` (the happy path).
2. ``VLMError`` for anything that goes wrong with a cloud call.
3. ``HostAgentPauseRequested`` for the host-agent flow: the analyze call
   writes a request file to disk, raises this sentinel, and the CLI shows
   the user instructions. The host agent (Claude Code / Cowork) reads the
   request, writes ``VLM_RESPONSE.json``, and the user runs ``autocut
   resume`` to continue.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from autocut.models import AnalysisHints, ClipPlan, Keyframe


class VLMError(RuntimeError):
    """Raised when a cloud VLM call fails (network, auth, schema, timeout)."""


class HostAgentPauseRequested(Exception):  # noqa: N818
    # Intentional control-flow signal, not an error condition; the "Error"
    # suffix N818 wants would mislead readers.
    """Raised by the host-agent provider to pause the pipeline.

    Carries the path of the request file the host agent must read, plus the
    path where the response JSON is expected.
    """

    def __init__(self, request_path: Path, response_path: Path) -> None:
        super().__init__(
            f"host-agent provider paused: write the result to {response_path} "
            f"then run `autocut resume`"
        )
        self.request_path = request_path
        self.response_path = response_path


class CostEstimate(BaseModel):
    """Pre-call estimate of how much a VLM analysis will cost."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    n_input_images: int = Field(ge=0)
    estimated_input_tokens: int = Field(ge=0)
    estimated_output_tokens: int = Field(ge=0)
    estimated_total_usd: float = Field(ge=0)

    @property
    def is_free(self) -> bool:
        """``True`` for providers like ``host`` that don't bill per call."""
        return self.estimated_total_usd == 0


class VLMProvider(ABC):
    """Common interface every VLM provider implements.

    The pipeline holds a single provider instance per run and calls
    ``analyze`` once with the full keyframe set (or per-chunk for very long
    videos — handled in M3 by the pipeline itself).
    """

    #: Stable identifier used in logs, manifests, and the model selector.
    name: ClassVar[str]

    @abstractmethod
    async def analyze(
        self,
        keyframes: list[Keyframe],
        hints: AnalysisHints,
        *,
        video_id: str,
        duration_sec: float,
        timeout_sec: int = 300,
    ) -> ClipPlan:
        """Send keyframes to the VLM and return a validated ``ClipPlan``.

        ``video_id`` is a stable identifier the model echoes back in the
        ``ClipPlan``. ``duration_sec`` is the full source-video length, so
        the model can place timestamps in absolute terms.

        Implementations must raise ``VLMError`` for cloud failures and
        ``HostAgentPauseRequested`` for the deferred host-agent flow.
        """

    @abstractmethod
    def estimate_cost(self, n_keyframes: int) -> CostEstimate:
        """Return a best-effort pre-call cost estimate.

        The pipeline uses this to enforce ``config.security.cost_cap_usd``.
        Providers that do not bill per call (``host_agent``) return zero.
        """

    @abstractmethod
    def health_check(self) -> bool:
        """Cheap probe: ``True`` if the provider is reachable / configured."""
