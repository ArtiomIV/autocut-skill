"""Shared VLM types: the provider interface and the exceptions it can raise.

Every concrete provider (``openrouter``, future
``anthropic``/``openai``/``gemini``/``ollama``/``lmstudio``) implements
``VLMProvider`` so the pipeline can stay provider-agnostic.

Two return shapes are possible from ``analyze()``:

1. A validated ``ClipPlan`` (the happy path).
2. ``VLMError`` for anything that goes wrong with a cloud call.

The local/host path no longer runs through a provider: the orchestrating agent
drives the deterministic ``probe``/``sheet``/``cut``/``merge`` subcommands itself
(see the ``autocut-run`` skill), so there is no pause/resume sentinel here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from autocut.models import AnalysisHints, ClipPlan, DetectionResult, Keyframe


class VLMError(RuntimeError):
    """Raised when a cloud VLM call fails (network, auth, schema, timeout)."""


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

        Implementations must raise ``VLMError`` for cloud failures.
        """

    @abstractmethod
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
        """Classify the video content type using a small set of detection keyframes.

        ``audio_description`` is a short text block built by the detector
        from the waveform statistics (no transcription in v0.1.0); providers
        feed it to the model as a system/user-prompt fragment.

        ``transcript_text`` / ``audio_clip_path`` / ``video_clip_paths`` are
        reserved for future capability-aware paths (Whisper-light in v0.2.0,
        Gemini ``input_audio``/video upload). v0.1.0 implementations may
        ignore them; the signature is forward-compatible so callers do not
        need to change when the audio/video paths land.

        Implementations must raise ``VLMError`` for cloud failures or schema
        violations.
        """

    async def supports_video(self) -> bool:
        """Whether this provider can ingest a video clip directly.

        Default ``False``: stills-based providers (host agent today) take the
        keyframe path. Providers that override this to ``True`` MUST implement
        ``analyze_video_clip``. The pipeline calls this to choose the payload.
        """
        return False

    async def analyze_video_clip(
        self,
        clip_path: Path,
        hints: AnalysisHints,
        *,
        video_id: str,
        clip_duration_sec: float,
        timeout_sec: int = 300,
    ) -> ClipPlan:
        """Analyse a single (compressed) video clip → ``ClipPlan``.

        Only meaningful for providers whose ``supports_video`` returns ``True``;
        the default raises so a mis-routed call fails loudly rather than
        silently. Timestamps are returned RELATIVE to the clip.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support direct video analysis")

    async def analyze_contact_sheets(
        self,
        sheets: list[Path],
        hints: AnalysisHints,
        *,
        video_id: str,
        duration_sec: float,
        frame_times: list[float],
        timeout_sec: int = 300,
    ) -> ClipPlan:
        """Analyse a candidate window rendered as indexed contact sheet(s).

        Used by the two-pass fine pass on the video route: each sheet is a grid of
        small frames, each cell labelled with its index, paired with a
        ``frame_times`` index->time map. Only meaningful for providers that also
        implement ``analyze_video_clip``; the default raises so a mis-routed call
        fails loudly. Timestamps are returned RELATIVE to the clip.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support contact-sheet analysis")

    async def supports_audio(self) -> bool:
        """Whether this provider can ingest an audio clip directly.

        Default ``False``. Providers that override this to ``True`` MUST
        implement ``analyze_audio_clip``. Used for the talk/podcast path where
        the model hears speech instead of looking at frames.
        """
        return False

    async def analyze_audio_clip(
        self,
        clip_path: Path,
        hints: AnalysisHints,
        *,
        video_id: str,
        clip_duration_sec: float,
        timeout_sec: int = 300,
    ) -> ClipPlan:
        """Analyse a single (extracted) audio clip → ``ClipPlan``.

        Only meaningful for providers whose ``supports_audio`` returns ``True``;
        the default raises so a mis-routed call fails loudly. Timestamps are
        returned RELATIVE to the clip.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support direct audio analysis")

    @abstractmethod
    def estimate_cost(self, n_keyframes: int) -> CostEstimate:
        """Return a best-effort pre-call cost estimate.

        The pipeline uses this to enforce ``config.security.cost_cap_usd``.
        Providers that do not bill per call (``host_agent``) return zero.
        """

    @abstractmethod
    def health_check(self) -> bool:
        """Cheap probe: ``True`` if the provider is reachable / configured."""
