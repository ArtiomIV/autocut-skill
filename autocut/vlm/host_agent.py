"""Host-agent VLM provider — delegates inference to Claude Code / Cowork.

This is the zero-cost path: when AutoCut runs inside an AI agent that already
has a vision-capable model on tap (Claude Code, Cowork), we don't make any
outbound API call. Instead the provider:

1. Writes a markdown brief to ``CLIPS/VLM_REQUEST.md`` describing the task,
   the schema, and the keyframe filenames the agent must look at.
2. Raises ``HostAgentPauseRequested`` so the CLI can print resume
   instructions and exit.
3. The host agent (the model running the conversation) reads the
   keyframes, generates a JSON ``ClipPlan``, and saves it to
   ``CLIPS/VLM_RESPONSE.json``.
4. The user runs ``autocut resume``. ``resume_from_disk`` parses the JSON
   and the pipeline picks up exactly where it left off.

There is no async network call in this path; ``analyze`` is async only to
match the ``VLMProvider`` interface.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import ClassVar

from pydantic import ValidationError

from autocut.models import AnalysisHints, ClipPlan, ContentHint, DetectionResult, Keyframe
from autocut.video.frame_sampler import FrameSpec
from autocut.vlm.base import (
    CostEstimate,
    HostAgentPauseRequested,
    VLMError,
    VLMProvider,
)
from autocut.vlm.prompts import (
    PROMPT_VERSION,
    build_system_prompt,
    build_user_prompt,
)

log = logging.getLogger(__name__)

REQUEST_FILENAME = "VLM_REQUEST.md"
RESPONSE_FILENAME = "VLM_RESPONSE.json"


class HostAgentProvider(VLMProvider):
    """Pauses the pipeline and asks the surrounding AI agent to fill in the JSON."""

    name: ClassVar[str] = "host"

    def __init__(self, work_dir: str | Path, *, agent_hint: str = "host-agent") -> None:
        """Create a host-agent provider that uses ``work_dir`` for the handoff files."""
        self._work_dir = Path(work_dir)
        self._agent_hint = agent_hint

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
        """Write the request file and pause; ``resume_from_disk`` does the rest."""
        if not keyframes:
            raise VLMError("host_agent.analyze called with no keyframes")

        self._work_dir.mkdir(parents=True, exist_ok=True)
        request_path = self._work_dir / REQUEST_FILENAME
        response_path = self._work_dir / RESPONSE_FILENAME

        specs = [
            FrameSpec(timestamp=kf.timestamp, scene_index=kf.scene_index or -1) for kf in keyframes
        ]
        markdown = _render_request_markdown(
            system_prompt=build_system_prompt(hints),
            user_prompt=build_user_prompt(
                video_id=video_id,
                duration_sec=duration_sec,
                hints=hints,
                specs=specs,
            ),
            keyframes=keyframes,
            response_path=response_path,
        )
        request_path.write_text(markdown, encoding="utf-8")
        log.info(
            "host_agent paused: wrote %s, waiting for %s",
            request_path,
            response_path,
        )
        raise HostAgentPauseRequested(request_path=request_path, response_path=response_path)

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
        """v0.1.0 stub: returns a low-confidence ``other`` so the pipeline falls back.

        Host-agent detection would require a second pause/resume cycle
        (DETECTION_REQUEST.md → DETECTION.json before VLM_REQUEST.md).
        That UX cost was judged too high for v0.1.0: users who want a
        specific profile can pass ``--content-hint`` explicitly, and the
        ``HYBRID_PROFILE`` fallback preserves the current behaviour for
        the ``auto`` case.

        When v0.1.x adds the second pause flow, this stub becomes the
        only implementation change needed — every other Phase E surface
        (prompts, models, pipeline wiring) is already capability-aware.
        """
        # The signature accepts every forward-compat arg; for the stub we
        # ignore them all explicitly so static analysers do not warn.
        del (
            keyframes,
            audio_description,
            video_id,
            duration_sec,
            timeout_sec,
            transcript_text,
            audio_clip_path,
            video_clip_paths,
        )
        log.warning(
            "host_agent.detect_content is a v0.1.0 stub — returning low-confidence "
            "'other'; pass --content-hint explicitly to skip detection or use "
            "--vlm openrouter for full auto-detect."
        )
        return DetectionResult(
            content_hint=ContentHint.other,
            confidence=0.0,
            reasoning="host-agent auto-detect is deferred to v0.1.x; HYBRID profile applied as fallback",
        )

    def estimate_cost(self, n_keyframes: int) -> CostEstimate:
        """Host-agent uses the existing AI subscription — zero marginal cost."""
        return CostEstimate(
            provider=self.name,
            model=self._agent_hint,
            n_input_images=n_keyframes,
            estimated_input_tokens=0,
            estimated_output_tokens=0,
            estimated_total_usd=0.0,
        )

    def health_check(self) -> bool:
        """Healthy iff we can create the work directory.

        We can't actually verify that an agent is listening (the whole point
        is that *something else* will pick up the request file), so success
        here just means the handoff path is writable.
        """
        try:
            self._work_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            log.debug("host_agent work dir not writable: %s", exc)
            return False
        return True

    # ------------------------------------------------------------------
    # Resume path
    # ------------------------------------------------------------------

    def resume_from_disk(
        self,
        *,
        request_path: Path | None = None,
        response_path: Path | None = None,
    ) -> ClipPlan:
        """Load and validate the JSON the host agent wrote.

        ``request_path`` is accepted but not parsed — it's kept for symmetry
        and for future use (e.g. echoing back which version of the prompt
        produced this response).
        """
        del request_path  # currently unused
        rp = response_path or (self._work_dir / RESPONSE_FILENAME)
        if not rp.is_file():
            raise VLMError(
                f"host_agent response file not found: {rp} "
                f"(ask the agent to write the ClipPlan JSON there, then re-run)"
            )
        try:
            payload = json.loads(rp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise VLMError(f"failed to read host_agent response: {exc}") from exc
        if not isinstance(payload, dict):
            raise VLMError("host_agent response top-level value is not an object")

        metadata = payload.setdefault("metadata", {})
        if not isinstance(metadata, dict):
            raise VLMError("host_agent response metadata field is not an object")
        metadata.setdefault("vlm_provider", self.name)
        metadata.setdefault("vlm_model", self._agent_hint)
        metadata.setdefault("prompt_version", PROMPT_VERSION)

        try:
            return ClipPlan.model_validate(payload)
        except ValidationError as exc:
            raise VLMError(f"host_agent response failed ClipPlan validation: {exc}") from exc


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


_REQUEST_TEMPLATE = """\
# AutoCut — host-agent VLM request

The AutoCut pipeline is paused. You (the AI agent running this conversation)
are being asked to do the vision analysis step on the agent subscription
instead of calling an external API.

## What to do

1. Read each keyframe image listed below with your file/image tool.
2. Decide which segments of the source video are worth keeping as
   highlight clips, applying the constraints in the system prompt and
   the schema in the user prompt (both reproduced below).
3. Write the result as a single JSON object matching the ``ClipPlan``
   schema to:

   `{response_path}`

4. Tell the user to run `autocut resume`.

Do not output anything else. The response file must contain only the JSON.

## Keyframes

{keyframe_listing}

## System prompt (verbatim)

```
{system_prompt}
```

## User prompt (verbatim — includes the JSON schema you must match)

```
{user_prompt}
```
"""


def _render_request_markdown(
    *,
    system_prompt: str,
    user_prompt: str,
    keyframes: list[Keyframe],
    response_path: Path,
) -> str:
    listing_lines: list[str] = []
    for i, kf in enumerate(keyframes, start=1):
        scene_tag = f"scene {kf.scene_index}" if kf.scene_index is not None else "uniform sample"
        listing_lines.append(f"{i}. `{kf.path}` — t = {_format_ts(kf.timestamp)} ({scene_tag})")
    return _REQUEST_TEMPLATE.format(
        response_path=response_path,
        keyframe_listing="\n".join(listing_lines),
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


def _format_ts(ts: object) -> str:
    """Render any ``timedelta``-like value to ``HH:MM:SS.mmm`` for the brief."""
    seconds = getattr(ts, "total_seconds", lambda: float(ts))()  # type: ignore[arg-type]
    total_ms = max(0, int(float(seconds) * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds_part, milliseconds = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}.{milliseconds:03d}"
