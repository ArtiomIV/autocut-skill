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

from autocut.models import AnalysisHints, ClipPlan, DetectionResult, Keyframe
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
    build_video_user_prompt,
)

log = logging.getLogger(__name__)

REQUEST_FILENAME = "VLM_REQUEST.md"
RESPONSE_FILENAME = "VLM_RESPONSE.json"

# Phase E (host-agent detection) request/response filenames. Distinct from
# the analysis ones so both phases can leave artefacts in the same work
# dir without colliding, and so the CLI can tell which phase to resume.
DETECTION_REQUEST_FILENAME = "DETECTION_REQUEST.md"
DETECTION_RESPONSE_FILENAME = "DETECTION_RESPONSE.json"


class HostAgentProvider(VLMProvider):
    """Pauses the pipeline and asks the surrounding AI agent to fill in the JSON."""

    name: ClassVar[str] = "host"

    def __init__(
        self,
        work_dir: str | Path,
        *,
        agent_hint: str = "host-agent",
        supports_video: bool = False,
    ) -> None:
        """Create a host-agent provider that uses ``work_dir`` for the handoff files.

        ``supports_video`` declares whether the surrounding AI agent can ingest
        a video file directly. There is no live capability endpoint for the
        host (unlike OpenRouter's ``/models``), so this is opt-in: the user
        sets it via ``autocut run --host-video`` only when they know the agent
        can watch an MP4. Default ``False`` keeps the keyframe (stills) path.
        """
        self._work_dir = Path(work_dir)
        self._agent_hint = agent_hint
        self._supports_video = supports_video

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

    async def supports_video(self) -> bool:
        """Whether the surrounding agent was declared video-capable (opt-in flag)."""
        return self._supports_video

    async def analyze_video_clip(
        self,
        clip_path: Path,
        hints: AnalysisHints,
        *,
        video_id: str,
        clip_duration_sec: float,
        timeout_sec: int = 300,
    ) -> ClipPlan:
        """Write a VIDEO request and pause; the agent watches the MP4 directly.

        The host transport has no base64 size ceiling (the agent reads the file
        from disk, not an inline payload), so this is a single-pass handoff: the
        whole compressed clip is referenced by path and the agent returns one
        ``ClipPlan`` with timestamps relative to the clip — which, for the
        whole-video single pass the pipeline uses today, equal absolute source
        time. ``resume_from_disk`` (shared with the keyframe path) parses the
        response. Raises ``HostAgentPauseRequested`` exactly like ``analyze``.
        """
        del timeout_sec  # the host pause has no network timeout
        if not clip_path.is_file():
            raise VLMError(f"host_agent.analyze_video_clip: video clip not found: {clip_path}")

        self._work_dir.mkdir(parents=True, exist_ok=True)
        request_path = self._work_dir / REQUEST_FILENAME
        response_path = self._work_dir / RESPONSE_FILENAME

        markdown = _render_video_request_markdown(
            system_prompt=build_system_prompt(hints),
            user_prompt=build_video_user_prompt(
                video_id=video_id,
                duration_sec=clip_duration_sec,
                hints=hints,
            ),
            clip_path=clip_path,
            response_path=response_path,
        )
        request_path.write_text(markdown, encoding="utf-8")
        log.info(
            "host_agent paused (video): wrote %s referencing %s, waiting for %s",
            request_path,
            clip_path,
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
        """Pause for the host agent to classify the content type.

        Writes ``DETECTION_REQUEST.md`` next to the keyframes, raises
        ``HostAgentPauseRequested``, and expects ``DETECTION_RESPONSE.json``
        to be filled in by the surrounding agent before the user runs
        ``autocut resume``. The resume command catches the detection
        phase, applies the classification, then re-enters the pipeline
        for the standard analysis pause.

        ``timeout_sec`` and the forward-compat audio/video/transcript
        kwargs are accepted but unused: this implementation is a synchronous
        pause and only passes the textual audio description to the agent.
        """
        del timeout_sec, transcript_text, audio_clip_path, video_clip_paths

        if not keyframes:
            raise VLMError("host_agent.detect_content called with no keyframes")

        self._work_dir.mkdir(parents=True, exist_ok=True)
        request_path = self._work_dir / DETECTION_REQUEST_FILENAME
        response_path = self._work_dir / DETECTION_RESPONSE_FILENAME

        markdown = _render_detection_request_markdown(
            video_id=video_id,
            duration_sec=duration_sec,
            audio_description=audio_description,
            keyframes=keyframes,
            response_path=response_path,
        )
        request_path.write_text(markdown, encoding="utf-8")
        log.info(
            "host_agent paused for detection: wrote %s, waiting for %s",
            request_path,
            response_path,
        )
        raise HostAgentPauseRequested(request_path=request_path, response_path=response_path)

    def resume_detection_from_disk(
        self,
        *,
        request_path: Path | None = None,
        response_path: Path | None = None,
    ) -> DetectionResult:
        """Load and validate the detection JSON the host agent wrote.

        Mirrors ``resume_from_disk`` but expects a ``DetectionResult``
        schema (``content_hint`` / ``confidence`` / ``reasoning``) instead
        of a ``ClipPlan``. The lenient aliasing applied to the openrouter
        path is replicated here so the host agent's response stays robust
        against minor schema drift (e.g. an agent writing ``category``
        instead of ``content_hint``).
        """
        del request_path  # currently unused
        rp = response_path or (self._work_dir / DETECTION_RESPONSE_FILENAME)
        if not rp.is_file():
            raise VLMError(
                f"host_agent detection response file not found: {rp} "
                f"(ask the agent to write the DetectionResult JSON there, then re-run)"
            )
        try:
            payload = json.loads(rp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise VLMError(f"failed to read host_agent detection response: {exc}") from exc
        if not isinstance(payload, dict):
            raise VLMError("host_agent detection response top-level value is not an object")

        # Apply the same lenient aliases the openrouter parser uses so the
        # host agent has identical wiggle room with the schema.
        if "content_hint" not in payload:
            for alias in ("category", "content_type", "type", "label"):
                if alias in payload:
                    payload["content_hint"] = payload.pop(alias)
                    break
        if isinstance(payload.get("content_hint"), str):
            payload["content_hint"] = payload["content_hint"].lower().strip()
        if isinstance(payload.get("confidence"), str):
            raw_conf = payload["confidence"].strip().rstrip("%")
            try:
                value = float(raw_conf)
            except ValueError as exc:
                raise VLMError(
                    f"host_agent detection confidence non-numeric: {payload['confidence']!r}"
                ) from exc
            payload["confidence"] = value / 100.0 if value > 1.0 else value

        try:
            return DetectionResult.model_validate(payload)
        except ValidationError as exc:
            raise VLMError(
                f"host_agent detection response failed DetectionResult validation: {exc}"
            ) from exc

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


_VIDEO_REQUEST_TEMPLATE = """\
# AutoCut — host-agent VLM request (VIDEO)

The AutoCut pipeline is paused. You (the AI agent running this conversation)
are being asked to do the vision analysis step on the agent subscription
instead of calling an external API.

## What to do

1. Watch the video clip with your video/file tool:

   `{clip_path}`

   If you CANNOT open or analyse a video file, do NOT guess from a single
   frame. Stop and tell the user to re-run `autocut run` WITHOUT the
   `--host-video` flag so AutoCut falls back to the keyframe (stills) path.

2. Decide which segments are worth keeping as highlight clips, applying the
   constraints in the system prompt and the schema in the user prompt (both
   reproduced below). Timestamps are RELATIVE to this clip — its first frame
   is 00:00:00.000.

3. Write the result as a single JSON object matching the ``ClipPlan`` schema
   to:

   `{response_path}`

4. Tell the user to run `autocut resume`.

Do not output anything else. The response file must contain only the JSON.

## System prompt (verbatim)

```
{system_prompt}
```

## User prompt (verbatim — includes the JSON schema you must match)

```
{user_prompt}
```
"""


def _render_video_request_markdown(
    *,
    system_prompt: str,
    user_prompt: str,
    clip_path: Path,
    response_path: Path,
) -> str:
    return _VIDEO_REQUEST_TEMPLATE.format(
        clip_path=clip_path,
        response_path=response_path,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


_DETECTION_REQUEST_TEMPLATE = """\
# AutoCut — host-agent CONTENT DETECTION request (Phase E)

The AutoCut pipeline is paused at the **detection** step. Before running
the expensive analysis pass, AutoCut needs you to classify the video so
it can pick the right highlight-extraction profile (sport vs talk vs
hybrid). This is the first of two pauses; the second is the standard
clip-analysis request and will come after you resume.

## What to do

1. Read each detection keyframe image listed below with your file/image tool.
2. Use the audio profile description (next section) as additional signal.
3. Pick ONE content type from the enum and write the JSON object to:

   `{response_path}`

4. Tell the user to run `autocut resume`. AutoCut will apply the
   classification, then pause again with the regular `VLM_REQUEST.md`
   so you can produce the ClipPlan.

Do NOT output anything else in the response file. JSON only.

## Schema you must match

```
{{
  "content_hint": "highlights|talk|hybrid",
  "confidence":  <number 0.0..1.0>,
  "reasoning":   "<one short sentence describing the dominant signal>"
}}
```

Guidance:
- ``content_hint`` is the editing MODE: ``highlights`` for footage whose value
  is its best visual moments (sport/action/gameplay/reactions), ``talk`` for
  speech-driven content (interview/podcast/monologue), ``hybrid`` for anything
  unclear or mixed.
- ``confidence`` is your self-assessment. Use >= 0.8 only when visual +
  audio signals clearly agree.
- When undecided between two modes, prefer ``hybrid`` over a guess.

## Audio profile (computed from the waveform, no transcription)

```
{audio_description}
```

## Detection keyframes (stratified random, spanning the full timeline)

{keyframe_listing}
"""


def _render_detection_request_markdown(
    *,
    video_id: str,
    duration_sec: float,
    audio_description: str,
    keyframes: list[Keyframe],
    response_path: Path,
) -> str:
    """Build the markdown brief the host agent reads to classify the video."""
    # ``video_id`` is logged downstream but not used in the body today;
    # accept it so the call site mirrors ``_render_request_markdown``.
    del video_id, duration_sec
    listing_lines: list[str] = []
    for i, kf in enumerate(keyframes, start=1):
        listing_lines.append(f"{i}. `{kf.path}` — t = {_format_ts(kf.timestamp)}")
    return _DETECTION_REQUEST_TEMPLATE.format(
        response_path=response_path,
        audio_description=audio_description.rstrip(),
        keyframe_listing="\n".join(listing_lines),
    )


def _format_ts(ts: object) -> str:
    """Render any ``timedelta``-like value to ``HH:MM:SS.mmm`` for the brief."""
    seconds = getattr(ts, "total_seconds", lambda: float(ts))()  # type: ignore[arg-type]
    total_ms = max(0, int(float(seconds) * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds_part, milliseconds = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}.{milliseconds:03d}"
