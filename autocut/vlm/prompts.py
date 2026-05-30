"""Prompt templates and the JSON schema we hand to the VLM.

Two design choices that matter for cost and correctness:

1. The JSON schema is derived from ``ClipPlan`` so the prompt and the
   downstream validator can never drift apart. If we add a field to
   ``Clip`` tomorrow, the schema in the prompt updates automatically.
2. The prompt is versioned (``PROMPT_VERSION``). Output ``ClipPlan``
   records carry the version they were generated with, so when we
   iterate on phrasing we can audit which prompt produced which run.
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Final

from autocut.models import AnalysisHints, ClipPlan, PromptTemplateId
from autocut.video.frame_sampler import FrameSpec

# v3 (2026-05-29): cap the sport lead-in at ~1-2s (no long approach/circling/
# idle pause before the action) so clips stop drifting long and loose. Pairs
# with temperature=0 on the analysis calls for run-to-run consistency.
# v2 (2026-05-28): reworked sport/talk clip-boundary guidance so the model
# includes the wind-up and follow-through of the key moment instead of cutting
# tight on the impact. The detection prompt is unchanged, hence its own version.
PROMPT_VERSION: Final[str] = "v3"
DETECTION_PROMPT_VERSION: Final[str] = "v1"


# ---------------------------------------------------------------------------
# JSON schema (derived from ClipPlan)
# ---------------------------------------------------------------------------


def clip_plan_schema() -> str:
    """Return the JSON Schema for ``ClipPlan`` as a compact string."""
    schema = ClipPlan.model_json_schema()
    return json.dumps(schema, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


_CORE_RULES: Final[str] = """\
You are a video editor assistant analysing keyframes extracted from a video.
You will receive an ordered list of keyframes, each labelled with the
timestamp at which it was sampled from the source video.

Your job: identify the segments worth keeping as standalone highlight clips,
then return them as STRICT JSON matching the provided schema. No prose
before or after the JSON. No explanations. No markdown fences.

Hard rules:
- Each clip must be between {min_dur:.1f} and {max_dur:.1f} seconds long.
- Clips must NOT overlap.
- ``start`` and ``end`` use the ``HH:MM:SS.mmm`` format.
- ``score`` is an integer 0-10 (10 = mandatory, must keep).
- ``category`` is one of: highlight, reaction, dialogue, transition, filler.
- ``tags`` is a list of 1-5 short strings.
- Be conservative: prefer fewer high-score clips over many low-score ones.
- If no segment is worth keeping, return an empty ``clips`` array.
"""


_SPORT_GUIDANCE: Final[str] = """\

Content guidance — sport / combat / action:
- An action is only confirmed if you can see it across AT LEAST TWO
  consecutive keyframes. A single frame that looks like a punch, kick,
  or impact is often a feint, study phase, or a missed strike — do NOT
  score it >= 7 alone.
- Score >= 7 examples: multi-frame exchanges with motion blur on more
  than one body, clinches with clear body contact, knockdowns,
  follow-through punches landing, escalating combinations.
- Score < 7 examples: defensive guard, ref breaks, fighters circling
  or studying each other, single isolated jab without follow-up,
  movement without engagement.
- Cut tight. Start AT MOST ~1-2 seconds before the decisive action —
  just enough to make the wind-up of the landing combination legible,
  NEVER more. Do NOT include the long approach, the circling, the
  studying, or any idle pause before the action begins — those dead
  seconds make the clip drag. ALWAYS include the immediate
  follow-through and reaction (the opponent reeling, the referee
  stepping in, the celebration), and NEVER end on the frame of impact
  itself — the aftermath is what makes the clip land.
"""


_TALK_GUIDANCE: Final[str] = """\

Content guidance — talk / podcast / interview:
- Frames carry no audio, so judge on visual cues: speaker leaning
  forward, expressive hand gestures, eye contact intensity, reaction
  shots of the listener (laughter, surprise, agreement).
- Score >= 7 examples: speaker emphatically gesturing during a key
  point, audience or interviewer reacting visibly (laughter,
  surprise), close-up emotional moments, shared moments of agreement.
- Score < 7 examples: static talking heads with no visible energy,
  transitions between speakers, neutral discussion frames.
- Clips can be longer here: a good moment needs both setup AND payoff
  to read on its own. Start a beat before the key line or gesture so it
  has context, and ALWAYS include the reaction that follows (the laugh,
  the nod, the stunned pause, the interviewer's response). NEVER end on
  the punchline frame itself — the reaction is what makes it land.
"""


_HYBRID_GUIDANCE: Final[str] = """\

Content guidance — mixed / unknown content:
- Apply general highlight judgment: pick visually distinctive moments
  that would read as standalone clips out of context.
- When uncertain about the content type, lean conservative on scores
  and prefer fewer well-justified clips over many speculative ones.
"""


_SYSTEM_TEMPLATES: Final[dict[PromptTemplateId, str]] = {
    "sport": _CORE_RULES + _SPORT_GUIDANCE,
    "talk": _CORE_RULES + _TALK_GUIDANCE,
    "hybrid": _CORE_RULES + _HYBRID_GUIDANCE,
}


_USER_TEMPLATE: Final[str] = """\
Video metadata: content type = {content_hint}, language = {language},
goal = {goal}, source duration = {duration_sec:.2f}s.

Keyframes (in chronological order):
{keyframe_lines}
{motion_block}
Return JSON matching this schema (do not output the schema itself, output
your analysis):
{schema}

Fill ``video_id`` with {video_id!r} and ``duration_sec`` with {duration_sec}.
Leave ``metadata`` as an empty object — provenance fields (provider, model,
prompt version, timing) are filled in by the caller.
"""


# Used by the video-input path: no keyframe list (the model watches the clip)
# and no motion block (a video conveys motion directly). Timestamps come back
# relative to the clip; the batch engine re-bases them to absolute source time.
_VIDEO_USER_TEMPLATE: Final[str] = """\
Video metadata: content type = {content_hint}, language = {language},
goal = {goal}, clip duration = {duration_sec:.2f}s.

You are watching the full video clip provided. Identify the segments worth
keeping as standalone highlight clips. Report each ``start``/``end`` RELATIVE
to THIS clip, where 00:00:00.000 is the first frame you see.

Return JSON matching this schema (do not output the schema itself, output
your analysis):
{schema}

Fill ``video_id`` with {video_id!r} and ``duration_sec`` with {duration_sec}.
Leave ``metadata`` as an empty object — provenance fields (provider, model,
prompt version, timing) are filled in by the caller.
"""


# Used by the audio-input path: the model LISTENS to the clip (no keyframes,
# no visual cues). The signal is the spoken content, so the whole framing
# differs from the keyframe/video prompts — judge what is SAID.
_AUDIO_SYSTEM: Final[str] = """\
You are a video editor assistant. You are LISTENING to the audio track of a
video clip — speech, tone of voice, laughter, applause, pauses. You CANNOT see
the picture; judge purely on what is said and how it is said.

Your job: SELECT only the few standout highlight moments — the ones that would
land OUT OF CONTEXT on social media. This is highlight SELECTION, NOT
transcription and NOT segmentation: most of a talk is ordinary conversation that
is NOT a highlight, so most of the timeline produces NO clip at all. Be highly
selective. Return the result as STRICT JSON matching the provided schema. No
prose before or after the JSON. No markdown fences.

Hard rules:
- Each clip must be between {min_dur:.1f} and {max_dur:.1f} seconds long.
- Clips must NOT overlap.
- ``start`` and ``end`` use the ``HH:MM:SS.mmm`` format, RELATIVE to this clip
  (its first moment is 00:00:00.000).
- ``score`` is an integer 0-10 (10 = mandatory, must keep).
- Include a clip ONLY if you would score it 7 or higher (the viral test below).
  OMIT every segment you would score below 7 — do NOT return filler, routine
  discussion, setup, or ordinary talking.
- ``category`` is one of: highlight, reaction, dialogue, transition, filler.
- ``tags`` is a list of 1-5 short strings.
- If no segment is worth keeping, return an empty ``clips`` array.

Content guidance — find the VIRAL moments (the Instagram / TikTok scroll test):
- THE BAR: picture someone scrolling their feed. Would THIS clip make them STOP
  and watch? Score >= 7 only for moments that pass that test — a sharp
  punchline, a bold or surprising claim, a funny exchange or comeback, an
  emotional or personal reveal, a vivid story beat, a clear laugh/applause
  reaction. These are the clips that would "catch" a viewer mid-scroll.
- AIM FOR THE RIGHT AMOUNT — neither too few nor too many. Capture EVERY
  genuinely scroll-stopping moment, and ONLY those. Do NOT clip ordinary
  discussion, logistics, one-word answers or banter ("yes", "ok", "exactly"),
  hesitations, or plain setup — those score below 7 and must be omitted. A long
  talk usually has somewhere between a handful and a couple dozen real viral
  moments: not one per sentence, but not just one or two either.
- Give each clip enough lead-in for the hook to make sense AND the payoff or
  reaction that follows. NEVER cut mid-sentence.
- Put the quotable line or a tight summary in ``description`` so a human
  scanning the manifest knows what was said.
"""


_AUDIO_USER_TEMPLATE: Final[str] = """\
Audio metadata: content type = {content_hint}, language = {language},
goal = {goal}, clip duration = {duration_sec:.2f}s.

You are listening to the full audio clip provided. Identify the spoken segments
worth keeping as standalone highlight clips. Report each ``start``/``end``
RELATIVE to THIS clip, where 00:00:00.000 is the first moment you hear.

Return JSON matching this schema (do not output the schema itself, output
your analysis):
{schema}

Fill ``video_id`` with {video_id!r} and ``duration_sec`` with {duration_sec}.
Leave ``metadata`` as an empty object — provenance fields (provider, model,
prompt version, timing) are filled in by the caller.
"""


def build_audio_system_prompt(hints: AnalysisHints) -> str:
    """System message for the audio-input path (listen, hunt viral spoken moments)."""
    return _AUDIO_SYSTEM.format(min_dur=hints.min_duration_sec, max_dur=hints.max_duration_sec)


def build_audio_user_prompt(
    *,
    video_id: str,
    duration_sec: float,
    hints: AnalysisHints,
) -> str:
    """User message for the audio-input path. Timestamps are clip-relative."""
    return _AUDIO_USER_TEMPLATE.format(
        content_hint=hints.content_hint.value,
        language=hints.language,
        goal=hints.goal,
        duration_sec=duration_sec,
        schema=clip_plan_schema(),
        video_id=video_id,
    )


def format_timestamp(ts: timedelta) -> str:
    """Format ``HH:MM:SS.mmm`` so timestamps in the prompt line up with the schema."""
    total_ms = max(0, int(ts.total_seconds() * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, milliseconds = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def build_system_prompt(hints: AnalysisHints) -> str:
    """Return the system message: core rules + per-template guidance.

    Template selection is driven by ``hints.prompt_template`` (set from the
    active ``ContentProfile`` in the pipeline). Unknown template IDs fall
    back to the ``hybrid`` variant defensively.
    """
    template = _SYSTEM_TEMPLATES.get(hints.prompt_template, _SYSTEM_TEMPLATES["hybrid"])
    return template.format(min_dur=hints.min_duration_sec, max_dur=hints.max_duration_sec)


# ---------------------------------------------------------------------------
# Detection prompt (Phase E) — classify content type from sparse keyframes
# ---------------------------------------------------------------------------


_DETECTION_SYSTEM: Final[str] = """\
You classify videos so a downstream highlight-extraction pipeline can pick
the right strategy. You receive a small set of keyframes sampled across the
video timeline and a short textual description of the audio profile
(computed from the waveform, not transcribed — there is no speech text).

Pick EXACTLY ONE content type from this list:
- boxing: combat sports — boxing, MMA, kickboxing, wrestling, martial arts
- sport: other sports — football, basketball, tennis, racing, golf
- gameplay: video game footage (in-engine view, HUD visible, etc.)
- talk: monologue, interview, presentation, talking-head video
- podcast: multi-person conversational format (2+ people on camera together)
- other: anything else — vlog, cooking, music performance, tutorial, mixed

Return STRICT JSON only. No prose before or after. No markdown fences.
Schema:
{
  "content_hint": "boxing|sport|gameplay|talk|podcast|other",
  "confidence": <number 0.0 to 1.0>,
  "reasoning": "<one short sentence>"
}

Guidance:
- ``confidence`` is your own self-assessment of how certain you are. Use
  high values (>=0.8) only when the visual + audio signals clearly agree.
- When in doubt between two categories, return ``other`` rather than
  guessing — the downstream HYBRID profile handles ambiguous content
  gracefully.
- ``reasoning`` is for the human reading the log, not for self-talk. Keep
  it to one sentence describing the main signal you used.
"""


_DETECTION_USER: Final[str] = """\
Video duration: {duration_sec:.1f}s. Total keyframes sampled: {n_frames}.

{audio_description}

Keyframes (chronological, sampled across the full timeline):
{keyframe_lines}

Classify the content and return the JSON object.
"""


def build_detection_system_prompt() -> str:
    """Return the system message used for the auto-detect pre-step."""
    return _DETECTION_SYSTEM


def build_detection_user_prompt(
    *,
    duration_sec: float,
    keyframe_timestamps: list[timedelta],
    audio_description: str,
) -> str:
    """Return the user message: audio summary + keyframe timeline.

    The actual image bytes are attached by the provider as image content
    parts; the user-prompt text only includes the timestamps so the model
    can map each image to a point in the video.
    """
    keyframe_lines = "\n".join(
        f"  - frame {i + 1}: t = {format_timestamp(ts)}" for i, ts in enumerate(keyframe_timestamps)
    )
    return _DETECTION_USER.format(
        duration_sec=duration_sec,
        n_frames=len(keyframe_timestamps),
        audio_description=audio_description.rstrip(),
        keyframe_lines=keyframe_lines,
    )


def build_user_prompt(
    video_id: str,
    duration_sec: float,
    hints: AnalysisHints,
    specs: list[FrameSpec],
) -> str:
    """Return the user message: hints + keyframe timeline + target schema.

    Note: the actual image bytes are attached by the provider as separate
    image parts in the multimodal payload; here we only include the
    *timestamps* so the model can map each image to a point in the video.
    """
    keyframe_lines = "\n".join(
        f"  - frame {i + 1}: t = {format_timestamp(spec.timestamp)}" for i, spec in enumerate(specs)
    )
    return _USER_TEMPLATE.format(
        motion_block=_build_motion_block(hints.motion_windows_sec),
        content_hint=hints.content_hint.value,
        language=hints.language,
        goal=hints.goal,
        duration_sec=duration_sec,
        keyframe_lines=keyframe_lines,
        schema=clip_plan_schema(),
        video_id=video_id,
    )


def build_video_user_prompt(
    *,
    video_id: str,
    duration_sec: float,
    hints: AnalysisHints,
) -> str:
    """User message for the video-input path (no keyframe list, no motion block).

    The model watches the actual clip, so it perceives motion itself and the
    returned timestamps are relative to the clip — the batch engine offsets
    them to absolute source time.
    """
    return _VIDEO_USER_TEMPLATE.format(
        content_hint=hints.content_hint.value,
        language=hints.language,
        goal=hints.goal,
        duration_sec=duration_sec,
        schema=clip_plan_schema(),
        video_id=video_id,
    )


def _build_motion_block(windows: list[tuple[float, float]]) -> str:
    """Render the optional high-intensity-windows guidance block.

    Returns an empty string when no motion windows are known (every sampler
    other than ``motion``), so the surrounding template collapses cleanly.
    """
    if not windows:
        return ""
    rendered = "; ".join(
        f"{format_timestamp(timedelta(seconds=start))}-{format_timestamp(timedelta(seconds=end))}"
        for start, end in windows
    )
    return (
        "\nMotion analysis (optical flow + audio onsets) flags these high-intensity "
        "windows — the decisive action is most likely inside them, and keyframes are "
        "sampled densely there. Examine these windows closely and compare consecutive "
        "keyframes WITHIN a window to confirm an action before scoring it:\n"
        f"  {rendered}\n"
    )
