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

# v11 (2026-06-01): two fixes from E2E v10 frame review. (a) CORE: tell the model its
# view is low-frame-rate so a fast impact lands 2-5s BEFORE the frame where its effect
# shows -> set start 3-5s earlier (the real KO punch at 5:30 was missed, clip opened on
# the 5:33 standing-8 count). Prompt-level mitigation; the deterministic DSP impact-snap
# (refine.py) stays dormant. (b) HIGHLIGHTS: exclude breaks IN the action too (round-end
# bell + between-round rest, timeouts, restarts) — a between-round pause had scored 7.
# v10 (2026-06-01): highlights guidance now explicitly EXCLUDES pre-action/ceremony
# (entrances, walk-ins, intros, anthems, warm-ups, podium) — the model was scoring
# ring walkouts a 7, same as a near-KO, and ranking cannot separate them (same
# score), so the lever is the prompt. Generic (any sport/event), highlights-only.
# v9 (2026-06-01): soften the highlights guidance so a STRONG-IMPACT moment counts
# even without a clean resolution — a near-KO / staggering blow / heated swing is a
# highlight scored on its INTENSITY, not on whether it ended in a knockdown or goal.
# Stays selective (the "confirm before scoring high" rule is unchanged). Fixes v8
# being correctly strict but on the wrong axis (it only rewarded definitive outcomes).
# v8 (2026-06-01): harden the event-anchoring rule — recognising an aftermath
# MEANS an event preceded it, so ``start`` MUST be set at the inferred event
# start, never on the aftermath; if the start can't be located, fall back to AT
# LEAST 5s before the aftermath. Also resolves the v7 conflict with the
# highlights "cut tight ~1-2s before the beat" rule (the beat is the EVENT, not
# its aftermath). Still a model estimate, not frame-accurate (DSP snap is that).
# v7 (2026-06-01): GENERIC "capture the event, not only its consequences" rule in
# the shared core (and audio). Superseded by v8's stronger phrasing + 5s fallback.
# v6 (2026-06-01): highlights guidance rejected AFTERMATH-only clips (superseded).
# v5 (2026-06-01): reinforce that an empty result is correct even on a clip the
# coarse pass pre-selected (the fine pass must reject a worthless candidate
# rather than force a "best of nothing"). Two-pass candidate padding also widened
# to ±30s so the fine pass sees the whole moment (e.g. a full slow-mo replay).
# v4 (2026-06-01): add the two-pass prompts — a COARSE/locate system prompt
# (high recall, wide candidate regions, video + audio) used by pass 1, and a
# QUERY system prompt (hunt a specific described moment, no auto-best) used
# whenever ``hints.query`` is set. The fine highlights/talk/hybrid prompts are
# unchanged from v3.
# v3 (2026-05-29): cap the sport lead-in at ~1-2s (no long approach/circling/
# idle pause before the action) so clips stop drifting long and loose. Pairs
# with temperature=0 on the analysis calls for run-to-run consistency.
# v2 (2026-05-28): reworked sport/talk clip-boundary guidance so the model
# includes the wind-up and follow-through of the key moment instead of cutting
# tight on the impact. The detection prompt is unchanged, hence its own version.
PROMPT_VERSION: Final[str] = "v11"
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
- ANCHOR THE CLIP ON THE EVENT, NEVER ON ITS CONSEQUENCES. A decisive event (a
  landing punch, a goal, a fall, a punchline) is often shorter than the gap
  between the frames you are given, so you may never see the event frame itself —
  you see only its AFTERMATH: a referee's count, a knockdown, a celebration,
  applause or laughter, players resetting, a result on screen. That aftermath is
  PROOF the event happened in the seconds immediately before it. The instant you
  recognise an aftermath, you have ALSO concluded an event preceded it — so you
  MUST set ``start`` at the point where, in your judgement, that EVENT BEGAN, and
  NEVER on the aftermath. Read the build-up (the wind-up, the approach, the rising
  action) to locate that point. If you cannot tell exactly where the event began,
  you MUST still set ``start`` to AT LEAST 5 seconds before the first frame of the
  aftermath — do not place it on the aftermath under any circumstance. The clip
  must always contain the event itself; a clip that opens on the count, the
  celebration, or the reaction with the action already finished is WRONG.
- YOUR VIEW OF THE FOOTAGE IS SAMPLED AT A LOW FRAME RATE, so a fast impact (a
  punch, a kick, a collision, a fall) frequently happens 2-5 seconds BEFORE the
  first frame in which you notice its effect — you rarely see the impact frame
  itself. Do NOT trust your sense of the exact impact instant. For any
  impact-type event, set ``start`` 3-5 seconds EARLIER than the moment you
  believe the impact occurred, so the real strike is certainly inside the clip.
  Erring early is correct here: a clip that begins just after the impact has
  missed the very thing it is about.
- If no segment is worth keeping, return an empty ``clips`` array. Being shown
  this footage does NOT oblige you to return a clip — this segment may have been
  pre-selected by a coarse first pass and contain nothing of value. When there is
  genuinely no worthwhile moment here (only ambient dead time, no event and no
  consequences of one), an empty result is the CORRECT answer.
"""


_HIGHLIGHTS_GUIDANCE: Final[str] = """\

Content guidance — highlights (the best, viral-worthy moments; any content):
- Select ONLY moments that would make someone scrolling their feed STOP and
  watch, and that stand on their own out of context: a decisive action, a peak
  of intensity, a spectacular or surprising beat, a strong emotional reaction.
  If the video contains no such moment, return an EMPTY ``clips`` array — do NOT
  emit a "best of nothing" just to have an output.
- A moment does NOT need a clean resolution to be a highlight. A decisive OUTCOME
  (a knockdown, a scored goal, a finished play) is only ONE kind of peak — a
  near-miss carries the same charge: a blow that visibly staggers or hurts
  someone without dropping them, a dangerous attack barely survived, a shot that
  rattles the frame, a heated confrontation, a sudden swing of momentum. KEEP
  these too, and score them on the INTENSITY of the moment, NOT on whether it
  ended in a definitive result. The signal that the intensity spiked is visible:
  someone is hurt, rocked or off-balance, the participants or crowd react, the
  action surges, an official intervenes. Stay selective even so — this is NOT
  every exchange, throw or routine action, only the moments where the intensity
  clearly PEAKS.
- Pre-action and ceremony are NOT highlights, no matter how well produced.
  Entrances and walk-ins, introductions and line-ups, anthems, warm-ups, the
  coin toss or staredown, interviews, podium and award moments, and idle waiting
  set the scene but do NOT stand on their own in a feed. Score them low and OMIT
  them even when the lighting, crowd, or production looks dramatic — dramatic
  staging is not the same as a dramatic action.
- Breaks IN the action are not highlights either: the round-ending bell and the
  rest between rounds, timeouts, stoppages, substitutions, restarts, and the
  lull while an official separates or resets the competitors. Do NOT anchor a
  clip on the round-ending buzzer, and do NOT extend a clip into the break that
  follows — end it on the last live action.
- Confirm before scoring high. A single ambiguous frame (a blur that might be a
  strike, a face that might be reacting) is often nothing — a feint, a study
  phase, a transition. Score >= 7 ONLY when the moment is clearly readable
  across the frames, not on one lucky still.
- Cut tight WHEN YOU CAN SEE THE EVENT. Start ~1-2 seconds before the decisive
  beat — and the decisive beat is the EVENT ITSELF (the landing punch, the goal),
  NOT its aftermath; when you can see only the aftermath, the ANCHOR-ON-THE-EVENT
  rule above governs the start instead. Just enough to make the wind-up legible,
  NEVER the long approach, circling, or idle pause before it begins (those dead
  seconds make the clip drag). ALWAYS include the immediate follow-through and
  reaction (the aftermath, the response, the celebration), and NEVER end on the
  frame of the peak itself — the aftermath is what makes the clip land.
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
    "highlights": _CORE_RULES + _HIGHLIGHTS_GUIDANCE,
    "talk": _CORE_RULES + _TALK_GUIDANCE,
    "hybrid": _CORE_RULES + _HYBRID_GUIDANCE,
}


# ---------------------------------------------------------------------------
# Two-pass prompts (v4): COARSE/locate (pass 1) and QUERY (specific moment)
# ---------------------------------------------------------------------------


# Pass 1 of the two-pass pipeline. The model is imprecise on long footage
# (it ingests video at ~1 fps internally) but precise on a short clip — so the
# first pass optimises for RECALL with wide, generous candidate regions, and a
# second pass re-cuts each one precisely. Deliberately NOT conservative: a miss
# here is permanent, an over-inclusion is trimmed downstream. Works whether the
# model watches video or listens to audio (the engine is payload-agnostic).
_COARSE_SYSTEM: Final[str] = """\
You are a video editor assistant doing a FIRST, COARSE pass over a piece of
footage. A SECOND, precise pass will refine the exact cut afterwards, so your
ONLY job now is RECALL: flag every region that MIGHT contain a moment worth
keeping, with GENEROUS, WIDE boundaries.

Return STRICT JSON matching the provided schema. No prose before or after the
JSON. No explanations. No markdown fences.

Hard rules:
- Each candidate region must be between {min_dur:.1f} and {max_dur:.1f} seconds.
- Prefer WIDER regions: when unsure where a moment starts or ends, include MORE
  context on both sides — the next pass trims it.
- It is BETTER to include a borderline region than to miss it. Do NOT be
  conservative here: over-inclusion is corrected downstream, a miss is permanent.
- Regions must NOT overlap; if two moments are adjacent, return ONE region that
  covers both.
- ``start`` and ``end`` use the ``HH:MM:SS.mmm`` format, RELATIVE to this clip
  (its first moment is 00:00:00.000).
- ``score`` (0-10) is your ROUGH confidence the region contains something worth
  keeping — it will be re-judged precisely later, so do not agonise over it.
- ``category`` is one of: highlight, reaction, dialogue, transition, filler.
- ``tags`` is a list of 1-5 short strings.
- If the footage genuinely contains nothing of interest, return an empty
  ``clips`` array.
"""


# Appended to the coarse prompt when a query is active: same recall-first
# behaviour, but the regions must be relevant to the requested moment.
_COARSE_QUERY_BLOCK: Final[str] = """\

TARGET — you are looking specifically for this:
"{query}"
Flag EVERY region that might match it, generously. When unsure whether a region
matches, INCLUDE it — the precise pass will confirm or reject it.
"""


# Fine pass when the user asked for a SPECIFIC described moment (``hints.query``
# is set). This is NOT auto-highlights: the model must find the requested moment
# and nothing else, and return empty if it does not occur.
_QUERY_GUIDANCE: Final[str] = """\

USER REQUEST — find this specific moment:
"{query}"

- Your task is NOT to pick the "best" or most viral moments. Find ONLY the
  segment(s) that match the USER REQUEST above, and cut them tightly.
- Include a segment ONLY if it clearly matches the request. If the requested
  moment does not occur in this footage, return an EMPTY ``clips`` array — do
  NOT substitute a different "good" moment to have an output.
- ``score`` reflects how well the segment matches the request (10 = this is
  unmistakably the requested moment; lower if it is a partial or possible match).
- Give a short lead-in for context and include the immediate follow-through, but
  keep the cut tight on the requested moment.
"""


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
- If no segment is worth keeping, return an empty ``clips`` array. Being given
  this audio does NOT oblige you to return a clip — it may have been pre-selected
  by a coarse first pass and contain nothing of value. When there is genuinely no
  worthwhile spoken moment here, an empty result is the CORRECT answer.

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
- Anchor the clip on the EVENT, never on its consequences. A reaction you hear —
  laughter, applause, a gasp, someone answering "wow" — is PROOF that the line or
  moment that caused it came just before. You MUST set ``start`` at the line that
  triggered it, never on the reaction. If you cannot tell exactly where that line
  began, set ``start`` to AT LEAST 5 seconds before the reaction. The clip must
  contain the cause AND its effect — NEVER only the reaction without the words
  that earned it.
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


# Audio pass 1 (coarse/locate): listen for RECALL — flag every spoken region
# that might be worth keeping, generously, for the precise pass to trim.
_AUDIO_COARSE_SYSTEM: Final[str] = """\
You are a video editor assistant doing a FIRST, COARSE pass over the audio track
of a clip — speech, tone, laughter, applause, pauses. You CANNOT see the picture.
A SECOND, precise pass will refine the exact cut afterwards, so your ONLY job now
is RECALL: flag every spoken region that MIGHT be worth keeping, with GENEROUS,
WIDE boundaries.

Return STRICT JSON matching the provided schema. No prose, no markdown fences.

Hard rules:
- Each candidate region must be between {min_dur:.1f} and {max_dur:.1f} seconds.
- Prefer WIDER regions: never cut mid-sentence; include the whole turn plus a
  little context on each side — the next pass trims it.
- It is BETTER to include a borderline region than to miss it. Over-inclusion is
  corrected downstream, a miss is permanent.
- Regions must NOT overlap; merge adjacent moments into ONE region.
- ``start``/``end`` use ``HH:MM:SS.mmm``, RELATIVE to this clip (first moment is
  00:00:00.000).
- ``score`` (0-10) is your ROUGH confidence — it will be re-judged precisely later.
- ``category`` is one of: highlight, reaction, dialogue, transition, filler.
  ``tags`` is 1-5 short strings.
- If nothing is worth keeping, return an empty ``clips`` array.
{query_block}"""


# Audio fine pass when a query is active: find the requested spoken moment only.
_AUDIO_QUERY_SYSTEM: Final[str] = """\
You are a video editor assistant LISTENING to the audio track of a clip — speech,
tone, laughter, applause, pauses. You CANNOT see the picture; judge purely on
what is said and how it is said.

USER REQUEST — find this specific moment:
"{query}"

Return STRICT JSON matching the provided schema. No prose, no markdown fences.

Hard rules:
- Your task is NOT to pick the "best" or most viral moments. Find ONLY the spoken
  segment(s) that match the USER REQUEST above.
- Include a segment ONLY if it clearly matches the request. If the requested
  moment is not spoken in this clip, return an EMPTY ``clips`` array — do NOT
  substitute a different "good" moment.
- Each clip must be between {min_dur:.1f} and {max_dur:.1f} seconds; never cut
  mid-sentence. Give a short lead-in and include the immediate response.
- Clips must NOT overlap. ``start``/``end`` use ``HH:MM:SS.mmm`` RELATIVE to this
  clip (first moment is 00:00:00.000).
- ``score`` reflects how well the segment matches the request (10 = unmistakably
  the requested moment).
- ``category`` is one of: highlight, reaction, dialogue, transition, filler.
  ``tags`` is 1-5 short strings.
- Put the quotable line or a tight summary in ``description``.
"""


_AUDIO_COARSE_QUERY_BLOCK: Final[str] = """\

TARGET — you are looking specifically for this:
"{query}"
Flag EVERY spoken region that might match it, generously; the precise pass will
confirm or reject each one.
"""


def build_audio_system_prompt(hints: AnalysisHints) -> str:
    """System message for the audio-input path.

    Mirrors the video path's precedence: coarse (pass 1) → query → fine. The
    coarse and query variants support ``hints.query``; the default is the
    selective "hunt viral spoken moments" prompt.
    """
    if hints.prompt_template == "coarse":
        query_block = _AUDIO_COARSE_QUERY_BLOCK if hints.query else ""
        return _AUDIO_COARSE_SYSTEM.format(
            min_dur=hints.min_duration_sec,
            max_dur=hints.max_duration_sec,
            query_block=query_block,
            query=hints.query or "",
        )
    if hints.query:
        return _AUDIO_QUERY_SYSTEM.format(
            min_dur=hints.min_duration_sec,
            max_dur=hints.max_duration_sec,
            query=hints.query,
        )
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


def _select_system_template(hints: AnalysisHints) -> str:
    """Pick the system template for the keyframe/video path.

    Precedence:
    1. ``prompt_template == "coarse"`` → the recall-first locate prompt (pass 1
       of the two-pass pipeline), with the query target appended if a query is
       active so coarse + query locate the right regions.
    2. otherwise a query (``hints.query``) → the QUERY guidance (hunt that
       specific moment, no auto-best).
    3. otherwise the fine per-mode guidance (highlights / talk / hybrid), with
       an unknown template id falling back to hybrid.
    """
    if hints.prompt_template == "coarse":
        return _COARSE_SYSTEM + (_COARSE_QUERY_BLOCK if hints.query else "")
    if hints.query:
        return _CORE_RULES + _QUERY_GUIDANCE
    return _SYSTEM_TEMPLATES.get(hints.prompt_template, _SYSTEM_TEMPLATES["hybrid"])


def build_system_prompt(hints: AnalysisHints) -> str:
    """Return the system message: core rules + per-template guidance.

    Template selection is driven by ``hints.prompt_template`` (set from the
    active ``ContentProfile`` in the pipeline) and ``hints.query``. See
    ``_select_system_template`` for the precedence.
    """
    return _select_system_template(hints).format(
        min_dur=hints.min_duration_sec,
        max_dur=hints.max_duration_sec,
        query=hints.query or "",
    )


# ---------------------------------------------------------------------------
# Detection prompt (Phase E) — classify content type from sparse keyframes
# ---------------------------------------------------------------------------


_DETECTION_SYSTEM: Final[str] = """\
You classify a video so a downstream highlight-extraction pipeline can pick the
right editing MODE. You receive a small set of keyframes sampled across the
video timeline and a short textual description of the audio profile (computed
from the waveform, not transcribed — there is no speech text).

Pick EXACTLY ONE editing MODE from this list:
- highlights: footage whose value is its best VISUAL moments — sport, combat,
  action, gameplay, stunts, reactions. The pipeline hunts the peak moments.
- talk: speech-driven — monologue, interview, presentation, podcast. The value
  is in WHAT IS SAID rather than the picture.
- hybrid: anything unclear, mixed, or that fits neither cleanly (vlog, cooking,
  music, tutorial). When in doubt, choose this.

Return STRICT JSON only. No prose before or after. No markdown fences.
Schema:
{
  "content_hint": "highlights|talk|hybrid",
  "confidence": <number 0.0 to 1.0>,
  "reasoning": "<one short sentence>"
}

Guidance:
- ``confidence`` is your own self-assessment of how certain you are. Use
  high values (>=0.8) only when the visual + audio signals clearly agree.
- When in doubt between two modes, return ``hybrid`` rather than guessing —
  the downstream HYBRID profile handles ambiguous content gracefully.
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
