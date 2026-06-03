---
name: autocut-detect
description: >-
  Classify the CONTENT TYPE / editing mode of a video with AutoCut — is it
  sport/action highlights, talk/interview, or hybrid? Use this when the user
  wants to identify or classify what kind of video it is (for example before
  deciding how to edit or cut it), WITHOUT running the full extraction. Prints a
  small JSON {content_hint, confidence, reasoning}. Needs a cloud VLM
  (openrouter). To actually extract the clips afterwards use autocut-run.
---

# autocut detect

`autocut detect` runs ONE cheap VLM call to classify the kind of video, then
prints a JSON object and exits. It does **not** sample many keyframes, analyse
the whole timeline, or cut anything — it is the lightweight "what is this video?"
pre-step.

```
autocut detect VIDEO [--vlm openrouter] [--vlm-model ID] \
    [--output-dir DIR] [--keyframes N]
```

Output (stdout, JSON):

```json
{"content_hint": "talk", "confidence": 0.92, "reasoning": "two people seated, ..."}
```

## When to use it

- You are about to call `autocut run` and want to **confirm the mode** the agent
  picked, or surface it to the user, before spending on the full extraction.
- You genuinely cannot tell the kind of video from the request alone.

Usually you do **not** need this: as the orchestrating agent you can just decide
the MODE yourself and pass it to `autocut run --content-hint <mode>`. `detect` is
the explicit machine-classification path for when you want a second opinion.

## Constraints

- **Needs a synchronous cloud provider** (`--vlm openrouter`). It does NOT work
  with `--vlm host`: the host agent IS the orchestrator and should classify the
  video directly rather than go through a paused detection cycle. If you call it
  on host it errors with that advice.
- `--keyframes N` (3..20, default 9) controls how many stratified-random stills
  the classifier sees. The JPEGs land in `--output-dir` (default `./CLIPS`) and
  are reused if you launch `autocut run` next.
- `content_hint` is one of `highlights` / `talk` / `hybrid` (the editing modes
  `autocut run` understands). Below the confidence threshold the pipeline would
  fall back to `hybrid`.

## Examples

```bash
# Classify, then decide how to run
autocut detect interview.mp4 --vlm openrouter
# -> {"content_hint":"talk","confidence":0.9,...}
autocut run interview.mp4 --content-hint talk
```
