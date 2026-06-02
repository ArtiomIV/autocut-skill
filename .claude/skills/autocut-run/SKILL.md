---
name: autocut-run
description: >-
  Extract highlight clips from a video with AutoCut. Use when the user wants to
  auto-cut the best/viral moments of a video, or to find and cut a SPECIFIC
  described moment (a query). Needs a VLM provider (host or openrouter). For
  deterministic trimming use autocut-cut; for concatenation use autocut-merge.
---

# autocut run

`autocut run` analyses a video with a VLM and writes a **`plan.json`** (the ranked
clips with their timestamps — pre/post-roll already baked in per the mode). It does
**NOT** cut any MP4. YOU, the orchestrating agent, then review/edit `plan.json` and
produce the clips deterministically with **`autocut cut --from-json plan.json
--video VIDEO --output-dir DIR`** (optionally `--min-score N`). `run` itself does no
content classification — you pick the editing MODE / query up front.

Workflow: `run` → review/edit `plan.json` → `cut --from-json` → (optional) `merge
--from-manifest`. Splitting analysis from cutting lets you adjust a boundary or drop
a clip before committing to MP4s.

```
autocut run VIDEO [--content-hint MODE] [--query "<moment>"] \
    [--vlm host|openrouter] [--vlm-model ID] [--output-dir DIR] \
    [--accurate|--fast] [--single-pass] [--dry-run] [-y]
```

For long (>60s) openrouter video/audio runs, AutoCut uses a **two-pass**
coarse→fine analysis by default: it locates candidate regions across the whole
timeline, then re-analyses each one in isolation for tight, accurate cut
boundaries (it starts a knockdown clip on the punch, not the referee count). It
costs ~2x the model calls; pass `--single-pass` to disable. No effect on short
videos or the host/keyframe routes.

## Agent decision matrix — pick MODE × intent

Read the user's request crossed with the kind of video, then choose:

| User intent | Video kind | Use |
|---|---|---|
| "give me the best/viral moments", "make highlights" | action, sport, reactions, anything bursty | `--content-hint highlights` |
| "find/cut WHEN <specific thing happens>" | any | `--query "<the moment, elaborated>"` |
| "clip the interesting bits" of a talk/interview/podcast | speech-driven, talking head | `--content-hint talk` |
| unclear / mixed / you're not sure | mixed, vlog, unknown | `--content-hint hybrid` (or omit — defaults to hybrid) |

Rules of thumb:

- **highlights is strict**: it keeps only clips scoring ≥ 7 and may legitimately
  return **zero clips** if nothing clears the bar. That is a valid outcome — do
  not retry with a lower bar to force output.
- **`--query` is NOT highlights.** A request for one described moment ("the
  knockdown in round 3", "when they mention the price") is a query. Elaborate the
  user's words into a clear, specific description and pass it. A query disables
  the motion pre-filter (so low-motion targets like "the ring girl entering" are
  not sampled away) — the model filters via your query text instead.
- You can combine a mode and a query (e.g. `--content-hint talk --query "the
  moment she admits the mistake"`); the query drives selection, the mode tunes
  clip length.
- If you genuinely cannot tell the kind of video and have a cloud key, you may
  call `autocut detect` (openrouter, synchronous) for a machine classification
  first — but usually you can just decide and pass the mode.

## Provider

- `--vlm openrouter` (default if configured): sends the compressed video (or
  audio for talk) to the model; fully automatic, costs money (cost cap applies).
  Writes `plan.json` when done — then cut it with `autocut cut --from-json`.
- `--vlm host`: AutoCut writes `VLM_REQUEST.md` and pauses; YOU read it, write
  `VLM_RESPONSE.json`, then run `autocut resume --work-dir DIR` (which writes
  `plan.json`). Then cut with `autocut cut --from-json`.

## Examples

```bash
# 1) Analyse (writes ./CLIPS/plan.json, no MP4s)
autocut run match.mp4 --content-hint highlights

# 2) Review/edit ./CLIPS/plan.json if needed, then cut every clip
autocut cut --from-json ./CLIPS/plan.json --video match.mp4 --output-dir ./CLIPS

# 3) (optional) compose a reel from the cut clips
autocut merge --from-manifest ./CLIPS/manifest.json --min-score 8 -o reel.mp4

# Find one specific moment, then cut only the strong matches
autocut run interview.mp4 --query "when the guest reveals why he quit"
autocut cut --from-json ./CLIPS/plan.json --video interview.mp4 --output-dir ./CLIPS --min-score 7

# Analysis only, inspect the plan first (no plan written on a dry run)
autocut run mystery.mp4 --content-hint hybrid --dry-run
```
