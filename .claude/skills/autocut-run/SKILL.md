---
name: autocut-run
description: >-
  Extract highlight clips from a video with AutoCut. Use when the user wants to
  auto-cut the best/viral moments of a video, or to find and cut a SPECIFIC
  described moment (a query). Needs a VLM provider (host or openrouter). For
  deterministic trimming use autocut-cut; for concatenation use autocut-merge.
---

# autocut run

`autocut run` analyses a video with a VLM and produces highlight clips in
`./CLIPS/` (one MP4 per kept clip + a `manifest.json`). YOU, the orchestrating
agent, pick the editing MODE and (optionally) a query up front — `run` itself
does no content classification.

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
- `--vlm host`: AutoCut writes `VLM_REQUEST.md` and pauses; YOU read it, write
  `VLM_RESPONSE.json`, then run `autocut resume --work-dir DIR`. Add
  `--host-video` only if you can actually watch a video file.

## Examples

```bash
# Auto-highlights of a sparring video (may return nothing if it's all warm-up)
autocut run match.mp4 --content-hint highlights

# Find and cut one specific moment
autocut run interview.mp4 --query "when the guest reveals why he quit"

# Conservative pass on an unknown clip, no files written (inspect first)
autocut run mystery.mp4 --content-hint hybrid --dry-run
```
