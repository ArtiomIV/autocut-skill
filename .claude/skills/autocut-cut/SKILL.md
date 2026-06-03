---
name: autocut-cut
description: >-
  Trim / render / export video clips with ffmpeg — deterministic, NO VLM, no
  analysis. Use this to produce the actual MP4 files from a plan.json that
  autocut-run already wrote (--from-json), including requests like "only render
  the clips scoring 7 or higher", or to cut one exact [start, end] segment from a
  video. This is the step that writes the MP4s once a plan.json exists or the
  timestamps are already known. To FIND the moments first use autocut-run; to
  join the cut clips into one reel use autocut-merge.
---

# autocut cut

`autocut cut` is the deterministic trimming step: ffmpeg only, no model, no cost.
It has two modes.

## Mode 1 — cut a whole plan.json (the normal follow-up to `autocut run`)

```
autocut cut --from-json PLAN.json --output-dir DIR [--video VIDEO] \
    [--min-score N] [--accurate|--fast]
```

Cuts **every** clip listed in the plan into `DIR/separate/*.mp4` and writes a
`manifest.json` (so `autocut merge --from-manifest` can compose a reel next). The
plan's timestamps already include any pre/post-roll, so the cut is **1:1** — it
pads nothing.

- `--video` is optional: the video path is taken from the plan; pass it (or the
  positional `VIDEO`) only to override (e.g. the plan moved machines).
- `--min-score N` cuts only clips with `final_score >= N` (0 = all). Handles
  0..N clips — zero matches just prints "nothing to cut".

## Mode 2 — cut one exact segment

```
autocut cut VIDEO --start 00:01:12.500 --end 00:01:27.000 --output clip.mp4
```

`--start`/`--end` accept `HH:MM:SS.mmm` or plain seconds; `--end` must be > start.

## Accuracy

- `--fast` (default): **stream-copy** — fast and lossless, but snaps to the
  nearest keyframe (typically ±1–2s). Good enough for most clips.
- `--accurate`: **re-encode** (libx264) for frame-accurate boundaries. Slower,
  re-compresses. Use when a tight, exact in/out point matters.

## Examples

```bash
# After `autocut run` wrote ./CLIPS/plan.json:
autocut cut --from-json ./CLIPS/plan.json --output-dir ./CLIPS

# Only the strong clips, frame-accurate:
autocut cut --from-json ./CLIPS/plan.json --output-dir ./CLIPS --min-score 7 --accurate

# One manual segment:
autocut cut match.mp4 -s 00:08:01 -e 00:08:14 -o ko.mp4
```
