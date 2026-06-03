---
name: autocut-merge
description: >-
  Concatenate / join already-cut MP4 clips into ONE video — a highlights reel —
  with ffmpeg, deterministic, NO VLM. Use this to build or assemble a reel from
  an autocut manifest.json (optionally keeping only clips scoring N+) or from
  explicit MP4 file paths, once the individual clips already exist. It does NOT
  analyse the video or create clips: to find the moments use autocut-run, to trim
  the individual clips use autocut-cut.
---

# autocut merge

`autocut merge` joins MP4s into a single file via the ffmpeg concat demuxer.
Deterministic, no model, no cost. It is the last step of the chain:
`run` → `cut --from-json` → **`merge`**. Two modes.

## Mode 1 — manifest-aware (the normal follow-up to `autocut cut`)

```
autocut merge --from-manifest CLIPS/manifest.json --min-score N \
    -o reel.mp4 [--order chronological|score-desc|manifest]
```

Reads the `manifest.json` that `autocut cut` wrote, keeps clips with
`final_score >= min-score`, and concatenates their separate-output MP4s into one
reel. Errors (does not write an empty file) if nothing clears the bar.

## Mode 2 — explicit file list

```
autocut merge a.mp4 b.mp4 c.mp4 -o reel.mp4
```

Dumb concat in the order given. The inputs **must share codec / resolution /
fps** (they do when they all came from the same `autocut cut` run).

## Order (`--order`)

- `chronological` (default): by each clip's start time in the source.
- `score-desc`: highest `final_score` first (a "best moments first" reel).
- `manifest`: keep the manifest's existing order.

`--from-manifest` and positional inputs are **mutually exclusive** — pass one.

## Examples

```bash
# Reel of the strong clips, best first:
autocut merge --from-manifest ./CLIPS/manifest.json --min-score 7 --order score-desc -o reel.mp4

# Join three specific clips in order:
autocut merge intro.mp4 ko.mp4 outro.mp4 -o final.mp4
```
