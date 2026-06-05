---
name: autocut-run
description: >-
  Extract highlight clips from a video with AutoCut. Use this whenever the user
  wants the best / viral / funniest moments, highlights, the top moments worth
  sharing, or the best few seconds of a match, stream, podcast or interview —
  even if they don't say the word "highlight". Also use it to find and cut a
  SPECIFIC described moment (a query), e.g. "the moment he admits he lied". Two
  ways to run: analyse the frames YOURSELF locally for free (host), or run the
  autonomous cloud pipeline (openrouter, paid). For plain trimming of a known
  segment use autocut-cut; to join existing clips into a reel use autocut-merge.
---

# autocut run — find and cut the highlights

AutoCut finds the best moments of a video and cuts them. There are **two ways**,
and you offer the user the choice up front:

- **HOST (you analyse — free):** you, the agent, look at the frames yourself using
  the deterministic `autocut` subcommands. Zero API cost, you can be steered. Best
  for short/medium videos and for interviews. **No API key needed.**
- **CLOUD (autonomous — paid):** `autocut run --vlm openrouter` sends the video to
  a cloud model and writes a `plan.json` on its own. Best for long videos or
  fire-and-forget. **Needs an OpenRouter API key.**

## Step 0 — make sure AutoCut is installed

If `autocut --version` fails (command not found), install it once:

```bash
uv tool install git+https://github.com/ArtiomIV/autocut-skill
```

(If `uv` itself is missing: `curl -LsSf https://astral.sh/uv/install.sh | sh` on
mac/Linux, or `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"` on
Windows. ffmpeg is bundled — no separate install.)

## Step 1 — probe, then pick HOST or CLOUD

```bash
autocut probe VIDEO    # -> JSON: duration_sec, fps, width, height, codecs
```

Use the duration to suggest a default, then **ask the user** (unless they already
said which):

| Situation | Suggested default |
|---|---|
| Video ≤ ~15 min | **HOST** (free; you read the frames) |
| Video > ~15 min | **CLOUD** (the frames would flood your context) |
| No OpenRouter key configured | **HOST** (cloud needs a key) |
| You cannot open / view images | **CLOUD** |

The user can always override. Then follow the matching recipe below.

---

## HOST recipe (you analyse the frames — free)

You drive the deterministic tools. A contact **sheet** is a grid image: each cell
is one frame with its integer index burned in; `index.json` maps every index to
its exact second. You read the grid, pick moments, cut them.

**Legibility (adapt to resolution):** the cells must be readable. For low-resolution
sources (from `probe`, width ≤ ~640 / height ≤ ~480) the default 6×8 grid packs the
cells too small — pass **bigger cells and fewer per sheet**, e.g. `--cell-px 320
--cols 4 --rows 5`. For HD sources the 256px / 6×8 default is fine. The fine pass
should always be legible (`--cell-px 320` is a safe default).

### Short / medium video (≤ ~15 min) — single dense pass

```bash
autocut sheet VIDEO --fps 2 --out CLIPS/sheet      # dense grid + index.json
```

1. **Open** every `CLIPS/sheet/sheet_*.jpg` and read `CLIPS/sheet/index.json`.
2. **Identify** the highlight moments (see *Judgement* below). For each, read the
   cell indices at its start and end and look them up in `index.json` to get exact
   seconds. An impact (punch, fall, goal) often happens a cell or two BEFORE the
   frame where you first see its effect — start a touch earlier.
3. **Cut** each chosen `[start, end]`:
   ```bash
   autocut cut VIDEO --start <START> --end <END> --accurate -o CLIPS/s<score>_clip_NN.mp4
   ```
4. **⭐ VERIFY THE CUT (MANDATORY — do not skip):** sheet the clip you just made and
   confirm the decisive event is actually inside it (see *Verify every cut* below).
   If it's missing or truncated, RE-CUT before moving on.
5. **(Optional) reel:** `autocut merge CLIPS/s*_clip_*.mp4 -o CLIPS/reel.mp4`.

### Long video (> ~15 min) — chunk + coarse→fine + cut as you go

A long video at 2 fps is too many frames for one look. Process it **chunk by
chunk and cut incrementally**, so your context only holds one chunk at a time:

For each ~5-minute window `[T, T+300]`:
1. **Coarse** (sparse, cheap): `autocut sheet VIDEO --interval 3 --from T --to T+300 --out CLIPS/coarse_T` → open it, spot the 0–few candidate regions in this chunk.
2. **Fine** (dense) on each candidate region `[a, b]`: `autocut sheet VIDEO --fps 2 --from a --to b --out CLIPS/fine_a` → open it, pick exact `[start, end]`.
3. **Cut immediately**: `autocut cut VIDEO --start <START> --end <END> --accurate -o CLIPS/s<score>_clip_NN.mp4`.
4. **⭐ VERIFY THE CUT (MANDATORY — do not skip):** sheet the clip and confirm the
   decisive event is inside it (see *Verify every cut* below); RE-CUT if missing.
   Only then may you forget this chunk's frames and move to the next window.

Finally: `autocut merge CLIPS/s*_clip_*.mp4 -o CLIPS/reel.mp4`.

---

## CLOUD recipe (autonomous — paid)

### Key (security)

The cloud path needs an OpenRouter API key, stored in the OS keyring (never in a
file, never in plaintext, never echoed). If `autocut keys list` doesn't show
`openrouter`, ask the **user** to store their own key:

```bash
autocut keys set openrouter      # prompts for the key, saves it to the keyring
```

Do not ask the user to paste the key into the chat; the command prompts for it
securely. Then:

### Run → cut

```bash
autocut run VIDEO --vlm openrouter [--content-hint MODE] [--query "<moment>"]
# writes CLIPS/plan.json (ranked clips, roll baked in; real + estimated cost recorded)
autocut cut --from-json CLIPS/plan.json --video VIDEO --output-dir CLIPS [--min-score N]
autocut merge --from-manifest CLIPS/manifest.json --min-score 8 -o CLIPS/reel.mp4   # optional
```

For long (>60s) cloud runs AutoCut does a **two-pass** coarse→fine automatically
for tight boundaries (~2× the model calls; add `--single-pass` to disable). Review
`plan.json` before cutting if you want to drop or nudge a clip.

---

## Judgement — what to keep (both paths)

Pick the MODE/intent from the request crossed with the kind of video:

| User intent | Use |
|---|---|
| "best/viral moments", "make highlights" | **highlights** mode |
| "find/cut WHEN <specific thing happens>" | **query**: a clear description of that one moment |
| "interesting bits" of a talk/interview/podcast | **talk** mode |
| unclear / mixed | **hybrid** (the safe default) |

(On cloud these map to `--content-hint highlights|talk|hybrid` / `--query`; on host
they just guide YOUR selection.)

### On HOST: load the SAME rules the cloud uses (MANDATORY first step of judging)

**Before you look at a single sheet to pick clips, you MUST run and FOLLOW:**

```bash
autocut guidance highlights     # or: talk | hybrid
```

This is **not optional**. It prints the exact editorial rules the cloud model gets
in its prompt — there is ONE source, so host and cloud judge clips identically.
Skipping it is the documented cause of past misreads. Run it, read it, apply it.
The rules that matter most (and the ones agents most often get wrong):

- **ANCHOR ON THE EVENT, NOT ITS AFTERMATH.** A referee's count, a fighter on the
  canvas, a celebration is PROOF an event happened just before — find that event
  and start there. **A count (even a standing-eight without a knockdown) means a
  strong moment occurred: KEEP it, never skip it because the outcome looks
  ambiguous.** This is the #1 mistake — do not dismiss a near-KO.
- **A KNOCKDOWN/KO is ALWAYS kept, score 9-10, never omitted.** Score on INTENSITY:
  a near-KO / staggering blow / heated exchange counts even without a clean outcome.
- **Keep slow-motion REPLAYS** (they're prime viral footage), score them high.
- **OMIT** dead time, circling, clinching, no-clean-strike "exchanges", and all
  pre-action/ceremony (entrances, anthems, podiums) and breaks (between-round rest).
- **highlights is strict**: zero clips is a valid outcome — never force best-of-nothing.
- **query ≠ highlights**: one described moment; elaborate it and find exactly that.
- Cut tight: wind-up + follow-through of the moment, nothing more.

### Verify every cut (MANDATORY self-check)

A boundary read off a sheet can be wrong — most often the clip ends BEFORE the
decisive moment (you anchored on an earlier flurry, not the actual knockdown +
count). **After every `cut`, you MUST check the clip you produced, not just trust
the timestamps:**

```bash
autocut sheet CLIPS/s<score>_clip_NN.mp4 --fps 2 --cell-px 320 --out CLIPS/verify_NN
```

Open `CLIPS/verify_NN/sheet_*.jpg` and confirm the **whole decisive event is inside
the clip**: the strike AND its result AND the aftermath that proves it (e.g. the
fighter going down AND the referee's count; the goal AND the net bulging). If the
clip starts late, ends before the count, or cuts the impact — **RE-CUT with corrected
boundaries and verify again.** Do not merge or report a clip until it passes this
check. This loop is what catches the #1 error (cutting the aftermath off the event).
