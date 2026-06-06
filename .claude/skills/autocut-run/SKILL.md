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

## Step 0.5 — if the file may still be arriving, wait for it first

A file **exists the instant a copy starts** but keeps growing for seconds (a phone
transfer into a watched folder, an upload, an AirDrop/network copy). If you probe or
cut it too early you can read a truncated video — or silently cut a clip that's missing
its end. So **whenever the video was just dropped in / you're watching a folder / the
user is transferring it, run this BEFORE `probe` (and before `run`), on BOTH paths:**

```bash
autocut wait-ready VIDEO          # blocks until the file finished copying, then exits 0
# tune if needed: --stable-for 2 (secs unchanged = ready)  --timeout 900  --poll 1
```

It polls the file's size+mtime and returns only once they've held steady for
`--stable-for` seconds and the file is openable (this also clears a Windows copy
lock); it also waits for a not-yet-present file to appear. **Exit 0 = ready, proceed;
exit 1 = it timed out** (still copying / locked / never arrived) — tell the user, do
not run on it. If you already have a settled local file (nothing is copying), you can
skip this step.

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

**Legibility — DOWNSCALING IS WHAT MAKES YOU MISREAD.** You can only judge what you
can clearly SEE. A cell smaller than the source frame throws away the only detail you
have, and a small, motion-blurred cell makes a knockdown + a referee's count look
exactly like a clinch — this is a real, documented failure. So err toward BIGGER cells
and MORE sheets everywhere; never shrink cells just to save a sheet:
- **Coarse** (locating regions): keep cells generous so you don't miss a region —
  `--cell-px 480 --cols 4 --rows 4`. Bump higher if anything looks ambiguous.
- **Fine** pass and **every verify**: use **`--cell-px 640` with a SMALL grid
  (`--cols 3 --rows 4`)**. On an SD source (≤640 wide, e.g. 360p) that is the NATIVE
  frame — zero quality lost; on HD it is still plenty of detail. This produces MANY
  more sheets — that is fine and expected, **legibility beats sheet count.** Never
  downscale the decisive moment to save a sheet.

### Short / medium video (≤ ~15 min) — single dense pass

```bash
autocut sheet VIDEO --fps 2 --cell-px 640 --cols 3 --rows 4 --out CLIPS/sheet   # dense, legible grid + index.json
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

### Long video (> ~15 min) — LOCATE cheaply, then fine → cut

A long video at 2 fps is too many frames to read blindly. FIRST locate the few
regions worth a look, THEN fine-sheet only those — so your context holds a handful
of windows, not the whole timeline.

**Step 0 — locate (pick ONE path):**

- **Loud action sport (crowd/commentary) with NO specific query → `signals` (cheapest):**
  ```bash
  autocut signals VIDEO        # JSON: audio_peaks (impact/crowd, ranked by strength) + scene_cuts
  ```
  The JSON is tiny — read it instead of opening dozens of coarse sheets. Your candidate
  windows are: the **strong audio peaks** (each marks a loud live moment — a landed
  shot, a roar) **and the scene-cut clusters** (replays / graphic transitions). **Always
  also sheet the ~30–60s AFTER each strong peak** — a slow-mo replay lives there and is
  often audio-SILENT, so the scene-cut is what flags it. This REPLACES the blanket coarse
  pass. **Advisory, never a gate:** it says where to look FIRST, it does not forbid
  looking elsewhere.

- **A specific `--query` (e.g. "the moment he raises the belt", "the ring-walk"), OR
  quiet content with no crowd → DO NOT use `signals`.** A silent/visual moment has no
  audio peak and the energy would steer you AWAY from it. Locate visually, per ~5-min
  window `[T, T+300]`:
  ```bash
  autocut sheet VIDEO --interval 2 --from T --to T+300 --cell-px 480 --cols 4 --rows 4 --out CLIPS/coarse_T
  ```
  → open it and spot **every** candidate region (there may be several; every 2s so a
  fast knockdown can't hide between frames).

**For each candidate window (from `signals` OR the coarse pass):**
1. **Fine** (dense, NATIVE res) on `[a, b]` — **pad it ±5–8s** so the whole event (cause AND count/aftermath) fits well inside, never at the edge: `autocut sheet VIDEO --fps 2 --from a-8 --to b+8 --cell-px 640 --cols 3 --rows 4 --out CLIPS/fine_a` → open it, confirm what it actually is (a real peak vs routine action), pick exact `[start, end]`.
2. **Cut immediately**: `autocut cut VIDEO --start <START> --end <END> --accurate -o CLIPS/s<score>_clip_NN.mp4`.
3. **⭐ VERIFY THE CUT (MANDATORY — do not skip):** sheet the clip and confirm the
   decisive event is inside it (see *Verify every cut* below); RE-CUT if missing. If you
   located via `signals`, also **cross-check**: a real knockdown should sit on an audio
   peak — a "knockdown" with no nearby peak is suspect. Only then move on.

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

### Selection discipline — your ONLY job is to pick the best moments

Read this before judging. These rules keep selection honest and are where agents
most often go wrong:

1. **Match the request exactly — no more, no less.**
   - Asked for **knockouts / KOs** → keep ONLY knockouts (actual knockdowns/KOs), not
     every exchange, not near-KOs.
   - Asked for **highlights / best / salient moments** → keep KOs **and** the other
     peaks (near-KOs, staggering blows, heated sustained exchanges, slow-mo replays).
     **Do NOT keep only the knockouts.** A salient-moments request wants those strong
     non-KO moments too — dropping a genuine big moment "to stay strict" is wrong here.
   - Asked for a **specific moment (query)** → find exactly that one thing.
2. **If what was asked for is not in the video, keep nothing for it.** Never force a
   "best of nothing" cut just to have an output — skip it and move on. Empty is valid.
3. **A window can hold SEVERAL moments, not one.** Don't stop at the first peak in a
   chunk — capture every moment that qualifies, each as its own clip.
4. **Do NOT reason about the state of the match — it is not your task and it makes you
   wrong.** Never conclude "the fight is over", "this is a TKO", "the match ended", or
   "this is the final round". Whether an event ends the bout is irrelevant to whether
   it makes a good clip. Just select good moments.
5. **A break is a non-moment — skip it, don't interpret it.** A full-screen graphic, a
   sponsor card, an empty ring with stools, fighters walking to corners = not a moment.
   Pass over it and keep scanning; never treat it as the end of anything, and never
   extend a clip into it.
6. **Slow-motion REPLAYS are top-value — score them at the MAXIMUM.** Broadcasts replay
   the best action (a KO, a big hit) in slow motion, often during the breaks. Keep every
   replay: it is prime viral footage and often your cleanest view of a fast event.
7. **Score on CLEAN, DECISIVE ACTION — not on activity or proximity.** A highlight is
   the action LANDING: a clean combination connecting, a shot that visibly hurts, a
   goal scored, a punchline that lands. Fighters merely close and busy with punches
   that MISS / are BLOCKED / are LIGHT is NOT a highlight (~4) — "looks intense" is not
   the test, clean CONTACT is. Verify the contact on the fine pass before scoring.
8. **Label by what you see, and KEEP only score ≥ 6 (drop everything below).** Only a
   clear knockout / knockdown WITH a count is a "KO" (9–10); a strong moment with clean
   landed action is a good highlight (7–8); activity without clean contact is ~4 and
   DROPS. Don't inflate, and don't keep sub-6 clips to pad the set.

### On HOST: load the SAME rules the cloud uses (MANDATORY first step of judging)

**Before you look at a single sheet to pick clips, you MUST run and FOLLOW:**

```bash
autocut guidance highlights                  # or: talk | hybrid
autocut guidance highlights --sport boxing   # + thin recognition layer for a sport
```

For a combat/known sport, add `--sport <name>` (e.g. `boxing`) — it appends a small
"what a count / knockdown / strong moment LOOKS like in this sport" layer on top of
the generic rules. Unknown sport → generic highlights (the generic always works).
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

### How to read a peak off the sheets (what to extract, and how)

The decisive event is usually TOO FAST to land in a cell — you see its CONSEQUENCE
(a reaction, an intervention, a celebration, a result on screen), not the act itself:

- Read the consequence first, then **ANCHOR BACKWARDS**: the event that caused it is
  2–5s earlier (the low frame rate hides the instant of impact). Set START there, END
  after the consequence that proves it.
- **If at the current cell size you cannot tell WHAT you are looking at** — a decisive
  moment or just routine action — that ambiguity is itself the signal to **re-sheet
  that window at `--cell-px 640`** before deciding. Never score or cut on a cell you
  cannot clearly read: re-sheet, don't guess.
- **Never judge an event that sits at the EDGE of your fine window.** If the candidate
  (a fighter going down, a referee stepping in, a celebration) appears in the FIRST or
  LAST cells of a fine sheet, then its cause or its aftermath (e.g. the count) is
  OUTSIDE the window — you are seeing the event only HALF. WIDEN the window by several
  seconds on that side and re-sheet before deciding. Always pad fine windows generously
  (±5–8s around the candidate). A half-seen event is the #1 cause of a real knockdown
  being wrongly dismissed as "just an exchange".

### Verify every cut (MANDATORY self-check)

A cut can be wrong two ways: it ends BEFORE the decisive moment, OR you mis-saw the
moment entirely on small, blurry cells (routine action read as a peak). **After every
`cut`, re-sheet the clip you produced — at NATIVE resolution, so the check can actually
catch a misread instead of rubber-stamping it:**

```bash
autocut sheet CLIPS/s<score>_clip_NN.mp4 --fps 2 --cell-px 640 --cols 3 --rows 4 --out CLIPS/verify_NN
```

⚠️ **The verify is worthless at the same small cell size you SELECTED with — you would
just re-confirm your own mistake.** Use big cells (`--cell-px 640`).

Open the sheets and confirm the **whole decisive event is genuinely inside the clip AND
is what you thought it was**: the action AND its result AND the aftermath that proves it
(e.g. the fighter going down AND the referee's count; the goal AND the net bulging). If
you still cannot clearly confirm it at native resolution, it is NOT a verified
highlight — drop it or get a higher-fidelity check, do not ship a guess. If the clip
starts late, ends before the payoff, or cuts the impact — RE-CUT and verify again. Do
not merge or report a clip until it passes. This loop catches both the #1 error
(cutting the event off) and the misread error (inventing one that isn't there).
