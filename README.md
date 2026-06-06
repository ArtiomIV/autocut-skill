# AutoCut Skill

> **Status: `v0.1.0-beta` — usable, still stabilising.** Open-source, cross-agent skill
> for automatic video highlight extraction with a Vision Language Model (VLM).

AutoCut finds the most clip-worthy moments of a video, cuts them with `ffmpeg`, and can
stitch them into a highlight reel — as individual clips, a merged reel, or both. It is
built as a set of **small deterministic commands your AI agent orchestrates**: the
*judgement* ("which moments are good?") is the model's, and everything else (probing,
sampling, cutting, joining) is exact, reproducible, and free.

---

## How it works

AutoCut deliberately splits every job into two kinds of step:

1. **Deterministic tools (no AI)** — thin, predictable wrappers around `ffmpeg` /
   computer-vision. Same input → same output, every time. These never cost anything.
2. **One judging step (the AI)** — the only place a model is involved: deciding which
   moments are worth keeping and scoring them.

The agent runs the tools and feeds the model exactly what it needs to judge. There are
**two ways** to do that judging:

- **HOST — free, no API key.** Your agent (Claude Code, Cowork, …) opens the contact
  sheets *itself*, reads the frames, picks the moments, and calls `cut`/`merge`. The
  agent **is** the brain — you can steer it in chat. Best for short/medium videos and
  for interviews.
- **CLOUD — autonomous, paid.** `autocut run --vlm openrouter` sends the video to a
  cloud VLM, which writes a ranked `plan.json` on its own; you then `cut` it. Best for
  long videos or fire-and-forget. Needs an OpenRouter key (stored in your OS keyring).

Both paths share the same editorial rules (`autocut guidance`) so they pick clips the
same way.

---

## The commands — what each one is for and how it works

AutoCut is intentionally **atomic**: each command does one thing, so an agent can
compose them (and skip the expensive VLM step whenever it isn't needed). Run
`autocut --help` or `autocut <cmd> --help` for the full option list.

### Deterministic — no AI, no cost

#### `autocut wait-ready VIDEO`
**What:** blocks until a file has finished being written, then exits.
**Why:** a file *exists* the instant a copy starts but its bytes keep arriving for
seconds (a phone transfer into a watched folder, an upload, a network copy). Touching it
too early can probe or cut a **truncated** video.
**How:** it polls the file's `(size, mtime)` and tries to open it for reading, and only
returns once they've held steady for `--stable-for` seconds (it also waits for a
not-yet-present file to appear, up to `--timeout`). Exit `0` = ready; exit `1` = timed
out (still copying / locked / never arrived). Call it **first**, before `probe`/`run`,
whenever the file might still be arriving.

#### `autocut probe VIDEO`
**What:** prints the video's metadata as JSON (duration, fps, resolution, codecs, size).
**Why:** it's the first look at the file — the agent uses the duration to choose **host
vs cloud** and the resolution to choose the contact-sheet density.
**How:** a thin `ffprobe` wrapper; fails cleanly if the file is missing or unreadable.

#### `autocut signals VIDEO`
**What:** prints **advisory** localisation signals as JSON — audio-energy peaks
(`impact`/`crowd`, ranked by strength) plus scene cuts (replays, graphic transitions).
**Why:** on a long, loud sport video it tells the agent *where to look first*, so it can
sheet only the interesting windows instead of the whole timeline (≈2× less to read).
**How:** extracts mono audio → a rolling-median/IQR adaptive peak detector; scene cuts
via PySceneDetect. It is **advisory, never a gate** — for a `--query` or quiet content
the agent ignores it and scans visually. `-k/--sensitivity` tunes how many peaks.

#### `autocut sheet VIDEO --out DIR`
**What:** renders a window of the video into **timestamped contact-sheet grids** — one
image per grid, each cell a frame with its integer index burned in — plus an
`index.json` mapping every index to its exact second.
**Why:** this is how a model "watches" the video cheaply and with **exact timestamps**:
read a cell's number, look it up in `index.json`, get the precise second to cut.
**How:** samples `[--from, --to]` at `--fps` (dense) or `--interval` (sparse). Use a
sparse `--interval 3` to locate regions across a long video, then a dense `--fps 2` on
the chosen window for sub-second boundaries. `--cell-px`/`--cols`/`--rows` trade
legibility for token cost (bigger cells = clearer, fewer per sheet).

#### `autocut cut`
**What:** trims clips with `ffmpeg`. Two modes:
- `--from-json plan.json --output-dir DIR` — cut **every** clip in a plan (from
  `autocut run`), optionally filtered by `--min-score`. Writes `separate/*.mp4` + a
  `manifest.json`.
- `VIDEO --start S --end E -o out.mp4` — cut **one** exact `[start, end]` segment.
**Why:** the actual export step — turns chosen timestamps into MP4 files. No model.
**How:** `--fast` (default) stream-copies (lossless, snaps to nearest keyframe);
`--accurate` re-encodes for frame-exact boundaries.

#### `autocut merge`
**What:** concatenates MP4 clips into one reel — positional files or
`--from-manifest manifest.json --min-score N`.
**Why:** builds the final highlight reel from already-cut clips. No model.
**How:** the `ffmpeg` concat demuxer; `--order chronological|score-desc|manifest`.
Inputs must share codec/resolution/fps.

#### `autocut guidance MODE`
**What:** prints the exact clip-selection rules for `highlights` / `talk` / `hybrid`
(optionally `--sport boxing` for a thin sport-recognition layer).
**Why:** these are the **same** editorial rules the cloud model gets in its prompt — one
source of truth, so the host agent and the cloud judge clips identically. On the host
path the agent reads this **before** judging the sheets.

### AI — needs a VLM (OpenRouter)

#### `autocut run VIDEO --vlm openrouter`
**What:** the autonomous cloud pipeline. Analyses the video and writes a ranked
`CLIPS/plan.json` (with the real + estimated cost recorded). It does **not** cut — you
review the plan, then run `autocut cut --from-json`.
**Why:** hands the whole "find the moments" job to a cloud VLM for long or
fire-and-forget videos.
**How:** for videos > 60s it runs a **two-pass** coarse→fine (locate regions, then
re-analyse each in isolation for tight boundaries; `--single-pass` to disable).
`--content-hint highlights|talk|hybrid` sets the mode; `--query "<moment>"` finds one
specific described moment.

#### `autocut detect VIDEO --vlm openrouter`
**What:** one cheap VLM call that classifies the video — `{content_hint, confidence,
reasoning}` — without running the full extraction.
**Why:** confirm the editing mode before spending on a full `run`. (Cloud only; on the
host path the agent classifies the video directly.)

### Setup & housekeeping

- `autocut bootstrap` — detect installed AI agents and install the four skills into each
  (idempotent; `--list`, `--dry-run`, `--force`, `--project PATH`).
- `autocut keys set openrouter` — store an API key in the OS keyring (never plaintext).
- `autocut doctor` — verify ffmpeg / config / keyring / providers.
- `autocut config` / `autocut models` — inspect config; list available cloud models.

### A typical run

```bash
autocut wait-ready my_video.mp4        # only if the file may still be copying in
autocut probe     my_video.mp4         # how long is it? → host or cloud?

# HOST (free): your agent sheets + reads + cuts for you, guided by `autocut guidance`.

# CLOUD (paid):
autocut run   my_video.mp4 --vlm openrouter --content-hint highlights
autocut cut   --from-json CLIPS/plan.json --video my_video.mp4 --output-dir CLIPS
autocut merge --from-manifest CLIPS/manifest.json --min-score 8 -o CLIPS/reel.mp4
```

---

## Why AutoCut

- **Use your existing Claude Code / Cowork subscription** — on the host path the agent
  reads the frames itself: no extra API key, no per-video cost.
- **Or one OpenRouter key, 200+ VLM models** — the cloud path is a thin shim over the
  OpenAI SDK pointed at OpenRouter. Pay-per-use; the real + estimated cost is written
  into every `plan.json`.
- **Cross-agent** — `autocut bootstrap` installs the skill into Claude Code, Cowork,
  Codex CLI, Gemini CLI, and (via `--project`) Cursor / Windsurf / aider / Zed.
- **Local-first** — no SaaS, no mandatory upload, no telemetry. Output stays in
  `./CLIPS/` in your cwd; API keys live only in the OS keyring.

## Quick start

**One-line install** (installs `uv` if missing, the `autocut` CLI with bundled ffmpeg,
and the agent skills):

```bash
# macOS / Linux
curl -LsSf https://raw.githubusercontent.com/ArtiomIV/autocut-skill/main/install.sh | sh
```

```powershell
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/ArtiomIV/autocut-skill/main/install.ps1 | iex"
```

**Or install the CLI directly** with [uv](https://docs.astral.sh/uv/) (ffmpeg is
bundled — nothing else to install):

```bash
uv tool install git+https://github.com/ArtiomIV/autocut-skill
autocut bootstrap           # detect installed AI agents + install the skill manifests
autocut doctor              # verify ffmpeg / config / providers
```

Then just ask your agent for the highlights, or drive the commands yourself.

## Roadmap

| Version | What it adds | Status |
|---|---|---|
| **v0.1.0-beta** | host (agent-driven) + openrouter cloud pipeline; `probe` / `signals` / `sheet` / `cut` / `merge` / `wait-ready` / `guidance`; cross-agent `bootstrap` | **current** |
| v0.2.0 | **Whisper for talk / interviews** — the agent can't "hear", so Whisper transcribes the audio and speech-driven highlights (quotes, key lines) finally work. Plus a **unified agent-orchestrated analysis**: one pipeline where DSP localises, the video is turned into sheets (or audio chunks), and the model judges each moment in isolation — retiring the costly whole-video cloud pass and sending audio to the cloud in timestamped chunks. | next |
| v0.3.0 | Local VLM providers (Ollama, LM Studio); CapCut export | planned |
| v0.4.0 | Hardening, optional GUI, optional MCP wrapper | planned |

See [`CHANGELOG.md`](CHANGELOG.md) for the detailed history.

## Security model

API keys are stored **only in the OS keyring** (Windows Credential Manager / macOS
Keychain / Linux Secret Service). The project never reads or writes `.env` or plaintext
secret files. A `gitleaks` pre-commit hook blocks accidental commits of key-shaped
strings. `subprocess` is always called with an argument list, never `shell=True`.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ArtiomIV/autocut-skill&type=Date)](https://star-history.com/#ArtiomIV/autocut-skill&Date)

## License

MIT — see [`LICENSE`](LICENSE).

## Acknowledgments

- [PySceneDetect](https://www.scenedetect.com/) (BSD-3) — scene detection
- [ffmpeg](https://ffmpeg.org/) (LGPL/GPL) — video probing & cutting
- [OpenRouter](https://openrouter.ai/) — universal VLM gateway
- [uv](https://docs.astral.sh/uv/) — packaging & tool install
