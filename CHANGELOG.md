# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0-beta.2] - 2026-07-29

### Fixed

- **Windows: hostile filenames no longer crash the ffmpeg helpers.** Every
  subprocess wrapper (`probe`, `cut`, `concat`, `compress`, audio extract,
  contact sheet, keyframes, CLI version check) decoded ffmpeg/ffprobe output
  with `text=True`, i.e. the locale ANSI codepage. ffmpeg speaks UTF-8 and
  echoes the input filename in its output (ffprobe embeds it in the JSON
  payload), so filenames with bytes outside the codepage — e.g. Japanese,
  whose UTF-8 encoding contains `0x81`/`0x90`, holes in cp1252 — raised
  `UnicodeDecodeError` instead of processing the file. All helpers now decode
  UTF-8 explicitly with `errors="replace"`. Found by a downstream dress
  rehearsal (cutman-ai) delivering 10 hostile-named files through a watcher.

## [0.1.0-beta.1] - 2026-06-06

First public beta. The CLI, the cross-agent skills, and both analysis paths
(agent-driven host + autonomous OpenRouter cloud) are usable; APIs may still shift
before the stable `v0.1.0`.

### Added

- `autocut wait-ready VIDEO` — a deterministic readiness gate the orchestrating
  agent calls before `probe`/`run` when a file may still be arriving (e.g. a phone
  transfer into a watched folder). A file *exists* the instant a copy starts but
  keeps growing; touching it early can probe or cut a truncated video. The command
  polls the file's `(size, mtime)` and the read lock, returning only once they have
  held steady for `--stable-for` seconds (also waits for a not-yet-present file to
  appear, up to `--timeout`). Exit 0 = ready; exit 1 = timed out. Wired as
  "Step 0.5" of the `autocut-run` recipe (both host and cloud paths).
- `autocut bootstrap` is now implemented and works across agents, not just Claude.
  It detects installed agents and installs AutoCut in each one's native shape,
  idempotently (`--list`, `--dry-run`, `--force`):
  - **skills-dir agents** (Claude Code / Cowork) get the four SKILL.md folders
    copied into `~/.claude/skills/`.
  - **instructions-file agents** (Codex via `~/.codex/AGENTS.md`, Gemini CLI via
    `~/.gemini/GEMINI.md`) get a small, marker-delimited AutoCut block upserted
    into their always-loaded instructions file (existing content preserved;
    re-runs replace the block in place, never duplicate it).
  - `--project PATH` (or `-p .`) additionally installs into a repo's
    `.claude/skills` + `AGENTS.md`, reaching editor agents (Cursor, Windsurf,
    aider, Zed, …) that read project-local files.
  The manifests are a single source of truth at `.claude/skills/` and are bundled
  into the wheel (`autocut/_skills/`), so a `uv tool install` ships them and
  bootstrap can lay them down — closing the gap where `npx skills` copies markdown
  but not the CLI engine the skills drive.
- One-line installers `install.sh` (macOS/Linux) and `install.ps1` (Windows):
  install `uv` if missing, `uv tool install` the CLI (ffmpeg bundled), run
  `autocut bootstrap`, then `autocut doctor`.
- Video-input analysis path (Phase G). Instead of sampling keyframes, the
  source is compressed to an analysis-grade copy and the model watches the
  video directly — it perceives motion and catches the decisive moment a flat
  keyframe batch dilutes, at a fraction of the cost.
  - **OpenRouter**: the compressed clip is sent inline as a base64 `video_url`
    block (pinned to the Vertex backend), split into overlapping ~5-minute
    batches by the L2 engine (`autocut.video_analysis`) when needed; the real
    billed cost is surfaced in the manifest.
  - **Host agent**: opt-in via `autocut run --vlm host --host-video`. The
    compressed MP4 is referenced by path in `VLM_REQUEST.md` and the agent
    watches it during the existing pause/resume — single pass, no base64 size
    ceiling. Defaults off (keyframe path) and is honoured across `resume`. If
    the agent cannot open a video, the request tells it to re-run without
    `--host-video` to fall back to keyframes.
- `autocut.pipeline._select_route` centralises the payload-x-transport routing
  (keyframe / openrouter-video / host-video) in one table so new payloads
  (e.g. audio) add a route and a thin runner rather than another `if` branch.

### Changed

- The `motion` sampler now sends the model ONLY the hot windows (sampled at one
  frame per second) and skips the dead time entirely, instead of a sparse
  all-video baseline plus dense windows. Feeding a stills VLM just the key
  moments stops the decisive action from being diluted by idle frames. With no
  hot window detected it still falls back to a uniform baseline.
- ffmpeg/ffprobe are now resolved through `autocut.video.ffmpeg_path`
  (system PATH first, then a bundled `static-ffmpeg` fallback fetched on first
  use). `static-ffmpeg` is a required dependency — it ships both `ffmpeg` and
  `ffprobe`, unlike the previous optional `imageio-ffmpeg` (ffmpeg only), which
  was removed. `autocut doctor` now reports which source each binary resolves
  to (system vs bundled) and its full path.
- `autocut run` now always produces per-clip `separate` outputs; the `--output`
  flag was removed. Composing a single reel is the job of the deterministic
  `autocut merge --from-manifest --min-score N` subcommand. This also removes a
  resume bug where a runtime-selected `merged`/`all` mode was not persisted to
  the resume sidecar, so a resumed host-agent run silently fell back to
  `separate` only.
- Prompt templates bumped to `v2`: sport and talk clip-boundary guidance now
  tell the model to include the wind-up and follow-through of the key moment
  (never end on the frame of impact / the punchline) instead of cutting tight
  on the action. Smaller models (e.g. Gemini Flash) followed the old "tight, no
  preamble" wording literally and cut right on the punch.

### Fixed

- OpenRouter provenance metadata (`vlm_provider`, `vlm_model`, `prompt_version`,
  `analysis_time_sec`) is now authoritative on our side instead of trusting the
  model's self-report. Gemini 3.x Flash misidentified itself as `gemini-1.5-pro`,
  corrupting the manifest's model attribution and cost tracking.
- OpenRouter content detection no longer truncates: the call's `max_tokens` was
  raised (256 → 2048) because "thinking" models spend part of the budget on
  internal reasoning, which left too little for the JSON and cut it mid-string —
  silently dropping the run into the HYBRID fallback. Response parsing now also
  strips markdown code fences / surrounding prose, and JSON errors include a
  snippet of the raw response for debugging.

### Scope of v0.1.0

- VLM providers: `host` (uses Claude Code / Cowork subscription) and `openrouter`
  (gateway to 200+ models via the `openai` SDK with custom `base_url`).
- `autocut run` produces `separate` per-clip outputs; reels are composed
  separately via `autocut merge`. The `merged` output writer remains available
  internally and through the `merge` subcommand.
- Output directory: always `./CLIPS/` in the user's cwd.

### Explicitly deferred

- Direct Anthropic / OpenAI / Gemini SDKs → v0.2.0
- CapCut export (`pyCapCut`) → v0.2.0 (pending upstream license clarification)
- Local providers (Ollama, LM Studio) → v0.3.0
- MCP server, GUI → v0.4.0+

## [0.1.0a1] — TBD

Initial alpha. Project scaffolding, security primitives, CLI skeleton.
