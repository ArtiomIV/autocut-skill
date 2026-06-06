# AutoCut Skill

> **Status: v0.1.0-alpha (in development).** Open-source cross-agent skill for automatic
> video highlight extraction using a Vision Language Model (VLM).

AutoCut analyses a video with a VLM, ranks the most clip-worthy segments, and cuts them
out with `ffmpeg` — saved as individual clips, a merged highlight reel, or both.

## Why AutoCut

- **Use your existing Claude Code / Cowork subscription** — the `host` provider delegates
  inference to the host agent, no extra API key, no extra cost.
- **One OpenRouter key, 200+ VLM models** — the `openrouter` provider is a thin shim over
  the OpenAI SDK pointed at `https://openrouter.ai/api/v1`. Pay-per-use, prices visible
  live in the config wizard.
- **Cross-agent** — works in Claude Code, Cowork, Codex CLI, Gemini CLI, Cursor, aider.
- **Local-first** — no SaaS, no mandatory upload to proprietary platforms, no telemetry.
  Output stays in `./CLIPS/` in the user's cwd.

## Status

Currently building **v0.1.0**. See [`CHANGELOG.md`](CHANGELOG.md) for what is in scope and
[`DESIGN.md`](../DESIGN.md) (working doc) for the full architecture.

| Version | What it adds | Status |
|---|---|---|
| v0.1.0 | `host` + `openrouter` providers, `separate` / `merged` / `all` output | in progress |
| v0.2.0 | Direct SDKs (Anthropic, OpenAI, Gemini) + CapCut export | planned |
| v0.3.0 | Local providers (Ollama, LM Studio) | planned |
| v0.4.0 | Hardening, optional GUI, optional MCP wrapper | planned |

## Quick start

**One-line install** (installs `uv` if missing, the `autocut` CLI with bundled
ffmpeg, and the agent skills):

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

Then just ask your agent for the highlights, or drive it yourself:

```bash
autocut run my_video.mp4    # cloud pipeline (needs an OpenRouter key); host mode is free
```

`autocut bootstrap` writes the four skills (`autocut-run`, `autocut-cut`,
`autocut-merge`, `autocut-detect`) into every detected agent's skills directory
(e.g. `~/.claude/skills/`). Re-run it any time to refresh; it is idempotent.

## Security model

API keys are stored **only in the OS keyring** (Windows Credential Manager / macOS
Keychain / Linux Secret Service). The project never reads or writes `.env` or plaintext
secret files. A `gitleaks` pre-commit hook blocks accidental commits of key-shaped
strings.

## License

MIT — see [`LICENSE`](LICENSE).

## Acknowledgments

- [PySceneDetect](https://www.scenedetect.com/) (BSD-3) — scene detection
- [ffmpeg](https://ffmpeg.org/) (LGPL/GPL) — video cutting
- [OpenRouter](https://openrouter.ai/) — universal VLM gateway
