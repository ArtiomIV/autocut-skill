#!/usr/bin/env sh
# AutoCut one-line installer (macOS / Linux).
#
#   curl -LsSf https://raw.githubusercontent.com/ArtiomIV/autocut-skill/main/install.sh | sh
#
# Installs the `autocut` CLI as an isolated uv tool (ffmpeg is bundled), then
# installs the agent skill manifests and runs a health check. Idempotent: re-run
# any time to upgrade. Override the source with AUTOCUT_REPO=...
set -eu

REPO="${AUTOCUT_REPO:-git+https://github.com/ArtiomIV/autocut-skill}"

say() { printf '\033[1m%s\033[0m\n' "$*"; }

# 1. Ensure uv is available (the only prerequisite).
if ! command -v uv >/dev/null 2>&1; then
  say "uv not found — installing it first…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# Make uv and its installed tools reachable in THIS shell session.
export PATH="$HOME/.local/bin:$PATH"

# 2. Install (or upgrade) the CLI in its own environment.
say "Installing the autocut CLI from $REPO …"
uv tool install --force "$REPO"

# 3. Install the agent skill manifests (Claude Code / Cowork, …).
say "Installing agent skills …"
if ! autocut bootstrap; then
  say "No AI agent detected yet — run 'autocut bootstrap' once your agent is installed."
fi

# 4. Verify the environment (never fatal).
autocut doctor || true

say "Done. Try:  autocut run my_video.mp4"
