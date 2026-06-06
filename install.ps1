<#
.SYNOPSIS
  AutoCut one-line installer (Windows).

.DESCRIPTION
  Installs the `autocut` CLI as an isolated uv tool (ffmpeg is bundled), then
  installs the agent skill manifests and runs a health check. Idempotent: re-run
  any time to upgrade. Override the source with $env:AUTOCUT_REPO.

.EXAMPLE
  powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/ArtiomIV/autocut-skill/main/install.ps1 | iex"
#>
$ErrorActionPreference = 'Stop'

$Repo = if ($env:AUTOCUT_REPO) { $env:AUTOCUT_REPO } else { 'git+https://github.com/ArtiomIV/autocut-skill' }

function Say($msg) { Write-Host $msg -ForegroundColor Cyan }

# 1. Ensure uv is available (the only prerequisite).
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Say 'uv not found - installing it first...'
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
}
# Make uv and its installed tools reachable in THIS session.
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"

# 2. Install (or upgrade) the CLI in its own environment.
Say "Installing the autocut CLI from $Repo ..."
uv tool install --force $Repo

# 3. Install the agent skill manifests (Claude Code / Cowork, ...).
Say 'Installing agent skills ...'
autocut bootstrap
if ($LASTEXITCODE -ne 0) {
    Say "No AI agent detected yet - run 'autocut bootstrap' once your agent is installed."
}

# 4. Verify the environment (never fatal).
try { autocut doctor } catch {}

Say 'Done. Try:  autocut run my_video.mp4'
