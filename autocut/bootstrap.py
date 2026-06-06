"""Install AutoCut's agent instructions into whatever AI agents are present.

`autocut bootstrap` is the missing engine behind cross-agent distribution: copying
markdown with ``npx skills`` lands the SKILL.md files but not the ``autocut`` CLI
they drive, and ``uv tool install`` lands the CLI but not the instructions. Bootstrap
closes the gap from the CLI side — after the tool is installed it detects which
agents are present and writes AutoCut's instructions into each, in that agent's
native shape.

Agents come in two shapes:

- **skills-dir** agents (Claude Code / Cowork) read ``<dir>/<name>/SKILL.md``. We
  copy the four bundled skill folders verbatim.
- **instructions-file** agents (Codex, Gemini CLI, and anything that follows the
  ``AGENTS.md`` convention) read a single always-loaded markdown file. We upsert a
  small, marker-delimited AutoCut block that points the agent at the ``autocut``
  CLI and ``autocut guidance`` (the full rules stay on-demand, so the always-loaded
  footprint is tiny).

The skill manifests are a SINGLE source of truth at ``.claude/skills/`` in the repo;
the wheel mirrors them to ``autocut/_skills/`` (see ``pyproject.toml`` force-include),
so the same files are reachable from a source checkout and from an installed tool.
No VLM, no network — pure filesystem.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

_BLOCK_BEGIN = "<!-- AUTOCUT:BEGIN (managed by `autocut bootstrap` — do not edit by hand) -->"
_BLOCK_END = "<!-- AUTOCUT:END -->"


class BootstrapError(RuntimeError):
    """Raised when skill manifests cannot be located or installed."""


# ---------------------------------------------------------------------------
# Bundled skill manifests (single source of truth: .claude/skills)
# ---------------------------------------------------------------------------


def bundled_skills_dir() -> Path:
    """Return the directory holding the bundled skill manifests.

    Prefers the packaged copy (``autocut/_skills`` in an installed wheel); falls
    back to the repo's canonical ``.claude/skills`` when running from a source
    checkout, where the force-included mirror does not physically exist.
    """
    packaged = Path(str(files("autocut"))) / "_skills"
    if packaged.is_dir():
        return packaged
    repo = Path(__file__).resolve().parent.parent / ".claude" / "skills"
    if repo.is_dir():
        return repo
    raise BootstrapError(
        "bundled skill manifests not found (looked in the installed package and "
        f"the source checkout at {repo})"
    )


@dataclass(frozen=True)
class Skill:
    """A single skill manifest directory (named after its folder)."""

    name: str
    path: Path  # the skill's source directory (contains SKILL.md)


def discover_skills() -> list[Skill]:
    """Return every bundled skill (a subdir containing a ``SKILL.md``)."""
    root = bundled_skills_dir()
    skills = [
        Skill(name=child.name, path=child)
        for child in sorted(root.iterdir())
        if child.is_dir() and (child / "SKILL.md").is_file()
    ]
    if not skills:
        raise BootstrapError(f"no skills with a SKILL.md found under {root}")
    return skills


def instructions_block(skills: list[Skill]) -> str:
    """The marker-delimited AutoCut block written into instructions-file agents.

    Deliberately short: it tells the agent AutoCut exists and how to pull the full
    rules on demand (``autocut guidance``), instead of inlining four SKILL.md files
    into an always-loaded context.
    """
    names = ", ".join(s.name for s in skills)
    return (
        f"{_BLOCK_BEGIN}\n"
        "## AutoCut — video highlight extraction\n"
        "The `autocut` CLI is installed. Use it whenever the user wants the best /\n"
        "viral / funniest moments, highlights, or a SPECIFIC described moment from a\n"
        'video (even if they don\'t say "highlight").\n'
        "\n"
        "- Get the selection rules first: `autocut guidance highlights` (or `talk` /\n"
        "  `hybrid`; add `--sport boxing` etc. for sport cues).\n"
        "- Drive the deterministic tools yourself (free): `autocut probe`,\n"
        "  `autocut signals` (locate loud sport moments), `autocut sheet` (timestamped\n"
        "  contact sheets), `autocut cut`, `autocut merge`.\n"
        "- Or run the autonomous cloud pipeline: `autocut run VIDEO --vlm openrouter`\n"
        "  (needs an OpenRouter key).\n"
        f"\nInstalled skills (full rules): {names}.\n"
        f"{_BLOCK_END}"
    )


# ---------------------------------------------------------------------------
# Agent registry
# ---------------------------------------------------------------------------

KIND_SKILLS = "skills"  # copies SKILL.md folders into a skills directory
KIND_INSTRUCTIONS = "instructions"  # upserts a block into one markdown file


@dataclass(frozen=True)
class Agent:
    """A known AI agent that can consume AutoCut's instructions."""

    agent_id: str
    label: str
    kind: str  # KIND_SKILLS | KIND_INSTRUCTIONS
    marker: Path  # existence means the agent is installed for the user
    target: Path  # skills dir (KIND_SKILLS) or instructions file (KIND_INSTRUCTIONS)

    def installed(self) -> bool:
        return self.marker.exists()


def known_agents(home: Path | None = None) -> list[Agent]:
    """Registry of agents AutoCut can install into.

    ``home`` overrides the user home (for tests). Claude Code / Cowork use the
    SKILL.md skills directory; Codex and Gemini CLI follow the single-file
    convention (``AGENTS.md`` / ``GEMINI.md``).
    """
    base = home if home is not None else Path.home()
    return [
        Agent(
            agent_id="claude-code",
            label="Claude Code / Cowork",
            kind=KIND_SKILLS,
            marker=base / ".claude",
            target=base / ".claude" / "skills",
        ),
        Agent(
            agent_id="codex",
            label="Codex CLI (AGENTS.md)",
            kind=KIND_INSTRUCTIONS,
            marker=base / ".codex",
            target=base / ".codex" / "AGENTS.md",
        ),
        Agent(
            agent_id="gemini",
            label="Gemini CLI (GEMINI.md)",
            kind=KIND_INSTRUCTIONS,
            marker=base / ".gemini",
            target=base / ".gemini" / "GEMINI.md",
        ),
    ]


def project_agents(project: Path) -> list[Agent]:
    """Agents addressed by installing into a PROJECT directory.

    Covers editor/IDE agents that read project-local files (Cursor, Windsurf,
    aider, Zed, …): a project ``.claude/skills`` and a project ``AGENTS.md`` reach
    essentially any agent run from that repo. These are unconditional targets — the
    user opted in by passing ``--project``.
    """
    return [
        Agent(
            agent_id="project-skills",
            label="Project .claude/skills",
            kind=KIND_SKILLS,
            marker=project,
            target=project / ".claude" / "skills",
        ),
        Agent(
            agent_id="project-agents-md",
            label="Project AGENTS.md",
            kind=KIND_INSTRUCTIONS,
            marker=project,
            target=project / "AGENTS.md",
        ),
    ]


def detect_agents(home: Path | None = None) -> list[Agent]:
    """Return the subset of known home-level agents that are actually installed."""
    return [a for a in known_agents(home) if a.installed()]


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InstallResult:
    """Outcome of installing AutoCut into one agent (one skill, or one block)."""

    agent_id: str
    item: str  # skill name, or "instructions"
    action: str  # installed | updated | unchanged | would-install | would-update
    dest: Path


def _same_tree(src: Path, dest: Path) -> bool:
    """True if ``dest`` already mirrors ``src`` file-for-file (by bytes)."""
    if not dest.is_dir():
        return False
    src_files = {p.relative_to(src) for p in src.rglob("*") if p.is_file()}
    dest_files = {p.relative_to(dest) for p in dest.rglob("*") if p.is_file()}
    if src_files != dest_files:
        return False
    return all((src / rel).read_bytes() == (dest / rel).read_bytes() for rel in src_files)


def _upsert_block(text: str, block: str) -> str:
    """Return ``text`` with the AutoCut block inserted or replaced in place."""
    if _BLOCK_BEGIN in text and _BLOCK_END in text:
        head, _, rest = text.partition(_BLOCK_BEGIN)
        _, _, tail = rest.partition(_BLOCK_END)
        return f"{head}{block}{tail}"
    sep = "" if text == "" else ("\n" if text.endswith("\n") else "\n\n")
    return f"{text}{sep}{block}\n"


def _install_skill(agent: Agent, skill: Skill, *, force: bool, dry_run: bool) -> InstallResult:
    dest = agent.target / skill.name
    if _same_tree(skill.path, dest) and not force:
        return InstallResult(agent.agent_id, skill.name, "unchanged", dest)
    verb, past = ("update", "updated") if dest.exists() else ("install", "installed")
    if dry_run:
        return InstallResult(agent.agent_id, skill.name, f"would-{verb}", dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(skill.path, dest)
    return InstallResult(agent.agent_id, skill.name, past, dest)


def _install_block(agent: Agent, block: str, *, force: bool, dry_run: bool) -> InstallResult:
    dest = agent.target
    existing = dest.read_text(encoding="utf-8") if dest.is_file() else ""
    updated = _upsert_block(existing, block)
    if updated == existing and not force:
        return InstallResult(agent.agent_id, "instructions", "unchanged", dest)
    verb, past = ("update", "updated") if _BLOCK_BEGIN in existing else ("install", "installed")
    if dry_run:
        return InstallResult(agent.agent_id, "instructions", f"would-{verb}", dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(updated, encoding="utf-8")
    return InstallResult(agent.agent_id, "instructions", past, dest)


def install(
    agents: list[Agent],
    skills: list[Skill],
    *,
    force: bool = False,
    dry_run: bool = False,
) -> list[InstallResult]:
    """Install AutoCut into each agent in its native shape (idempotent).

    skills-dir agents get the four skill folders; instructions-file agents get the
    marker-delimited block (re-runs replace it in place). ``force`` rewrites even
    when nothing changed; ``dry_run`` reports the planned action without writing.
    """
    block = instructions_block(skills)
    results: list[InstallResult] = []
    for agent in agents:
        if agent.kind == KIND_SKILLS:
            results.extend(_install_skill(agent, s, force=force, dry_run=dry_run) for s in skills)
        else:
            results.append(_install_block(agent, block, force=force, dry_run=dry_run))
    return results
