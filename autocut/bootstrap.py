"""Install AutoCut's agent skill manifests into detected AI agents.

`autocut bootstrap` is the missing engine behind cross-agent distribution: copying
markdown with ``npx skills`` lands the SKILL.md files but not the ``autocut`` CLI
they drive, and ``uv tool install`` lands the CLI but not the skills. Bootstrap
closes the gap from the CLI side — after the tool is installed it detects which
agents are present on the machine and writes the bundled skill manifests into each
agent's skills directory.

The skill manifests are a SINGLE source of truth at ``.claude/skills/`` in the repo;
the wheel mirrors them to ``autocut/_skills/`` (see ``pyproject.toml`` force-include),
so the same files are reachable both from a source checkout and from an installed
tool. No VLM, no network — pure filesystem.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path


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


# ---------------------------------------------------------------------------
# Agent detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Agent:
    """A known AI agent that consumes ``SKILL.md`` manifests."""

    agent_id: str
    label: str
    marker: Path  # existence of this path means the agent is installed for the user
    skills_dir: Path  # where this agent looks up skills

    def installed(self) -> bool:
        return self.marker.exists()


def known_agents(home: Path | None = None) -> list[Agent]:
    """Registry of agents that use the Claude SKILL.md format.

    ``home`` overrides the user home (for tests). Claude Code and Cowork share the
    ``~/.claude/skills`` convention; project-local ``./.claude/skills`` is also a
    valid target an agent will read when run from that directory.
    """
    base = home if home is not None else Path.home()
    return [
        Agent(
            agent_id="claude-code",
            label="Claude Code / Cowork",
            marker=base / ".claude",
            skills_dir=base / ".claude" / "skills",
        ),
    ]


def detect_agents(home: Path | None = None) -> list[Agent]:
    """Return the subset of known agents that are actually installed."""
    return [a for a in known_agents(home) if a.installed()]


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InstallResult:
    """Outcome of installing one skill into one agent."""

    agent_id: str
    skill: str
    action: str  # "installed" | "updated" | "unchanged" | "would-install" | "would-update"
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


def install_skills(
    agents: list[Agent],
    skills: list[Skill],
    *,
    force: bool = False,
    dry_run: bool = False,
) -> list[InstallResult]:
    """Copy each skill into each agent's skills directory (idempotent).

    A skill already present and identical is left untouched ("unchanged") unless
    ``force``. When ``dry_run`` nothing is written; the planned action is reported.
    """
    results: list[InstallResult] = []
    for agent in agents:
        for skill in skills:
            dest = agent.skills_dir / skill.name
            already = _same_tree(skill.path, dest)
            exists = dest.exists()
            if already and not force:
                results.append(InstallResult(agent.agent_id, skill.name, "unchanged", dest))
                continue
            verb, past = ("update", "updated") if exists else ("install", "installed")
            if dry_run:
                results.append(InstallResult(agent.agent_id, skill.name, f"would-{verb}", dest))
                continue
            if dest.exists():
                shutil.rmtree(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(skill.path, dest)
            results.append(InstallResult(agent.agent_id, skill.name, past, dest))
    return results
