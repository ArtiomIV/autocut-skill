"""Unit tests for ``autocut.bootstrap`` — agent detection + install (both kinds).

Detection and install are exercised against a fake home and synthetic skill
directories in ``tmp_path`` (no real ``~/.claude`` mutation). The bundled-skills
accessor is checked against the repo's real ``.claude/skills``.
"""

from __future__ import annotations

from pathlib import Path

from autocut.bootstrap import (
    KIND_INSTRUCTIONS,
    KIND_SKILLS,
    Agent,
    Skill,
    detect_agents,
    discover_skills,
    install,
    instructions_block,
    known_agents,
    project_agents,
)


def _make_skill(root: Path, name: str, body: str = "# skill\n") -> Skill:
    d = root / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(body, encoding="utf-8")
    return Skill(name=name, path=d)


def _skills_agent(home: Path) -> Agent:
    return Agent(
        agent_id="claude-code",
        label="Claude Code",
        kind=KIND_SKILLS,
        marker=home / ".claude",
        target=home / ".claude" / "skills",
    )


def _instructions_agent(home: Path) -> Agent:
    return Agent(
        agent_id="codex",
        label="Codex",
        kind=KIND_INSTRUCTIONS,
        marker=home / ".codex",
        target=home / ".codex" / "AGENTS.md",
    )


# ---------------------------------------------------------------------------
# Bundled skills (real repo manifests)
# ---------------------------------------------------------------------------


def test_discover_skills_finds_the_repo_manifests() -> None:
    skills = discover_skills()
    names = {s.name for s in skills}
    assert {"autocut-run", "autocut-cut", "autocut-merge", "autocut-detect"} <= names
    assert all((s.path / "SKILL.md").is_file() for s in skills)


# ---------------------------------------------------------------------------
# Agent detection
# ---------------------------------------------------------------------------


def test_detect_agents_present(tmp_path: Path) -> None:
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".gemini").mkdir()
    detected = {a.agent_id for a in detect_agents(home=tmp_path)}
    assert detected == {"claude-code", "gemini"}


def test_detect_agents_absent(tmp_path: Path) -> None:
    assert detect_agents(home=tmp_path) == []


def test_registry_covers_skills_and_instructions(tmp_path: Path) -> None:
    kinds = {a.agent_id: a.kind for a in known_agents(home=tmp_path)}
    assert kinds["claude-code"] == KIND_SKILLS
    assert kinds["codex"] == KIND_INSTRUCTIONS
    assert kinds["gemini"] == KIND_INSTRUCTIONS


# ---------------------------------------------------------------------------
# Install — skills-dir agents
# ---------------------------------------------------------------------------


def test_install_skill_then_unchanged(tmp_path: Path) -> None:
    skill = _make_skill(tmp_path / "src", "autocut-run", "# run\n")
    agent = _skills_agent(tmp_path / "home")

    first = install([agent], [skill])
    assert [r.action for r in first] == ["installed"]
    dest = agent.target / "autocut-run" / "SKILL.md"
    assert dest.read_text(encoding="utf-8") == "# run\n"

    second = install([agent], [skill])
    assert [r.action for r in second] == ["unchanged"]


def test_install_skill_updates_on_change(tmp_path: Path) -> None:
    skill = _make_skill(tmp_path / "src", "autocut-run", "# v1\n")
    agent = _skills_agent(tmp_path / "home")
    install([agent], [skill])
    (skill.path / "SKILL.md").write_text("# v2\n", encoding="utf-8")

    res = install([agent], [skill])
    assert [r.action for r in res] == ["updated"]
    assert (agent.target / "autocut-run" / "SKILL.md").read_text(encoding="utf-8") == "# v2\n"


def test_install_skill_removes_stale_files(tmp_path: Path) -> None:
    skill = _make_skill(tmp_path / "src", "autocut-run")
    (skill.path / "extra.md").write_text("old\n", encoding="utf-8")
    agent = _skills_agent(tmp_path / "home")
    install([agent], [skill])
    assert (agent.target / "autocut-run" / "extra.md").exists()

    (skill.path / "extra.md").unlink()
    res = install([agent], [skill])
    assert [r.action for r in res] == ["updated"]
    assert not (agent.target / "autocut-run" / "extra.md").exists()


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    skill = _make_skill(tmp_path / "src", "autocut-run")
    agent = _skills_agent(tmp_path / "home")
    res = install([agent], [skill], dry_run=True)
    assert [r.action for r in res] == ["would-install"]
    assert not agent.target.exists()


# ---------------------------------------------------------------------------
# Install — instructions-file agents
# ---------------------------------------------------------------------------


def test_install_block_into_new_file(tmp_path: Path) -> None:
    skill = _make_skill(tmp_path / "src", "autocut-run")
    agent = _instructions_agent(tmp_path / "home")

    res = install([agent], [skill])
    assert [r.action for r in res] == ["installed"]
    text = agent.target.read_text(encoding="utf-8")
    assert "AUTOCUT:BEGIN" in text and "AUTOCUT:END" in text
    assert "autocut guidance" in text


def test_install_block_is_idempotent_and_preserves_other_content(tmp_path: Path) -> None:
    skill = _make_skill(tmp_path / "src", "autocut-run")
    agent = _instructions_agent(tmp_path / "home")
    agent.target.parent.mkdir(parents=True)
    agent.target.write_text("# My existing notes\nkeep me\n", encoding="utf-8")

    install([agent], [skill])
    after_first = agent.target.read_text(encoding="utf-8")
    assert "# My existing notes" in after_first  # user content preserved

    res = install([agent], [skill])
    assert [r.action for r in res] == ["unchanged"]
    # Re-running must not duplicate the block.
    assert agent.target.read_text(encoding="utf-8").count("AUTOCUT:BEGIN") == 1


def test_install_block_replaces_in_place_when_content_changes(tmp_path: Path) -> None:
    s1 = _make_skill(tmp_path / "src1", "autocut-run")
    agent = _instructions_agent(tmp_path / "home")
    install([agent], [s1])

    # A second skill changes the block's "Installed skills:" line -> update in place.
    s2 = _make_skill(tmp_path / "src2", "autocut-cut")
    res = install([agent], [s1, s2])
    assert [r.action for r in res] == ["updated"]
    assert agent.target.read_text(encoding="utf-8").count("AUTOCUT:BEGIN") == 1


def test_instructions_block_lists_skill_names(tmp_path: Path) -> None:
    skills = [
        _make_skill(tmp_path / "src", "autocut-run"),
        _make_skill(tmp_path / "src", "autocut-cut"),
    ]
    block = instructions_block(skills)
    assert "autocut-run" in block and "autocut-cut" in block


# ---------------------------------------------------------------------------
# Project install (any agent)
# ---------------------------------------------------------------------------


def test_project_install_writes_skills_and_agents_md(tmp_path: Path) -> None:
    proj = tmp_path / "repo"
    proj.mkdir()
    skill = _make_skill(tmp_path / "src", "autocut-run")

    res = install(project_agents(proj), [skill])
    actions = {r.agent_id: r.action for r in res}
    assert actions["project-skills"] == "installed"
    assert actions["project-agents-md"] == "installed"
    assert (proj / ".claude" / "skills" / "autocut-run" / "SKILL.md").is_file()
    assert "AUTOCUT:BEGIN" in (proj / "AGENTS.md").read_text(encoding="utf-8")
