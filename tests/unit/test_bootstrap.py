"""Unit tests for ``autocut.bootstrap`` — agent detection + skill install.

Detection and install are exercised against a fake home and synthetic skill
directories in ``tmp_path`` (no real ``~/.claude`` mutation). The bundled-skills
accessor is checked against the repo's real ``.claude/skills``.
"""

from __future__ import annotations

from pathlib import Path

from autocut.bootstrap import (
    Agent,
    Skill,
    detect_agents,
    discover_skills,
    install_skills,
    known_agents,
)


def _make_skill(root: Path, name: str, body: str = "# skill\n") -> Skill:
    d = root / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(body, encoding="utf-8")
    return Skill(name=name, path=d)


def _agent_for(home: Path) -> Agent:
    return Agent(
        agent_id="claude-code",
        label="Claude Code",
        marker=home / ".claude",
        skills_dir=home / ".claude" / "skills",
    )


# ---------------------------------------------------------------------------
# Bundled skills (real repo manifests)
# ---------------------------------------------------------------------------


def test_discover_skills_finds_the_repo_manifests() -> None:
    skills = discover_skills()
    names = {s.name for s in skills}
    # The four shipped skills must all be present and each have a SKILL.md.
    assert {"autocut-run", "autocut-cut", "autocut-merge", "autocut-detect"} <= names
    assert all((s.path / "SKILL.md").is_file() for s in skills)


# ---------------------------------------------------------------------------
# Agent detection
# ---------------------------------------------------------------------------


def test_detect_agents_present(tmp_path: Path) -> None:
    (tmp_path / ".claude").mkdir()
    detected = detect_agents(home=tmp_path)
    assert [a.agent_id for a in detected] == ["claude-code"]


def test_detect_agents_absent(tmp_path: Path) -> None:
    assert detect_agents(home=tmp_path) == []


def test_known_agents_targets_claude_skills_dir(tmp_path: Path) -> None:
    (agent,) = known_agents(home=tmp_path)
    assert agent.skills_dir == tmp_path / ".claude" / "skills"


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------


def test_install_then_unchanged_is_idempotent(tmp_path: Path) -> None:
    src = tmp_path / "src"
    skill = _make_skill(src, "autocut-run", "# run\n")
    agent = _agent_for(tmp_path / "home")

    first = install_skills([agent], [skill])
    assert [r.action for r in first] == ["installed"]
    assert (agent.skills_dir / "autocut-run" / "SKILL.md").read_text(encoding="utf-8") == "# run\n"

    second = install_skills([agent], [skill])
    assert [r.action for r in second] == ["unchanged"]


def test_install_updates_when_source_changes(tmp_path: Path) -> None:
    src = tmp_path / "src"
    skill = _make_skill(src, "autocut-run", "# v1\n")
    agent = _agent_for(tmp_path / "home")
    install_skills([agent], [skill])

    (skill.path / "SKILL.md").write_text("# v2\n", encoding="utf-8")
    res = install_skills([agent], [skill])
    assert [r.action for r in res] == ["updated"]
    assert (agent.skills_dir / "autocut-run" / "SKILL.md").read_text(encoding="utf-8") == "# v2\n"


def test_force_reinstalls_identical_skill(tmp_path: Path) -> None:
    src = tmp_path / "src"
    skill = _make_skill(src, "autocut-run")
    agent = _agent_for(tmp_path / "home")
    install_skills([agent], [skill])

    res = install_skills([agent], [skill], force=True)
    assert [r.action for r in res] == ["updated"]


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    src = tmp_path / "src"
    skill = _make_skill(src, "autocut-run")
    agent = _agent_for(tmp_path / "home")

    res = install_skills([agent], [skill], dry_run=True)
    assert [r.action for r in res] == ["would-install"]
    assert not agent.skills_dir.exists()


def test_install_removes_stale_files_on_update(tmp_path: Path) -> None:
    src = tmp_path / "src"
    skill = _make_skill(src, "autocut-run")
    (skill.path / "extra.md").write_text("old\n", encoding="utf-8")
    agent = _agent_for(tmp_path / "home")
    install_skills([agent], [skill])
    assert (agent.skills_dir / "autocut-run" / "extra.md").exists()

    # Drop the extra file from the source; an update must mirror the removal.
    (skill.path / "extra.md").unlink()
    res = install_skills([agent], [skill])
    assert [r.action for r in res] == ["updated"]
    assert not (agent.skills_dir / "autocut-run" / "extra.md").exists()
