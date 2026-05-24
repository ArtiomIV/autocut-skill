"""Tests for ``autocut.config`` — defaults, normalisation, layered loading."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from autocut.config import (
    AutoCutConfig,
    AutoCutSettings,
    OutputConfig,
    config_dir,
    config_path,
)


def test_defaults_are_internally_consistent() -> None:
    cfg = AutoCutConfig()
    assert cfg.vlm.provider == "openrouter"
    assert cfg.output.modes == ["separate"]
    assert cfg.scoring.min_score == 5


def test_all_mode_expands_to_separate_and_merged() -> None:
    cfg = AutoCutConfig.model_validate({"output": {"modes": ["all"]}})
    assert cfg.output.modes == ["separate", "merged"]


def test_duplicate_modes_are_collapsed() -> None:
    cfg = AutoCutConfig.model_validate({"output": {"modes": ["separate", "merged", "separate"]}})
    assert cfg.output.modes == ["separate", "merged"]


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(ValidationError):
        AutoCutConfig.model_validate({"vlm": {"provider": "anthropic"}})


def test_unknown_top_level_section_is_rejected() -> None:
    with pytest.raises(ValidationError):
        AutoCutConfig.model_validate({"telemetry": {"enabled": True}})


def test_score_bounds_are_enforced() -> None:
    with pytest.raises(ValidationError):
        AutoCutConfig.model_validate({"scoring": {"min_score": 11}})


def test_output_modes_default_factory_is_isolated() -> None:
    # Defensive check: mutating one instance's list must not bleed into another.
    a = OutputConfig()
    b = OutputConfig()
    a.modes.append("merged")
    assert b.modes == ["separate"]


def test_settings_load_falls_back_to_defaults_when_file_missing(tmp_path: Path) -> None:
    settings = AutoCutSettings.load(path=tmp_path / "no_such_config.toml")
    assert settings.config.vlm.provider == "openrouter"


def test_settings_load_reads_toml(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        """
[vlm]
provider = "host"
model = "claude-sonnet-4-6"

[scoring]
min_score = 7
""".strip(),
        encoding="utf-8",
    )
    settings = AutoCutSettings.load(path=cfg_file)
    assert settings.config.vlm.provider == "host"
    assert settings.config.vlm.model == "claude-sonnet-4-6"
    assert settings.config.scoring.min_score == 7


def test_env_var_override_takes_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text('[vlm]\nprovider = "openrouter"\n', encoding="utf-8")
    monkeypatch.setenv("AUTOCUT_CONFIG__VLM__PROVIDER", "host")
    settings = AutoCutSettings.load(path=cfg_file)
    assert settings.config.vlm.provider == "host"


def test_config_dir_respects_xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert config_dir() == tmp_path / "autocut"
    assert config_path() == tmp_path / "autocut" / "config.toml"
