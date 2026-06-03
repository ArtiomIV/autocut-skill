"""Layered configuration loader.

Precedence (highest wins): CLI flags > environment variables > user TOML file > defaults.

The user TOML lives at ``~/.config/autocut/config.toml`` (XDG-style on every OS,
including Windows). The file is created lazily by the config wizard; missing
file is not an error — we fall back to defaults.

API keys are NEVER stored here. They live only in the OS keyring (see
``autocut.security.keys``). The redaction filter in
``autocut.security.redaction`` makes sure they cannot accidentally land in
logs either.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Provider literal — v0.1.0 ships only ``host`` and ``openrouter``.
# Adding a new provider requires extending this Literal, the factory in
# ``autocut.vlm.factory``, and the wizard in ``cli.__main__``.
# ---------------------------------------------------------------------------

VLMProvider = Literal["host", "openrouter"]
OutputMode = Literal["separate", "merged", "all"]
MergeOrder = Literal["score", "chronological", "alternating"]


# ---------------------------------------------------------------------------
# Section models
# ---------------------------------------------------------------------------


class VLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: VLMProvider = "openrouter"
    model: str = "google/gemini-3.5-flash"
    fallback: VLMProvider | None = "openrouter"
    fallback_model: str | None = "google/gemini-3.5-flash"


def _default_output_modes() -> list[OutputMode]:
    return ["separate"]


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    modes: list[OutputMode] = Field(default_factory=_default_output_modes)
    base_dir: Path = Path("./CLIPS")
    merge_order: MergeOrder = "score"


class ScoringConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_score: int = Field(default=5, ge=0, le=10)
    max_clips: int = Field(default=20, ge=1, le=500)


class ContentDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hint: str = "auto"
    language: str = Field(default="it", pattern=r"^[a-z]{2}(-[A-Z]{2})?$")
    goal: str = "highlight reel"


class SecurityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cost_cap_usd: float = Field(default=1.0, ge=0)
    privacy_warn: bool = True
    # ``require_confirm_capcut`` is reserved for v0.2.0; intentionally not declared yet.


class AdvancedConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keyframe_resolution: int = Field(default=768, ge=128, le=2048)
    scene_threshold: float = Field(default=27.0, gt=0)
    keyframes_per_scene: int = Field(default=2, ge=1, le=10)


# ---------------------------------------------------------------------------
# Root config — loaded by AutoCutSettings below
# ---------------------------------------------------------------------------


class AutoCutConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vlm: VLMConfig = Field(default_factory=VLMConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    content_defaults: ContentDefaults = Field(default_factory=ContentDefaults)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    advanced: AdvancedConfig = Field(default_factory=AdvancedConfig)

    @model_validator(mode="after")
    def _normalise_output_modes(self) -> AutoCutConfig:
        # ``all`` is sugar for the union of ``separate`` and ``merged``.
        if "all" in self.output.modes:
            self.output.modes = ["separate", "merged"]
        # De-duplicate while preserving order.
        seen: set[str] = set()
        unique: list[OutputMode] = []
        for m in self.output.modes:
            if m not in seen:
                unique.append(m)
                seen.add(m)
        self.output.modes = unique
        return self


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def config_dir() -> Path:
    """Return the AutoCut config directory, respecting XDG on every OS."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "autocut"


def config_path() -> Path:
    return config_dir() / "config.toml"


# ---------------------------------------------------------------------------
# Settings loader
# ---------------------------------------------------------------------------


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


_ENV_PREFIX = "AUTOCUT_CONFIG__"


def _env_overrides() -> dict[str, Any]:
    """Collect ``AUTOCUT_CONFIG__<section>__<field>`` env vars into a nested dict."""
    result: dict[str, Any] = {}
    for key, value in os.environ.items():
        if not key.startswith(_ENV_PREFIX):
            continue
        parts = key[len(_ENV_PREFIX) :].lower().split("__")
        if not parts or not all(parts):
            continue
        cursor: dict[str, Any] = result
        for segment in parts[:-1]:
            next_node = cursor.setdefault(segment, {})
            if not isinstance(next_node, dict):
                # A scalar already lives here; env layout is ambiguous, skip.
                break
            cursor = next_node
        else:
            cursor[parts[-1]] = value
    return result


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Return ``base`` with ``overlay`` deep-merged on top (overlay wins)."""
    out: dict[str, Any] = dict(base)
    for k, v in overlay.items():
        existing = out.get(k)
        if isinstance(v, dict) and isinstance(existing, dict):
            out[k] = _deep_merge(existing, v)
        else:
            out[k] = v
    return out


class AutoCutSettings(BaseModel):
    """Top-level wrapper. Layered loading: defaults < TOML < env vars.

    Environment variables use the ``AUTOCUT_CONFIG__`` prefix with ``__`` as
    nested delimiter. Example: ``AUTOCUT_CONFIG__VLM__PROVIDER=host`` overrides
    ``[vlm].provider``.
    """

    model_config = ConfigDict(extra="forbid")

    config: AutoCutConfig = Field(default_factory=AutoCutConfig)

    @classmethod
    def load(cls, *, path: Path | None = None) -> AutoCutSettings:
        """Load config: defaults < TOML file < env vars (highest wins)."""
        toml_path = path or config_path()
        toml_data = _load_toml(toml_path)
        merged = _deep_merge(toml_data, _env_overrides())
        config_obj = AutoCutConfig.model_validate(merged) if merged else AutoCutConfig()
        return cls(config=config_obj)
