"""Unit tests for the CLI ``_build_hints_from_cli`` helper.

The ``run`` subcommand turns ``--content-hint`` / ``--query`` into an
``AnalysisHints`` override (or ``None`` to let the pipeline resolve its own
defaults). These tests pin the query plumbing: a bare ``--query`` must still
build hints (mode falls back to ``auto``), whitespace is normalised, and an
unknown mode fails loud.
"""

from __future__ import annotations

import pytest
import typer

from autocut.config import AutoCutConfig
from autocut.models import ContentHint
from cli.__main__ import _build_hints_from_cli


def test_no_overrides_returns_none() -> None:
    assert _build_hints_from_cli(None, None, AutoCutConfig()) is None


def test_bare_query_builds_hints_with_auto_mode() -> None:
    hints = _build_hints_from_cli(None, "the knockdown in round 3", AutoCutConfig())
    assert hints is not None
    assert hints.content_hint is ContentHint.auto
    assert hints.query == "the knockdown in round 3"


def test_mode_and_query_combine() -> None:
    hints = _build_hints_from_cli("highlights", "when they mention prices", AutoCutConfig())
    assert hints is not None
    assert hints.content_hint is ContentHint.highlights
    assert hints.query == "when they mention prices"


def test_mode_only_leaves_query_none() -> None:
    hints = _build_hints_from_cli("talk", None, AutoCutConfig())
    assert hints is not None
    assert hints.content_hint is ContentHint.talk
    assert hints.query is None


def test_whitespace_query_is_normalised_to_none() -> None:
    # A whitespace-only query with no mode is treated as "no override".
    assert _build_hints_from_cli(None, "   ", AutoCutConfig()) is None
    # With a mode, the blank query is dropped but the mode hints still build.
    hints = _build_hints_from_cli("hybrid", "  ", AutoCutConfig())
    assert hints is not None
    assert hints.query is None


def test_unknown_mode_raises_bad_parameter() -> None:
    with pytest.raises(typer.BadParameter, match="unknown --content-hint"):
        _build_hints_from_cli("boxing", None, AutoCutConfig())
