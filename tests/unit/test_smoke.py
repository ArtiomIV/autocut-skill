"""Smoke tests: importing the package and CLI must not blow up."""

from __future__ import annotations


def test_package_imports() -> None:
    import autocut

    assert autocut.__version__


def test_cli_imports() -> None:
    from cli import app

    assert app is not None


def test_cli_help_runs() -> None:
    from typer.testing import CliRunner

    from cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "autocut" in result.output.lower()


def test_cli_version() -> None:
    from typer.testing import CliRunner

    from autocut import __version__
    from cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output
