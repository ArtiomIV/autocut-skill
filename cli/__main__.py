"""Typer-based command-line entrypoint for the ``autocut`` executable.

In v0.1.0-alpha (this milestone) every command is a stub that prints a clear
message about which milestone wires up the real behaviour. The CLI surface is
defined now so downstream milestones only need to fill the bodies.
"""

from __future__ import annotations

import getpass
import shutil
import subprocess
import sys
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from autocut import __version__
from autocut.config import AutoCutSettings, config_path
from autocut.security import (
    SERVICE_NAME,
    KeyringError,
    SupportedProvider,
    delete_key,
    get_key,
    install_redaction_filter,
    list_providers_with_key,
    set_key,
)

# Install the log redaction filter as early as possible so any subsequent
# library call that ends up logging an API-key-shaped string is scrubbed.
install_redaction_filter()

console = Console()
err_console = Console(stderr=True, style="bold red")

app = typer.Typer(
    name="autocut",
    help="Automatic video highlight extraction with a Vision Language Model.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)

keys_app = typer.Typer(name="keys", help="Manage API keys in the OS keyring.")
config_app = typer.Typer(name="config", help="Inspect and edit AutoCut configuration.")
models_app = typer.Typer(name="models", help="Discover models exposed by a VLM provider.")

app.add_typer(keys_app, name="keys")
app.add_typer(config_app, name="config")
app.add_typer(models_app, name="models")


# ---------------------------------------------------------------------------
# Top-level commands
# ---------------------------------------------------------------------------


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"autocut {__version__}")
        raise typer.Exit


@app.callback()
def _main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            help="Show version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """Root callback. Currently only used to wire up ``--version``."""


@app.command()
def doctor() -> None:
    """Diagnose the local environment: ffmpeg, keyring, config, providers."""
    table = Table(title="AutoCut Doctor", show_header=False, padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()

    table.add_row("autocut", __version__)
    table.add_row("python", sys.version.split()[0])
    table.add_row("ffmpeg", _probe_binary("ffmpeg"))
    table.add_row("ffprobe", _probe_binary("ffprobe"))
    table.add_row("config", str(config_path()))

    try:
        providers = list_providers_with_key()
        table.add_row(
            "keyring", f"service={SERVICE_NAME!r}, providers with key: {providers or 'none'}"
        )
    except KeyringError as exc:
        table.add_row("keyring", f"[red]error: {exc}[/]")

    console.print(table)


@app.command()
def run(
    video: Annotated[str, typer.Argument(help="Path to the input video file.")],
) -> None:
    """Run the analysis + cutting pipeline on a video. (Available from M3.)"""
    err_console.print(
        f"`autocut run {video}` is not implemented yet. The pipeline wires up in milestone M3."
    )
    raise typer.Exit(code=2)


@app.command()
def resume() -> None:
    """Resume a paused pipeline (used by the host-agent provider). Available from M3."""
    err_console.print("`autocut resume` is not implemented yet. Available from M3.")
    raise typer.Exit(code=2)


@app.command()
def bootstrap() -> None:
    """Detect installed AI agents and install skill manifests. Available from M5."""
    err_console.print("`autocut bootstrap` is not implemented yet. Available from M5.")
    raise typer.Exit(code=2)


# ---------------------------------------------------------------------------
# autocut keys ...
# ---------------------------------------------------------------------------


@keys_app.command("list")
def keys_list() -> None:
    """List providers that currently have an API key in the OS keyring."""
    try:
        providers = list_providers_with_key()
    except KeyringError as exc:
        err_console.print(f"keyring error: {exc}")
        raise typer.Exit(code=1) from exc
    if not providers:
        console.print("No API keys configured. Use `autocut keys set <provider>`.")
        return
    for p in providers:
        console.print(f"  [green]✓[/green] {p}")


@keys_app.command("set")
def keys_set(provider: SupportedProvider) -> None:
    """Store the API key for [bold]PROVIDER[/bold] in the OS keyring (hidden input)."""
    key = getpass.getpass(f"Paste API key for {provider} (hidden): ")
    try:
        set_key(provider, key)
    except (ValueError, KeyringError) as exc:
        err_console.print(str(exc))
        raise typer.Exit(code=1) from exc
    console.print(f"[green]✓[/green] Stored key for {provider} in OS keyring.")


@keys_app.command("delete")
def keys_delete(provider: SupportedProvider) -> None:
    """Remove the API key for [bold]PROVIDER[/bold] from the OS keyring."""
    try:
        removed = delete_key(provider)
    except KeyringError as exc:
        err_console.print(f"keyring error: {exc}")
        raise typer.Exit(code=1) from exc
    if removed:
        console.print(f"[green]✓[/green] Deleted key for {provider}.")
    else:
        console.print(f"No key was stored for {provider} (nothing to delete).")


@keys_app.command("test")
def keys_test(provider: SupportedProvider) -> None:
    """Verify a stored key by issuing a minimal request to the provider. (Available from M3.)"""
    key = get_key(provider)
    if key is None:
        err_console.print(f"No key set for {provider}.")
        raise typer.Exit(code=1)
    err_console.print(
        f"`autocut keys test {provider}` will issue a real ping to the provider in M3."
    )
    raise typer.Exit(code=2)


# ---------------------------------------------------------------------------
# autocut config ...
# ---------------------------------------------------------------------------


@config_app.callback(invoke_without_command=True)
def config_show(ctx: typer.Context) -> None:
    """Show the current configuration (default action)."""
    if ctx.invoked_subcommand is not None:
        return
    settings = AutoCutSettings.load()
    cfg = settings.config
    console.print_json(cfg.model_dump_json(indent=2))


@config_app.command("path")
def config_show_path() -> None:
    """Print the absolute path of the config TOML file."""
    console.print(str(config_path()))


@config_app.command("wizard")
def config_wizard() -> None:
    """Interactive configuration wizard. (Available from M5.)"""
    err_console.print("`autocut config wizard` is not implemented yet. Available from M5.")
    raise typer.Exit(code=2)


# ---------------------------------------------------------------------------
# autocut models ...
# ---------------------------------------------------------------------------


@models_app.command("list")
def models_list(
    vlm: Annotated[
        str,
        typer.Option("--vlm", help="VLM provider to query (e.g. openrouter)."),
    ] = "openrouter",
) -> None:
    """List VLM-capable models exposed by the chosen provider. (Available from M3.)"""
    err_console.print(
        f"`autocut models list --vlm {vlm}` is not implemented yet. Available from M3."
    )
    raise typer.Exit(code=2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _probe_binary(name: str) -> str:
    """Return a short version string for ``name`` or a clear 'missing' message."""
    path = shutil.which(name)
    if path is None:
        return "[red]not found in PATH[/]"
    try:
        result = subprocess.run(  # noqa: S603 - args is a fixed list, no shell
            [path, "-version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return f"[yellow]found at {path} but probe failed: {exc}[/]"
    first_line = result.stdout.splitlines()[0] if result.stdout else "(no output)"
    return first_line


if __name__ == "__main__":  # pragma: no cover
    app()
