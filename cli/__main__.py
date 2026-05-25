"""Typer-based command-line entrypoint for the ``autocut`` executable."""

from __future__ import annotations

import asyncio
import contextlib
import getpass
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from autocut import __version__
from autocut.config import AutoCutSettings, OutputMode, config_path
from autocut.models import ClipPlan, VideoMetadata
from autocut.pipeline import (
    AnalysisResult,
    CostCapExceeded,
    ResumeStateError,
    complete_from_plan,
    load_resume_state,
    run_analysis,
)
from autocut.scoring import RankedClip
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
from autocut.vlm import (
    CostEstimate,
    HostAgentPauseRequested,
    HostAgentProvider,
    UnavailableProviderError,
    VLMError,
    list_openrouter_models,
    make_provider,
)

# Force UTF-8 on stdio so unicode chars (✓, →, …) used in rich output don't
# crash on Windows consoles defaulting to cp1252. Must run before any
# console.print() call.
for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if reconfigure is not None:
        with contextlib.suppress(OSError, ValueError):
            reconfigure(encoding="utf-8", errors="replace")

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
    video: Annotated[Path, typer.Argument(help="Path to the input video file.")],
    vlm: Annotated[
        str | None,
        typer.Option("--vlm", help="VLM provider override (host | openrouter)."),
    ] = None,
    vlm_model: Annotated[
        str | None,
        typer.Option("--vlm-model", help="VLM model id override (provider-specific)."),
    ] = None,
    sampling: Annotated[
        str,
        typer.Option(
            "--sampling",
            help="Keyframe sampling strategy: scene | uniform | hybrid.",
        ),
    ] = "hybrid",
    output: Annotated[
        str | None,
        typer.Option(
            "--output",
            help="Comma-separated output modes: separate | merged | all. "
            "Defaults to the config (typically 'separate').",
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", help="Override output base dir (default: ./CLIPS)."),
    ] = None,
    accurate: Annotated[
        bool,
        typer.Option(
            "--accurate/--fast",
            help="Re-encode for frame-accurate cuts (slower) vs stream-copy.",
        ),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Skip confirmation prompts (e.g. cost cap exceeded).",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Run the analysis but do not write any output files.",
        ),
    ] = False,
) -> None:
    """Run the analysis pipeline on a video and produce the highlight clips."""
    if not video.exists() or not video.is_file():
        err_console.print(f"input video not found: {video}")
        raise typer.Exit(code=1)

    settings = AutoCutSettings.load()
    cfg = settings.config
    if output is not None:
        cfg.output.modes = _parse_output_modes(output)

    provider_name = vlm or cfg.vlm.provider
    model = vlm_model or cfg.vlm.model
    out_root = (output_dir or cfg.output.base_dir).resolve()
    work_dir = out_root  # host-agent provider writes request/response files here

    try:
        provider = make_provider(
            provider_name,
            model=model,
            work_dir=work_dir,
        )
    except UnavailableProviderError as exc:
        err_console.print(str(exc))
        raise typer.Exit(code=2) from exc
    except VLMError as exc:
        err_console.print(str(exc))
        raise typer.Exit(code=1) from exc

    try:
        result = asyncio.run(
            run_analysis(
                video,
                provider,
                config=cfg,
                output_root=out_root,
                sampling_strategy=sampling,
                write_outputs=not dry_run,
                accurate_cuts=accurate,
                confirm_cost=_make_cost_confirm(yes=yes),
            )
        )
    except HostAgentPauseRequested as pause:
        _print_pause_message(pause)
        raise typer.Exit(code=0) from None
    except CostCapExceeded as exc:
        err_console.print(str(exc))
        raise typer.Exit(code=1) from exc
    except VLMError as exc:
        err_console.print(f"VLM error: {exc}")
        raise typer.Exit(code=1) from exc

    _render_analysis_summary(result)


@app.command()
def resume(
    work_dir: Annotated[
        Path | None,
        typer.Option(
            "--work-dir",
            help="Directory containing VLM_REQUEST.md / VLM_RESPONSE.json (default: ./CLIPS).",
        ),
    ] = None,
) -> None:
    """Resume a paused host-agent run: validate the JSON, rank, write clips."""
    settings = AutoCutSettings.load()
    cfg = settings.config
    base = (work_dir or cfg.output.base_dir).resolve()

    try:
        state = load_resume_state(base)
    except ResumeStateError as exc:
        err_console.print(str(exc))
        raise typer.Exit(code=1) from exc

    provider = HostAgentProvider(work_dir=base, agent_hint=cfg.vlm.model)
    try:
        plan = provider.resume_from_disk()
    except VLMError as exc:
        err_console.print(str(exc))
        raise typer.Exit(code=1) from exc

    console.print(f"[green]OK[/] Loaded ClipPlan with {len(plan.clips)} clip(s).")

    video = Path(str(state["video_path"]))
    if not video.is_file():
        err_console.print(
            f"original video no longer at {video}; cannot cut clips. "
            f"Move the file back or re-run `autocut run` from scratch."
        )
        raise typer.Exit(code=1)

    try:
        metadata = VideoMetadata.model_validate(state["video_metadata"])
    except ValidationError as exc:
        err_console.print(f"resume state video_metadata is invalid: {exc}")
        raise typer.Exit(code=1) from exc

    result = complete_from_plan(
        plan,
        video=video,
        metadata=metadata,
        config=cfg,
        output_root=base,
        n_scenes=_state_int(state, "n_scenes"),
        n_keyframes=_state_int(state, "n_keyframes"),
        sampling_strategy=str(state.get("sampling_strategy", "hybrid")),
        accurate_cuts=bool(state.get("accurate_cuts", False)),
        write_outputs=bool(state.get("write_outputs", True)),
    )
    # Resume succeeded end-to-end; the sidecar has served its purpose.
    with contextlib.suppress(OSError):
        (base / ".autocut_resume.json").unlink(missing_ok=True)

    _render_analysis_summary(result)


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
    """Verify a stored key by issuing a minimal health check against the provider."""
    key = get_key(provider)
    if key is None:
        err_console.print(f"No key set for {provider}.")
        raise typer.Exit(code=1)
    try:
        vlm = make_provider(provider, model="placeholder", api_key=key)
    except VLMError as exc:
        err_console.print(str(exc))
        raise typer.Exit(code=1) from exc
    ok = vlm.health_check()
    if ok:
        console.print(f"[green]✓[/green] {provider} reachable with stored key.")
    else:
        err_console.print(f"{provider} health check failed; key may be invalid.")
        raise typer.Exit(code=1)


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
    limit: Annotated[
        int, typer.Option("--limit", help="Max rows to display (sorted by cheapest input).")
    ] = 20,
) -> None:
    """List VLM-capable models exposed by the chosen provider."""
    if vlm != "openrouter":
        err_console.print(f"--vlm {vlm} is not supported in v0.1.0. Use --vlm openrouter.")
        raise typer.Exit(code=2)

    api_key: str | None = None
    with contextlib.suppress(KeyringError):
        api_key = get_key("openrouter")

    try:
        models = asyncio.run(list_openrouter_models(api_key=api_key))
    except VLMError as exc:
        err_console.print(str(exc))
        raise typer.Exit(code=1) from exc

    table = Table(title=f"OpenRouter VLM models (top {limit} cheapest)")
    table.add_column("Model id", style="bold")
    table.add_column("Context", justify="right")
    table.add_column("Input $/1M", justify="right")
    table.add_column("Output $/1M", justify="right")

    for m in models[:limit]:
        ctx = _format_context(m.context_length)
        in_p = _format_price(m.usd_per_1m_input)
        out_p = _format_price(m.usd_per_1m_output)
        table.add_row(m.id, ctx, in_p, out_p)

    console.print(table)
    if len(models) > limit:
        console.print(f"[dim]({len(models) - limit} more — use --limit to see them)[/]")


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


def _format_context(value: int | None) -> str:
    if value is None:
        return "—"
    if value >= 1_000_000:
        return f"{value // 1000:>3d}K"
    return f"{value // 1000}K" if value >= 1000 else str(value)


def _format_price(value: float | None) -> str:
    return "—" if value is None else f"${value:.4f}"


def _print_pause_message(pause: HostAgentPauseRequested) -> None:
    console.print(
        "\n[bold yellow]Pipeline paused (host-agent provider).[/]\n"
        f"  request:  [cyan]{pause.request_path}[/]\n"
        f"  response: [cyan]{pause.response_path}[/]\n"
        "Read the request, write the ClipPlan JSON to the response path,\n"
        "then run [bold]autocut resume[/bold] to continue.\n"
    )


def _state_int(state: dict[str, object], key: str) -> int:
    """Read an int counter from the resume-state dict, defaulting to 0."""
    value = state.get(key, 0)
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def _parse_output_modes(raw: str) -> list[OutputMode]:
    """Split a comma-separated string into validated ``OutputMode`` values."""
    valid: set[str] = {"separate", "merged", "all"}
    modes: list[OutputMode] = []
    for token in raw.split(","):
        cleaned = token.strip().lower()
        if cleaned not in valid:
            raise typer.BadParameter(
                f"unknown output mode {cleaned!r}; choose from {sorted(valid)}"
            )
        # ``cast`` would also work; the literal-membership check above makes
        # this assignment type-safe for mypy.
        modes.append(cleaned)  # type: ignore[arg-type]
    return modes


def _make_cost_confirm(*, yes: bool):  # type: ignore[no-untyped-def]
    """Return a confirm callback consumed by the pipeline."""

    def _confirm(estimate: CostEstimate, cap_usd: float) -> bool:
        if yes:
            return True
        console.print(
            f"\n[bold yellow]Cost estimate ${estimate.estimated_total_usd:.4f}[/] "
            f"exceeds cap ${cap_usd:.2f} for "
            f"{estimate.provider}/{estimate.model} on {estimate.n_input_images} image(s)."
        )
        return typer.confirm("Proceed anyway?", default=False)

    return _confirm


def _render_analysis_summary(result: AnalysisResult) -> None:
    meta = result.metadata
    plan = result.plan
    console.print(
        f"\n[bold]Video[/]: {meta.path} — "
        f"{meta.width}x{meta.height} @ {meta.fps:.1f}fps, {meta.duration_sec:.1f}s\n"
        f"[bold]Keyframes extracted[/]: {len(result.keyframes)}\n"
        f"[bold]VLM[/]: {plan.metadata.vlm_provider} / {plan.metadata.vlm_model} "
        f"(prompt {plan.metadata.prompt_version})\n"
    )

    if not plan.clips:
        console.print("[yellow]No clips were proposed.[/]")
        return

    table = Table(title=f"{len(plan.clips)} clip candidate(s) ranked")
    table.add_column("#", justify="right")
    table.add_column("Start → End", style="bold")
    table.add_column("VLM", justify="right")
    table.add_column("Heur", justify="right")
    table.add_column("Final", justify="right", style="bold")
    table.add_column("Category")
    table.add_column("Description")

    rows_source = result.ranked if result.ranked else _fallback_rows(plan)
    for i, ranked in enumerate(rows_source, start=1):
        table.add_row(
            str(i),
            f"{ranked.clip.start} → {ranked.clip.end}",
            str(ranked.vlm_score),
            str(ranked.heuristic_score),
            str(ranked.final_score),
            ranked.clip.category.value,
            ranked.clip.description,
        )
    console.print(table)

    if result.dispatch:
        console.print("\n[bold]Outputs written[/]:")
        for mode, written in result.dispatch.by_mode.items():
            console.print(f"  [cyan]{mode}[/]: {len(written)} file(s)")
            for w in written:
                console.print(f"    - {w.path}")
        if result.dispatch.manifest_path:
            console.print(f"\n[bold]Manifest[/]: {result.dispatch.manifest_path}")
    else:
        console.print(
            "\n[dim]Dry run — no files were written. Re-run without --dry-run to produce clips.[/]"
        )

    console.print(
        f"\n[dim]Full ClipPlan JSON:[/]\n{json.dumps(plan.model_dump(mode='json'), indent=2)}"
    )


def _fallback_rows(plan: ClipPlan) -> list[RankedClip]:
    """Build display rows from raw ``ClipPlan`` when ranking dropped everything."""
    return [
        RankedClip(
            clip=clip,
            vlm_score=clip.score,
            heuristic_score=0,
            final_score=clip.score,
        )
        for clip in plan.clips
    ]


if __name__ == "__main__":  # pragma: no cover
    app()
