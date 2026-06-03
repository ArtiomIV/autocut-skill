"""Typer-based command-line entrypoint for the ``autocut`` executable."""

from __future__ import annotations

import asyncio
import contextlib
import getpass
import json
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from autocut import __version__
from autocut.config import AutoCutConfig, AutoCutSettings, config_path
from autocut.models import AnalysisHints, ContentHint
from autocut.pipeline import (
    AnalysisResult,
    CostCapExceeded,
    ResumeStateError,
    resume_host_analysis,
    run_analysis,
)
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
    UnavailableProviderError,
    VLMError,
    list_openrouter_models,
    make_provider,
    validate_openrouter_model,
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
            help=(
                "Keyframe sampling strategy: scene | uniform | hybrid | motion. "
                "'motion' runs optical flow + audio onset analysis to sample "
                "densely only where the action is (Phase D)."
            ),
        ),
    ] = "hybrid",
    content_hint: Annotated[
        str | None,
        typer.Option(
            "--content-hint",
            help=(
                "Editing MODE (auto | highlights | hybrid | talk). 'highlights' "
                "auto-selects the best/viral moments (strict, may return nothing); "
                "'talk' is for speech-driven content; 'hybrid' is the conservative "
                "default. Unset/auto -> hybrid. The orchestrating agent picks this."
            ),
        ),
    ] = None,
    query: Annotated[
        str | None,
        typer.Option(
            "--query",
            help=(
                "Find a SPECIFIC described moment instead of auto-highlights "
                '(e.g. "the knockdown in round 3", "when they mention prices"). '
                "Disables the motion pre-filter so low-motion targets are not "
                "sampled away — the model filters via the query instead."
            ),
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
    two_pass: Annotated[
        bool,
        typer.Option(
            "--two-pass/--single-pass",
            help=(
                "Default ON for long (>60s) openrouter video/audio runs: locate "
                "candidate regions coarsely first, then re-analyse each in isolation "
                "for tighter, more accurate cut boundaries (validated to start KO "
                "clips on the punch, not the referee count). Costs ~2x the calls; "
                "pass --single-pass to disable. No effect on short videos or the "
                "host/keyframe routes."
            ),
        ),
    ] = True,
    force_two_pass: Annotated[
        bool,
        typer.Option(
            "--force-two-pass",
            help=(
                "Run the two-pass coarse→fine refinement even on short (<=60s) "
                "openrouter video/audio sources, lifting the >60s gate. Implies "
                "--two-pass. Useful to get tight, frame-accurate boundaries on a "
                "short clip on demand. No effect on host/keyframe routes."
            ),
        ),
    ] = False,
) -> None:
    """Run the analysis pipeline on a video and produce the highlight clips."""
    if not video.exists() or not video.is_file():
        err_console.print(f"input video not found: {video}")
        raise typer.Exit(code=1)

    settings = AutoCutSettings.load()
    cfg = settings.config
    # ``run`` always produces per-clip separate outputs. Composing a single
    # reel is the job of the deterministic ``autocut merge`` subcommand, so the
    # pipeline never mutates the output mode at runtime — that keeps the
    # resume sidecar trivially consistent (no output mode to persist).
    cfg.output.modes = ["separate"]

    hints_override = _build_hints_from_cli(content_hint, query, cfg)

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

    _enforce_model_capabilities(provider_name, model)

    try:
        result = asyncio.run(
            run_analysis(
                video,
                provider,
                config=cfg,
                hints=hints_override,
                output_root=out_root,
                sampling_strategy=sampling,
                write_outputs=not dry_run,
                accurate_cuts=accurate,
                confirm_cost=_make_cost_confirm(yes=yes),
                two_pass=two_pass or force_two_pass,
                force_two_pass=force_two_pass,
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
            help="Directory containing DETECTION_RESPONSE.json / VLM_RESPONSE.json (default: ./CLIPS).",
        ),
    ] = None,
) -> None:
    """Resume a paused host-agent run (the image-only two-pass).

    The host pauses once or twice: ``run`` writes ``VLM_REQUEST.md`` +
    contact sheets + a ``.autocut_resume.json`` sidecar recording the phase.
    On a COARSE resume this reads the candidate regions, builds the dense FINE
    sheets and pauses AGAIN (run ``autocut resume`` once more). On a FINE resume
    it parses the absolute-time clips, dedups, ranks, and writes ``plan.json``
    (then cut with ``autocut cut --from-json``).
    """
    settings = AutoCutSettings.load()
    cfg = settings.config
    base = (work_dir or cfg.output.base_dir).resolve()

    _resume_analysis_phase(base, cfg)


def _resume_analysis_phase(base: Path, cfg: AutoCutConfig) -> None:
    """Drive one host image two-pass resume step (coarse→fine pause, or finish)."""
    try:
        result = resume_host_analysis(base, cfg)
    except HostAgentPauseRequested as pause:
        # Coarse resume built the fine sheets and paused again for the fine pass.
        console.print("[green]OK[/] Coarse candidates loaded; built the fine contact sheets.")
        _print_pause_message(pause)
        raise typer.Exit(code=0) from None
    except ResumeStateError as exc:
        err_console.print(str(exc))
        raise typer.Exit(code=1) from exc
    except VLMError as exc:
        err_console.print(str(exc))
        raise typer.Exit(code=1) from exc

    _render_analysis_summary(result)


@app.command()
def detect(
    video: Annotated[Path, typer.Argument(help="Path to the input video file.")],
    vlm: Annotated[
        str | None,
        typer.Option("--vlm", help="VLM provider override (host | openrouter)."),
    ] = None,
    vlm_model: Annotated[
        str | None,
        typer.Option("--vlm-model", help="VLM model id override (provider-specific)."),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            help=(
                "Where to drop detection keyframes (default: ./CLIPS). "
                "JPEGs are reused by ``autocut run`` if you launch the full "
                "pipeline next."
            ),
        ),
    ] = None,
    n_keyframes: Annotated[
        int,
        typer.Option(
            "--keyframes",
            min=3,
            max=20,
            help="Number of keyframes to sample (stratified random). Default 9.",
        ),
    ] = 9,
) -> None:
    """Classify the content type of VIDEO without running the full pipeline.

    Prints a JSON object ``{content_hint, confidence, reasoning}``.
    Useful before launching ``autocut run`` to confirm the auto-detected
    category, or for AI agents that want to surface the classification
    to a user before triggering the expensive highlight extraction.
    """
    from autocut.content import detect_content_hint
    from autocut.video import probe_video

    if not video.exists() or not video.is_file():
        err_console.print(f"input video not found: {video}")
        raise typer.Exit(code=1)

    settings = AutoCutSettings.load()
    cfg = settings.config
    provider_name = vlm or cfg.vlm.provider
    model = vlm_model or cfg.vlm.model
    out_root = (output_dir or cfg.output.base_dir).resolve()

    try:
        provider = make_provider(provider_name, model=model, work_dir=out_root)
    except UnavailableProviderError as exc:
        err_console.print(str(exc))
        raise typer.Exit(code=2) from exc
    except VLMError as exc:
        err_console.print(str(exc))
        raise typer.Exit(code=1) from exc

    _enforce_model_capabilities(provider_name, model)

    try:
        metadata = probe_video(video)
        result = asyncio.run(
            detect_content_hint(
                video,
                provider,
                metadata=metadata,
                output_root=out_root,
                n_keyframes=n_keyframes,
            )
        )
    except HostAgentPauseRequested as exc:
        # Standalone detection is synchronous (openrouter). The host agent
        # is itself the orchestrator and should classify the video directly
        # instead of going through a paused two-phase detection cycle.
        err_console.print(
            "autocut detect needs a synchronous (cloud) provider, e.g. "
            "--vlm openrouter. The host agent should pick the editing mode "
            "directly and pass it to `autocut run --content-hint <mode>`."
        )
        raise typer.Exit(code=2) from exc
    except VLMError as exc:
        err_console.print(f"detect failed: {exc}")
        raise typer.Exit(code=1) from exc

    console.print_json(
        json.dumps(
            {
                "content_hint": result.content_hint.value,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
            }
        )
    )


@app.command()
def cut(
    video: Annotated[
        Path | None,
        typer.Argument(
            help="Input video. Optional with --from-json (taken from the plan, "
            "or overridden here / via --video)."
        ),
    ] = None,
    video_opt: Annotated[
        Path | None,
        typer.Option(
            "--video",
            help="Input video (same as the positional VIDEO; from-json override or "
            "single-segment input).",
        ),
    ] = None,
    start: Annotated[
        str | None,
        typer.Option(
            "--start",
            "-s",
            help="Start timestamp (HH:MM:SS.mmm or seconds). Single-segment mode.",
        ),
    ] = None,
    end: Annotated[
        str | None,
        typer.Option(
            "--end",
            "-e",
            help="End timestamp (HH:MM:SS.mmm or seconds, must be > start). Single-segment mode.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output MP4 path (single-segment mode). Parent dir created if missing.",
        ),
    ] = None,
    from_json: Annotated[
        Path | None,
        typer.Option(
            "--from-json",
            help="Cut EVERY clip listed in this plan.json (produced by `autocut run`). "
            "Timestamps already include any pre/post-roll, so the cut is 1:1.",
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            help="Destination directory for --from-json clips (writes ./separate/*.mp4 + manifest.json).",
        ),
    ] = None,
    min_score: Annotated[
        int,
        typer.Option(
            "--min-score",
            help="With --from-json, cut only clips whose final_score >= this (0 = all).",
        ),
    ] = 0,
    accurate: Annotated[
        bool,
        typer.Option(
            "--accurate/--fast",
            help=(
                "Frame-accurate cut (re-encode with libx264) vs stream-copy. "
                "Stream-copy is default — fast and lossless but snaps to the "
                "nearest keyframe (typically ±1-2s)."
            ),
        ),
    ] = False,
) -> None:
    """Cut clips with ffmpeg. Deterministic, no VLM.

    Two modes:
    - **--from-json PLAN --output-dir DIR**: cut every clip in a `plan.json` (from
      `autocut run`), filtered by --min-score. The plan's timestamps already
      include the pre/post-roll, so this trims 1:1. Handles 0..N clips.
    - **single segment**: VIDEO --start --end --output — cut one `[start, end]`.
    """
    video = video or video_opt  # accept the video as a positional or as --video

    if from_json is not None:
        _cut_from_json(
            from_json,
            video_override=video,
            output_dir=output_dir,
            min_score=min_score,
            accurate=accurate,
        )
        return

    # Single-segment mode.
    from autocut.video import CutRequest, CutterError, cut_clip

    if video is None or start is None or end is None or output is None:
        raise typer.BadParameter(
            "single-segment cut needs VIDEO --start --end --output (or use --from-json)"
        )
    if not video.exists() or not video.is_file():
        err_console.print(f"input video not found: {video}")
        raise typer.Exit(code=1)

    start_td = _parse_cli_timestamp(start, field="--start")
    end_td = _parse_cli_timestamp(end, field="--end")
    if end_td <= start_td:
        raise typer.BadParameter("--end must be strictly greater than --start")

    request = CutRequest(start=start_td, end=end_td, output_path=output.resolve())
    try:
        produced = cut_clip(video, request, accurate=accurate)
    except CutterError as exc:
        err_console.print(f"cut failed: {exc}")
        raise typer.Exit(code=1) from exc

    duration_sec = (end_td - start_td).total_seconds()
    console.print(
        f"[green]OK[/] cut {duration_sec:.3f}s "
        f"({'accurate' if accurate else 'stream-copy'}) → {produced}"
    )


def _cut_from_json(
    plan_path: Path,
    *,
    video_override: Path | None,
    output_dir: Path | None,
    min_score: int,
    accurate: bool,
) -> None:
    """Cut every clip in a plan.json into ``output_dir`` (reuses the separate writer)."""
    from autocut.output import PlanReadError, dispatch_outputs, read_plan_json

    if output_dir is None:
        raise typer.BadParameter("--from-json requires --output-dir")
    try:
        plan_video, metadata, ranked = read_plan_json(plan_path, min_score=min_score)
    except PlanReadError as exc:
        err_console.print(str(exc))
        raise typer.Exit(code=1) from exc

    video = (video_override or plan_video).resolve()
    if not video.is_file():
        err_console.print(f"input video not found: {video} (override with the VIDEO argument)")
        raise typer.Exit(code=1)
    if not ranked:
        console.print(f"[yellow]no clips with final_score >= {min_score}; nothing to cut.[/]")
        return

    result = dispatch_outputs(
        video,
        ranked,
        metadata,
        output_dir.resolve(),
        modes=["separate"],
        accurate=accurate,
        pre_roll_sec=0.0,  # the roll is already baked into the plan's timestamps
        post_roll_sec=0.0,
    )
    n_written = sum(len(w) for w in result.by_mode.values())
    console.print(f"[green]OK[/] cut {n_written} clip(s) → {output_dir.resolve()}")
    if result.manifest_path:
        console.print(f"[bold]Manifest[/]: {result.manifest_path}")


@app.command()
def merge(
    inputs: Annotated[
        list[Path] | None,
        typer.Argument(
            help=(
                "MP4 files to concatenate in the given order. "
                "Mutually exclusive with --from-manifest."
            ),
        ),
    ] = None,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output MP4 path. Parent directory is created if missing.",
        ),
    ] = Path("highlights.mp4"),
    from_manifest: Annotated[
        Path | None,
        typer.Option(
            "--from-manifest",
            help=(
                "Read clips from an autocut manifest.json instead of positional args. "
                "Pair with --min-score to filter."
            ),
        ),
    ] = None,
    min_score: Annotated[
        int,
        typer.Option(
            "--min-score",
            help="When using --from-manifest: keep only clips with final_score >= this.",
            min=0,
            max=10,
        ),
    ] = 0,
    order: Annotated[
        str,
        typer.Option(
            "--order",
            help=(
                "Output order: 'chronological' (clip.start ascending) | "
                "'score-desc' (highest score first) | 'manifest' (manifest's existing order)."
            ),
        ),
    ] = "chronological",
) -> None:
    """Concatenate MP4 files into a single output via the ffmpeg concat demuxer.

    Two modes:

    - **Positional**: ``autocut merge a.mp4 b.mp4 c.mp4 -o reel.mp4`` —
      dumb concat in the given order. Inputs must share codec/resolution/fps.
    - **Manifest-aware**: ``autocut merge --from-manifest CLIPS/manifest.json
      --min-score 7 -o reel.mp4`` — reads the manifest, picks clips with
      ``final_score >= min_score``, concats their separate-output files.
    """
    from autocut.video import ConcatError, concat_videos

    if from_manifest is not None and inputs:
        err_console.print(
            "--from-manifest is mutually exclusive with positional input files. "
            "Pass either inputs OR --from-manifest, not both."
        )
        raise typer.Exit(code=2)
    if from_manifest is None and not inputs:
        err_console.print(
            "no inputs given: pass MP4 file paths positionally, OR use --from-manifest <path>."
        )
        raise typer.Exit(code=2)

    if from_manifest is not None:
        try:
            selected = _select_from_manifest(from_manifest, min_score=min_score, order=order)
        except ManifestSelectionError as exc:
            err_console.print(str(exc))
            raise typer.Exit(code=1) from exc
        if not selected:
            err_console.print(
                f"manifest has no clips with final_score >= {min_score}; nothing to merge."
            )
            raise typer.Exit(code=1)
        input_paths = selected
    else:
        # Typer guarantees inputs is non-empty here (we checked above).
        assert inputs is not None
        input_paths = [p.resolve() for p in inputs]

    try:
        produced = concat_videos(input_paths, output.resolve())
    except ConcatError as exc:
        err_console.print(f"merge failed: {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[green]OK[/] merged {len(input_paths)} file(s) → {produced}")


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


class ManifestSelectionError(RuntimeError):
    """Raised when ``--from-manifest`` cannot resolve a valid list of clip paths."""


def _parse_cli_timestamp(raw: str, *, field: str) -> timedelta:
    """Parse a CLI timestamp accepting both ``HH:MM:SS.mmm`` and plain seconds.

    The pydantic model validator only accepts the structured form when
    given a string; CLI users naturally type ``--start 5`` for "5 seconds".
    We try the numeric conversion first (matching the model's numeric
    path) and fall back to the structured grammar for ``00:00:05.500``
    style inputs.
    """
    from autocut.models import _parse_timestamp

    stripped = raw.strip()
    try:
        # Bare number → interpret as seconds, matching the model's numeric path.
        return _parse_timestamp(float(stripped))
    except (TypeError, ValueError):
        pass
    try:
        return _parse_timestamp(stripped)
    except (TypeError, ValueError) as exc:
        raise typer.BadParameter(f"{field}: {exc}") from exc


def _select_from_manifest(
    manifest_path: Path,
    *,
    min_score: int,
    order: str,
) -> list[Path]:
    """Read an autocut manifest.json and return the matching separate-output paths.

    Selection: keep clips with ``final_score >= min_score``.
    Ordering: ``chronological`` (clip.start asc) | ``score-desc`` |
    ``manifest`` (keep manifest's existing clip order).

    Raises ``ManifestSelectionError`` if the manifest is missing, malformed,
    or doesn't contain a ``separate`` output section.
    """
    valid_orders = {"chronological", "score-desc", "manifest"}
    if order not in valid_orders:
        raise ManifestSelectionError(
            f"unknown --order {order!r}; choose from {sorted(valid_orders)}"
        )

    if not manifest_path.is_file():
        raise ManifestSelectionError(f"manifest not found: {manifest_path}")
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestSelectionError(f"failed to read manifest {manifest_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ManifestSelectionError("manifest top-level value is not an object")

    clips = data.get("clips")
    outputs = data.get("outputs", {}) if isinstance(data.get("outputs"), dict) else {}
    separate_paths_raw = outputs.get("separate")

    if not isinstance(clips, list) or not clips:
        raise ManifestSelectionError("manifest has no 'clips' array or it is empty")
    if not isinstance(separate_paths_raw, list) or not separate_paths_raw:
        raise ManifestSelectionError(
            "manifest has no 'outputs.separate' paths — re-run `autocut run` "
            "first so per-clip MP4s exist on disk."
        )
    if len(separate_paths_raw) != len(clips):
        # Defensive: dispatcher writes them index-aligned today; flag any drift
        # explicitly rather than silently mis-matching scores to paths.
        raise ManifestSelectionError(
            f"manifest is inconsistent: {len(clips)} clips vs "
            f"{len(separate_paths_raw)} separate outputs"
        )

    # Pair each clip dict with its corresponding output path before filtering,
    # so we never lose the score↔path correspondence during sort.
    pairs: list[tuple[dict[str, object], Path]] = []
    for clip, raw_path in zip(clips, separate_paths_raw, strict=True):
        if not isinstance(clip, dict):
            continue
        pairs.append((clip, Path(str(raw_path))))

    filtered = [(c, p) for c, p in pairs if _final_score(c) >= min_score]

    if order == "score-desc":
        filtered.sort(key=lambda pair: _final_score(pair[0]), reverse=True)
    elif order == "chronological":
        filtered.sort(key=lambda pair: _clip_start_seconds(pair[0]))
    # ``manifest`` keeps the existing order — no sort.

    return [p for _, p in filtered]


def _final_score(clip: dict[str, object]) -> int:
    """Return ``clip.final_score`` as an int, defaulting to 0 when absent/malformed."""
    raw = clip.get("final_score", 0)
    if isinstance(raw, int | float):
        return int(raw)
    return 0


def _clip_start_seconds(clip: dict[str, object]) -> float:
    """Return the clip start in seconds, parsing the manifest's HH:MM:SS.mmm string."""
    raw = clip.get("start", "00:00:00.000")
    if isinstance(raw, int | float):
        return float(raw)
    if isinstance(raw, str):
        try:
            td = _parse_cli_timestamp(raw, field="manifest clip.start")
        except typer.BadParameter:
            return 0.0
        return td.total_seconds()
    return 0.0


def _probe_binary(name: str) -> str:
    """Return a version string for ``name`` plus its source (system vs bundled).

    Resolves through the shared ffmpeg resolver so ``doctor`` reports exactly
    what the pipeline will use, including the static-ffmpeg fallback (which it
    fetches on first call if no system binary exists).
    """
    # Lazy import: pulls in autocut.video (cv2 etc.); `keys`/`config` must not
    # pay that cost just to start up.
    from autocut.video.ffmpeg_path import (
        FFmpegResolveError,
        describe_ffmpeg,
        describe_ffprobe,
    )

    describe = describe_ffmpeg if name == "ffmpeg" else describe_ffprobe
    try:
        path, source = describe()
    except FFmpegResolveError as exc:
        return f"[red]not found (no system {name}; bundled fetch failed: {exc})[/]"
    try:
        result = subprocess.run(  # noqa: S603 - args is a fixed list, no shell
            [path, "-version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return f"[yellow]{source} at {path} but probe failed: {exc}[/]"
    version = result.stdout.splitlines()[0] if result.stdout else "(no output)"
    tag = "system" if source == "system" else "bundled static-ffmpeg"
    return f"{version}  [dim]({tag}: {path})[/]"


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


def _build_hints_from_cli(
    content_hint_raw: str | None,
    query: str | None,
    cfg: AutoCutConfig,
) -> AnalysisHints | None:
    """Build ``AnalysisHints`` from the CLI ``--content-hint`` / ``--query``.

    Returns ``None`` only when NEITHER override is given, so the pipeline
    falls back to its own resolution from ``cfg.content_defaults``. A bare
    ``--query`` (no explicit mode) still builds hints so the query is not
    lost — the mode then resolves to the default (HYBRID) downstream.
    """
    query = (query.strip() or None) if query else None
    if content_hint_raw is None and not query:
        return None

    hint = ContentHint.auto
    if content_hint_raw is not None:
        try:
            hint = ContentHint(content_hint_raw.lower())
        except ValueError as exc:
            valid = ", ".join(h.value for h in ContentHint)
            raise typer.BadParameter(
                f"unknown --content-hint {content_hint_raw!r}; choose from: {valid}"
            ) from exc

    defaults = cfg.content_defaults
    return AnalysisHints(
        content_hint=hint,
        goal=defaults.goal,
        language=defaults.language,
        query=query,
    )


def _enforce_model_capabilities(provider_name: str, model: str) -> None:
    """For openrouter, reject a model that lacks BOTH audio and video input.

    AutoCut routes one model across the whole content matrix; a video-only or
    audio-only model cannot cover it, so we fail fast with a clear message
    before any extraction or analysis happens. Host provider is exempt — its
    capability is user-declared, not catalogue-derived.
    """
    if provider_name.strip().lower() != "openrouter":
        return
    try:
        asyncio.run(validate_openrouter_model(model))
    except VLMError as exc:
        err_console.print(str(exc))
        raise typer.Exit(code=1) from exc


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
        # typer lacks type stubs so .confirm() returns Any; coerce to bool.
        return bool(typer.confirm("Proceed anyway?", default=False))

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

    if not result.ranked:
        # The model proposed clips but none cleared the keep threshold
        # (highlights = 7). An empty result is a valid, honest outcome — we do
        # not ship a best-of-nothing.
        console.print(
            f"[yellow]{len(plan.clips)} candidate(s) proposed, but none cleared "
            f"the keep threshold — no clips produced.[/]"
        )
        return

    table = Table(title=f"{len(result.ranked)} clip(s) kept")
    table.add_column("#", justify="right")
    table.add_column("Start → End", style="bold")
    table.add_column("VLM", justify="right")
    table.add_column("Heur", justify="right")
    table.add_column("Final", justify="right", style="bold")
    table.add_column("Category")
    table.add_column("Description")

    for i, ranked in enumerate(result.ranked, start=1):
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

    if result.plan_path:
        console.print(f"\n[bold]Plan[/]: {result.plan_path}")
        console.print(
            "[dim]Review/edit the plan, then cut the clips with:[/]\n"
            f"  autocut cut --from-json {result.plan_path} "
            f"--video {result.metadata.path} --output-dir <DIR>"
        )
    else:
        console.print(
            "\n[dim]Dry run — no plan written. Re-run without --dry-run to produce plan.json.[/]"
        )

    console.print(
        f"\n[dim]Full ClipPlan JSON:[/]\n{json.dumps(plan.model_dump(mode='json'), indent=2)}"
    )


if __name__ == "__main__":  # pragma: no cover
    app()
