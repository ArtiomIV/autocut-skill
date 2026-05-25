"""Run one or more output writers and persist a manifest.

The dispatcher takes the list of modes from the config (e.g.
``["separate", "merged"]``), runs the matching ``OutputWriter`` for each,
and finally writes ``manifest.json`` summarising the run for downstream
tooling (CapCut import scripts, log shipping, debugging).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from autocut.config import MergeOrder, OutputMode
from autocut.models import VideoMetadata
from autocut.output.base import OutputWriter, WrittenClip
from autocut.output.merged import MergedWriter
from autocut.output.separate import SeparateWriter
from autocut.scoring import RankedClip

log = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True, slots=True)
class DispatchResult:
    """Bundle returned by ``dispatch_outputs``: per-mode written files + manifest."""

    by_mode: dict[str, list[WrittenClip]] = field(default_factory=dict)
    manifest_path: Path | None = None


def dispatch_outputs(
    video_path: Path,
    ranked: list[RankedClip],
    metadata: VideoMetadata,
    output_dir: Path,
    *,
    modes: list[OutputMode],
    merge_order: MergeOrder = "score",
    accurate: bool = False,
    pre_roll_sec: float = 0.0,
    post_roll_sec: float = 0.0,
    extra_manifest: dict[str, object] | None = None,
) -> DispatchResult:
    """Run every requested writer in order and persist the manifest.

    ``pre_roll_sec`` / ``post_roll_sec`` are forwarded to each writer so the
    cutter can widen clip bounds. Defaults of zero mean tight cuts; the
    pipeline normally sources these from the active ``ContentProfile``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    writers: list[OutputWriter] = []
    for mode in modes:
        if mode == "separate":
            writers.append(SeparateWriter())
        elif mode == "merged":
            writers.append(MergedWriter(order=merge_order))
        elif mode == "all":
            # ``all`` should already have been normalised by AutoCutConfig,
            # but if someone calls dispatch_outputs directly with it we handle
            # it gracefully here.
            writers.append(SeparateWriter())
            writers.append(MergedWriter(order=merge_order))
        else:
            raise ValueError(f"unknown output mode: {mode!r}")

    by_mode: dict[str, list[WrittenClip]] = {}
    for writer in writers:
        written = writer.write(
            video_path,
            ranked,
            output_dir,
            accurate=accurate,
            pre_roll_sec=pre_roll_sec,
            post_roll_sec=post_roll_sec,
            video_duration_sec=metadata.duration_sec,
        )
        by_mode.setdefault(writer.name, []).extend(written)

    manifest_path = _write_manifest(
        output_dir=output_dir,
        video_path=video_path,
        metadata=metadata,
        ranked=ranked,
        by_mode=by_mode,
        extra=extra_manifest or {},
    )
    return DispatchResult(by_mode=by_mode, manifest_path=manifest_path)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _write_manifest(
    *,
    output_dir: Path,
    video_path: Path,
    metadata: VideoMetadata,
    ranked: list[RankedClip],
    by_mode: dict[str, list[WrittenClip]],
    extra: dict[str, object],
) -> Path:
    manifest = {
        "video": {
            "path": str(video_path),
            "duration_sec": metadata.duration_sec,
            "width": metadata.width,
            "height": metadata.height,
            "fps": metadata.fps,
            "video_codec": metadata.video_codec,
            "audio_codec": metadata.audio_codec,
        },
        "clips": [
            {
                "rank": i + 1,
                "id": r.clip.id,
                "start": _format_ts(r.clip.start.total_seconds()),
                "end": _format_ts(r.clip.end.total_seconds()),
                "duration_sec": round(r.clip.duration_sec, 3),
                "category": r.clip.category.value,
                "description": r.clip.description,
                "vlm_score": r.vlm_score,
                "heuristic_score": r.heuristic_score,
                "final_score": r.final_score,
                "tags": r.clip.tags,
            }
            for i, r in enumerate(ranked)
        ],
        "outputs": {mode: [str(w.path) for w in writes] for mode, writes in by_mode.items()},
        **extra,
    }
    manifest_path = output_dir / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("manifest: wrote %s with %d clip(s)", manifest_path, len(ranked))
    return manifest_path


def _format_ts(seconds: float) -> str:
    total_ms = max(0, int(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"
