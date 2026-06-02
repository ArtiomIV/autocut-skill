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
from autocut.models import Clip, VideoMetadata
from autocut.output.base import OutputWriter, WrittenClip
from autocut.output.merged import MergedWriter
from autocut.output.separate import SeparateWriter
from autocut.scoring import RankedClip
from autocut.scoring.ranker import final_score

log = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.json"
# ``run`` writes this analysis-only artefact (ranked clips with the pre/post-roll
# already baked into the timestamps, NO cut MP4s); ``cut --from-json`` consumes it.
PLAN_FILENAME = "plan.json"


class PlanReadError(RuntimeError):
    """Raised when a plan.json cannot be read or reconstructed into clips."""


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


# ---------------------------------------------------------------------------
# Plan JSON (run output) — analysis only, no cutting
# ---------------------------------------------------------------------------


def write_plan_json(
    *,
    output_dir: Path,
    video_path: Path,
    metadata: VideoMetadata,
    ranked: list[RankedClip],
    pre_roll_sec: float = 0.0,
    post_roll_sec: float = 0.0,
    extra: dict[str, object] | None = None,
) -> Path:
    """Write ``plan.json``: the ranked clips with pre/post-roll baked into their
    timestamps (clamped to the source), and NO cut MP4s.

    This is what ``run`` produces now. The orchestrating agent reviews/edits it,
    then ``cut --from-json`` reads it and trims each clip 1:1 (the roll is already
    in the timestamps, so the cut applies no further padding). ``rationale`` is
    persisted so the plan reconstructs faithfully.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    clips = []
    for i, r in enumerate(ranked):
        start = max(0.0, r.clip.start.total_seconds() - pre_roll_sec)
        end = min(metadata.duration_sec, r.clip.end.total_seconds() + post_roll_sec)
        clips.append(
            {
                "rank": i + 1,
                "id": r.clip.id,
                "start": _format_ts(start),
                "end": _format_ts(end),
                "duration_sec": round(end - start, 3),
                "category": r.clip.category.value,
                "description": r.clip.description,
                "rationale": r.clip.rationale,
                "vlm_score": r.vlm_score,
                "heuristic_score": r.heuristic_score,
                "final_score": r.final_score,
                "tags": r.clip.tags,
            }
        )
    plan = {
        "video": {
            "path": str(video_path),
            "duration_sec": metadata.duration_sec,
            "width": metadata.width,
            "height": metadata.height,
            "fps": metadata.fps,
            "video_codec": metadata.video_codec,
            "audio_codec": metadata.audio_codec,
            "container": metadata.container,
            "size_bytes": metadata.size_bytes,
        },
        "clips": clips,
        **(extra or {}),
    }
    plan_path = output_dir / PLAN_FILENAME
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    log.info("plan: wrote %s with %d clip(s)", plan_path, len(ranked))
    return plan_path


def read_plan_json(
    plan_path: Path, *, min_score: int = 0
) -> tuple[Path, VideoMetadata, list[RankedClip]]:
    """Reconstruct ``(video_path, metadata, ranked)`` from a ``plan.json``.

    Keeps only clips with ``final_score >= min_score`` (0..N may survive). The
    timestamps already include the roll, so downstream cutting pads nothing.
    Raises ``PlanReadError`` on a missing/malformed plan.
    """
    if not plan_path.is_file():
        raise PlanReadError(f"plan not found: {plan_path}")
    try:
        data = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanReadError(f"failed to read plan {plan_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise PlanReadError("plan top-level value is not an object")

    v = data.get("video")
    if not isinstance(v, dict):
        raise PlanReadError("plan has no 'video' object")
    try:
        metadata = VideoMetadata.model_validate(v)
    except ValueError as exc:
        raise PlanReadError(f"plan 'video' block is invalid: {exc}") from exc

    raw_clips = data.get("clips")
    if not isinstance(raw_clips, list):
        raise PlanReadError("plan has no 'clips' array")

    ranked: list[RankedClip] = []
    for c in raw_clips:
        if not isinstance(c, dict):
            continue
        fs = c.get("final_score")
        fs_int = int(fs) if isinstance(fs, int | float) else None
        vlm = c.get("vlm_score")
        vlm_int = int(vlm) if isinstance(vlm, int | float) else (fs_int or 0)
        heur = c.get("heuristic_score")
        heur_int = int(heur) if isinstance(heur, int | float) else 5
        if fs_int is None:
            fs_int = final_score(vlm_int, heur_int)
        if fs_int < min_score:
            continue
        try:
            clip = Clip.model_validate(
                {
                    "id": str(c.get("id") or f"clip_{len(ranked) + 1}"),
                    "start": c["start"],
                    "end": c["end"],
                    "category": c.get("category", "highlight"),
                    "description": c.get("description") or "clip",
                    "score": vlm_int,
                    "rationale": c.get("rationale") or "from plan.json",
                    "tags": c.get("tags", []),
                }
            )
        except (KeyError, ValueError) as exc:
            raise PlanReadError(f"plan clip is invalid: {exc}") from exc
        ranked.append(
            RankedClip(clip=clip, vlm_score=vlm_int, heuristic_score=heur_int, final_score=fs_int)
        )
    return Path(str(v.get("path", ""))), metadata, ranked
