"""Host image-only two-pass: build the contact sheets and pick candidates.

The host provider cannot ingest video or audio — it only reads image files from
disk and pauses for the surrounding agent. To match the cloud two-pass quality
(coarse locate → fine precise) without any video payload, the whole timeline is
rendered as TIMESTAMPED CONTACT SHEETS and analysed image-only:

* **Coarse** (sources > 60s): sparse grids at 1 frame / 3s over the WHOLE
  timeline, built in 5-minute batches (one ffmpeg call per batch) sharing one
  continuous burned-in index space. The agent returns wide candidate regions.
* **Fine**: DENSE grids at 2 FPS of every selected candidate window (padded
  ±20s), concatenated into ONE continuous index space so the agent sees them all
  in a single pause. Boundaries land on the sub-second event.

The index → time maps are in ABSOLUTE source time, so the agent's returned
``start``/``end`` are already absolute — unlike the cloud fine pass there is no
per-candidate re-basing. ``dedup_plan`` then drops overlapping duplicates.

This module is the host counterpart of ``autocut.video_analysis`` (the cloud
engine). It does the ffmpeg/montage work and the pure candidate maths; the
pause/resume handshake lives in ``autocut.vlm.host_agent`` and the orchestration
in ``autocut.pipeline``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from autocut.chunking import temporal_iou
from autocut.models import Clip, ClipPlan, ClipPlanMetadata
from autocut.video.contact_sheet import build_contact_sheets

log = logging.getLogger(__name__)

# Above this duration the host runs the two-pass (coarse + fine); at or below it
# the coarse pass is skipped and a single fine pass covers the whole video.
HOST_TWO_PASS_MIN_DURATION_SEC: float = 60.0

# Coarse pass: 1 frame / 3s keeps the sparse pass cheap while still catching every
# region; batched at 5 min (an exact multiple of the interval, so per-batch frame
# counts are exact and the continuous index space never drifts).
COARSE_INTERVAL_SEC: float = 3.0
COARSE_BATCH_SEC: float = 300.0
# Bigger cells than the cloud fine pass (192px): the host agent reads the burned
# index off the grid, and the whole sheet is downscaled to the model's max image
# size — at 192px on a 16:9 source the indices were borderline illegible. The host
# is free, so the larger payload costs nothing.
COARSE_CELL_PX: int = 256

# Fine pass: ALWAYS 2 FPS (0.5s) dense sampling, no frame cap — the host reads the
# sheets locally for free, so there is no reason to dilute a long window (a cap
# made the decisive punch fall between frames). ±20s padding gives the moment its
# lead-in and the slow-motion replay that often follows.
FINE_INTERVAL_SEC: float = 0.5
FINE_CELL_PX: int = 256  # legible burned index (host is free; see COARSE_CELL_PX)
FINE_PAD_SEC: float = 20.0
# Keep EVERY coarse candidate scored >= this, not a fixed top-N — a long match has
# many strong regions and a hard cap dropped the ones just outside it. A generous
# ceiling guards against a runaway coarse response (and bounds how many sheets the
# host agent must read).
CANDIDATE_MIN_SCORE: int = 5
CANDIDATE_CEILING: int = 40


# ---------------------------------------------------------------------------
# Sheet building
# ---------------------------------------------------------------------------


def build_coarse_sheets(
    video: Path,
    out_dir: Path,
    duration_sec: float,
) -> tuple[list[Path], list[float]]:
    """Render the WHOLE timeline as sparse contact sheets (1 frame / 3s).

    Returns ``(sheets, frame_times)`` where ``frame_times[i]`` is the ABSOLUTE
    source time (seconds) of the cell burning global index ``i``. Built in 5-min
    batches (exact multiples of the interval) that share one continuous index
    space via ``index_offset``, so the agent sees ONE ordered grid of the whole
    video and returns absolute candidate regions.
    """
    interval = COARSE_INTERVAL_SEC
    fps = 1.0 / interval
    sheets_all: list[Path] = []
    times_all: list[float] = []
    offset = 0
    batch_index = 0
    start = 0.0
    while start < duration_sec:
        end = min(duration_sec, start + COARSE_BATCH_SEC)
        window = end - start
        if window < interval:
            # Trailing remainder shorter than one sample: ffmpeg would emit no
            # frame at this fps, so skip it (we lose <3s at the very end).
            break
        n_frames = max(1, int(window / interval))
        batch_dir = out_dir / f"coarse_{batch_index:03d}"
        sheets = build_contact_sheets(
            video,
            batch_dir,
            fps=fps,
            cell_px=COARSE_CELL_PX,
            index_offset=offset,
            start_sec=start,
            trim_sec=n_frames * interval,
        )
        sheets_all.extend(sheets)
        times_all.extend(start + j * interval for j in range(n_frames))
        offset += n_frames
        batch_index += 1
        start = end
    log.info(
        "host coarse: %s -> %d sheet(s), %d frame(s) @ %.2f fps over %.1fs",
        video.name,
        len(sheets_all),
        len(times_all),
        fps,
        duration_sec,
    )
    return sheets_all, times_all


def build_fine_sheets(
    video: Path,
    out_dir: Path,
    windows: list[tuple[float, float]],
) -> tuple[list[Path], list[float]]:
    """Render every candidate ``window`` as DENSE 2-FPS sheets, one index space.

    ``windows`` are absolute ``(lo, hi)`` ranges (already padded; they MAY overlap
    — overlap is fine, the duplicate clips are dropped by ``dedup_plan`` later).
    Each is sampled at a fixed 2 FPS (no frame cap) and the burned indices continue
    across windows, so the whole set forms ONE continuous index space for a single
    pause. Returns ``(sheets, frame_times)`` with ``frame_times`` in ABSOLUTE source
    time — note it JUMPS between windows (the host prompt tells the model to trust
    the map).
    """
    sheets_all: list[Path] = []
    times_all: list[float] = []
    offset = 0
    for w_index, (lo, hi) in enumerate(windows):
        window = hi - lo
        if window < FINE_INTERVAL_SEC:
            continue  # too short to emit even one frame at the fine fps
        interval = FINE_INTERVAL_SEC  # always 2 FPS, no cap (host frames are free)
        fps = 1.0 / interval
        n_frames = max(1, int(window / interval))
        win_dir = out_dir / f"fine_{w_index:03d}"
        sheets = build_contact_sheets(
            video,
            win_dir,
            fps=fps,
            cell_px=FINE_CELL_PX,
            index_offset=offset,
            start_sec=lo,
            trim_sec=n_frames * interval,
        )
        sheets_all.extend(sheets)
        times_all.extend(lo + j * interval for j in range(n_frames))
        offset += n_frames
    log.info(
        "host fine: %d window(s) -> %d sheet(s), %d frame(s)",
        len(windows),
        len(sheets_all),
        len(times_all),
    )
    return sheets_all, times_all


# ---------------------------------------------------------------------------
# Candidate maths (pure)
# ---------------------------------------------------------------------------


def select_fine_windows(
    coarse_plan: ClipPlan,
    duration_sec: float,
    *,
    pad_sec: float = FINE_PAD_SEC,
    min_score: int = CANDIDATE_MIN_SCORE,
    ceiling: int = CANDIDATE_CEILING,
) -> list[tuple[float, float]]:
    """Turn coarse candidate clips into padded, absolute fine windows.

    Keeps EVERY candidate scored >= ``min_score`` (capped at ``ceiling`` as a
    safety net), not a fixed top-N, and pads each by ``pad_sec`` on both sides
    (clamped to the source). Windows are deliberately NOT merged — candidates are
    allowed to touch/overlap: each keeps its OWN window sampled densely at full
    2 FPS, and any duplicate clips that result are dropped later by ``dedup_plan``.
    Merging used to fuse a chain of adjacent candidates into one huge window that
    then got sampled too sparsely, hiding the decisive punch. Chronological order.
    """
    top = [c for c in sorted(coarse_plan.clips, key=lambda c: -c.score) if c.score >= min_score][
        :ceiling
    ]
    raw: list[tuple[float, float]] = []
    for clip in top:
        lo = max(0.0, clip.start.total_seconds() - pad_sec)
        hi = min(duration_sec, clip.end.total_seconds() + pad_sec)
        if hi > lo:
            raw.append((lo, hi))
    raw.sort()
    return raw


def dedup_plan(plan: ClipPlan, *, iou_threshold: float = 0.5) -> ClipPlan:
    """Drop overlapping duplicate clips, keeping the higher-scored one.

    The fine pass returns absolute-time clips directly (no re-basing), but
    padded candidate windows can overlap, so two windows may surface the same
    moment. Greedy temporal-IoU dedup mirrors ``chunking.merge_plans`` step 3,
    then re-sorts chronologically.
    """
    ordered = sorted(plan.clips, key=lambda c: (-c.score, c.start))
    kept: list[Clip] = []
    for clip in ordered:
        if any(
            temporal_iou(
                clip.start.total_seconds(),
                clip.end.total_seconds(),
                k.start.total_seconds(),
                k.end.total_seconds(),
            )
            >= iou_threshold
            for k in kept
        ):
            continue
        kept.append(clip)
    kept.sort(key=lambda c: c.start)
    return plan.model_copy(update={"clips": kept})


def empty_plan(
    video_id: str, duration_sec: float, *, agent_hint: str, prompt_version: str
) -> ClipPlan:
    """Return a valid empty ``ClipPlan`` (no candidate survived)."""
    return ClipPlan(
        video_id=video_id,
        duration_sec=duration_sec,
        clips=[],
        metadata=ClipPlanMetadata(
            vlm_provider="host",
            vlm_model=agent_hint,
            prompt_version=prompt_version,
        ),
    )
