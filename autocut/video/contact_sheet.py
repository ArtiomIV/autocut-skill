"""Render a candidate window as contact-sheet montages with per-cell timestamps.

The two-pass fine step needs sub-second boundary precision, which the model
cannot get from a video clip it re-samples to ~1fps. Sending the window as many
separate stills works but overloads the request (an E2E saw empty responses and
retries that re-sent every frame). A contact sheet collapses the frames into a
single image per sheet: one (or a few) image parts instead of dozens, so the
request stays small and reliable, while small cells keep the pixel-area token
cost low.

Each cell carries a short integer INDEX (its frame number in reading order),
burned in by ffmpeg ``drawtext``. The caller pairs the sheets with a JSON map
``index -> timestamp`` so the model reads a small, OCR-friendly integer off each
cell and looks up the EXACT clip-relative time in the map — no decimals to OCR
and no cell-counting to drift. Because the segment is cut to start at t=0 and the
``fps`` filter emits frames at ``i/fps`` seconds, frame ``i``'s time is exactly
``i/fps`` (clip-relative); ``video_analysis`` re-bases it to absolute source time.

One ffmpeg invocation does the whole pipeline: sample at ``fps`` -> downscale each
frame to ``cell_px`` -> draw its index -> ``tile`` into a ``cols`` x ``rows``
grid. Windows with more frames than one grid holds spill into additional sheets.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from autocut.video.ffmpeg_path import FFmpegResolveError, ffmpeg_binary

log = logging.getLogger(__name__)


class ContactSheetError(RuntimeError):
    """Raised when ffmpeg fails to produce the contact sheet(s)."""


_DEFAULT_FPS = 2.0
_DEFAULT_CELL_PX = 192
_DEFAULT_COLS = 6
_DEFAULT_ROWS = 8
_DEFAULT_FONT_SIZE = 18
_DEFAULT_JPEG_QSCALE = 4
_DEFAULT_TIMEOUT_SEC = 120

# Fonts to try for the burned-in timestamp when the caller passes none. First hit
# wins; drawtext needs a real TrueType file.
_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\consola.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
)


def build_contact_sheets(
    segment: Path,
    out_dir: Path,
    *,
    fps: float = _DEFAULT_FPS,
    cell_px: int = _DEFAULT_CELL_PX,
    cols: int = _DEFAULT_COLS,
    rows: int = _DEFAULT_ROWS,
    font_path: str | None = None,
    ffmpeg_path: str | None = None,
) -> list[Path]:
    """Render ``segment`` into one or more timestamped contact sheets.

    Returns the sheet image paths in chronological order (``sheet_000.jpg``,
    ``sheet_001.jpg``, …). ``out_dir`` is created if missing. Raises
    ``ContactSheetError`` on any ffmpeg/setup failure.
    """
    if not segment.is_file():
        raise ContactSheetError(f"segment not found: {segment}")
    if fps <= 0 or cell_px <= 0 or cols <= 0 or rows <= 0:
        raise ContactSheetError("fps, cell_px, cols and rows must all be > 0")

    try:
        binary = ffmpeg_binary(ffmpeg_path)
    except FFmpegResolveError as exc:
        raise ContactSheetError(f"ffmpeg not found: {exc}") from exc

    font = _resolve_font(font_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "sheet_%03d.jpg"

    vf = _build_filter(fps=fps, cell_px=cell_px, cols=cols, rows=rows, font=font)
    args = [
        binary,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(segment),
        "-vf",
        vf,
        "-q:v",
        str(_DEFAULT_JPEG_QSCALE),
        str(pattern),
    ]
    try:
        completed = subprocess.run(  # noqa: S603 - args is a fixed list, no shell
            args,
            capture_output=True,
            text=True,
            timeout=_DEFAULT_TIMEOUT_SEC,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ContactSheetError(f"ffmpeg timed out building contact sheet for {segment}") from exc
    except OSError as exc:
        raise ContactSheetError(f"failed to invoke ffmpeg: {exc}") from exc

    if completed.returncode != 0:
        raise ContactSheetError(
            f"ffmpeg failed building contact sheet for {segment}: {completed.stderr.strip()}"
        )

    sheets = sorted(out_dir.glob("sheet_*.jpg"))
    if not sheets:
        raise ContactSheetError(f"ffmpeg produced no sheets for {segment}")
    log.info(
        "contact sheet: %s -> %d sheet(s) (%dx%d cells @ %.1f fps, %dpx)",
        segment.name,
        len(sheets),
        cols,
        rows,
        fps,
        cell_px,
    )
    return sheets


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _build_filter(*, fps: float, cell_px: int, cols: int, rows: int, font: str) -> str:
    """Assemble the ffmpeg ``-vf`` filtergraph: fps → scale → drawtext → tile."""
    # Each stage is comma-chained. The burned label is the frame's INDEX in the
    # filtered stream (``%{n}``, 0-based), drawn BEFORE tile so each cell shows its
    # own index; the caller maps that index to the exact time. Only the Windows
    # font path needs colon-escaping (``%{n}`` has no colon).
    drawtext = (
        f"drawtext=fontfile='{_escape_fontpath(font)}'"
        r":text='%{n}'"
        ":x=4:y=4"
        f":fontsize={_DEFAULT_FONT_SIZE}"
        ":fontcolor=yellow:box=1:boxcolor=black@0.6:boxborderw=3"
    )
    return (
        f"fps={fps:g},"
        f"scale='if(gt(iw,ih),{cell_px},-2)':'if(gt(iw,ih),-2,{cell_px})',"
        f"{drawtext},"
        f"tile={cols}x{rows}:margin=4:padding=4:color=black"
    )


def _escape_fontpath(path: str) -> str:
    """Escape a font path for use inside an ffmpeg filtergraph value.

    Backslashes become forward slashes and the drive-letter colon is escaped, so
    ``C:\\Windows\\Fonts\\arial.ttf`` reads as ``C\\:/Windows/Fonts/arial.ttf``.
    """
    return path.replace("\\", "/").replace(":", r"\:")


def _resolve_font(font_path: str | None) -> str:
    """Return a usable TrueType font path, or raise if none is found."""
    if font_path:
        if not Path(font_path).is_file():
            raise ContactSheetError(f"font file not found: {font_path}")
        return font_path
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    raise ContactSheetError(
        "no TrueType font found for the burned-in timestamp; pass font_path explicitly"
    )
