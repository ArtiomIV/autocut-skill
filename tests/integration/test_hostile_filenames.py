"""Hostile filenames through the ffmpeg helpers (regression, 2026-07-29).

Windows regression found by a downstream dress rehearsal (cutman-ai):
every subprocess helper decoded ffmpeg/ffprobe output with `text=True`,
i.e. the locale ANSI codepage. ffmpeg speaks UTF-8 and echoes the input
filename in its output (ffprobe even embeds it in the JSON payload), so
a filename with bytes outside the codepage — e.g. Japanese, whose UTF-8
encoding contains 0x81/0x90, holes in cp1252 — raised UnicodeDecodeError
instead of processing the file. All helpers now decode UTF-8 explicitly
with errors="replace".
"""

from __future__ import annotations

import shutil
from datetime import timedelta
from pathlib import Path

from autocut.video.cutter import CutRequest, cut_clip
from autocut.video.probe import probe_video

# Japanese (0x81 bytes in UTF-8), cyrillic, emoji, spaces and metachars:
# the worst of what camera operators actually deliver.
HOSTILE_NAMES = [
    "名前のないクリップ.mp4",
    "видео финал.mp4",
    "🔥 knockout!!.mp4",
    "rossi & bianchi; ko.mp4",
]


def test_probe_survives_hostile_filenames(short_video: Path, tmp_path: Path) -> None:
    for name in HOSTILE_NAMES:
        hostile = tmp_path / name
        shutil.copy(short_video, hostile)
        metadata = probe_video(hostile)
        assert metadata.duration_sec > 3.0, name


def test_cut_survives_hostile_filenames(short_video: Path, tmp_path: Path) -> None:
    hostile = tmp_path / HOSTILE_NAMES[0]
    shutil.copy(short_video, hostile)
    output = tmp_path / "cut" / "出力クリップ.mp4"
    request = CutRequest(start=timedelta(seconds=1), end=timedelta(seconds=3), output_path=output)
    result = Path(cut_clip(hostile, request, accurate=False))
    assert result.is_file()
    assert result.stat().st_size > 1000
