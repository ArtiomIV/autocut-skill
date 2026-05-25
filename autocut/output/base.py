"""Shared output-writer interface and result records."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from autocut.scoring import RankedClip


@dataclass(frozen=True, slots=True)
class WrittenClip:
    """A single output file produced by a writer."""

    path: Path
    source: RankedClip


class OutputWriter(ABC):
    """Common interface for every output mode (separate, merged, future capcut)."""

    name: ClassVar[str]

    @abstractmethod
    def write(
        self,
        video_path: Path,
        clips: list[RankedClip],
        output_dir: Path,
        *,
        accurate: bool = False,
    ) -> list[WrittenClip]:
        """Produce output files for ``clips`` under ``output_dir``."""


# ---------------------------------------------------------------------------
# Filename slug — shared by every writer
# ---------------------------------------------------------------------------


_SLUG_BAD = re.compile(r"[^A-Za-z0-9\-]+")
_SLUG_SQUEEZE = re.compile(r"-{2,}")


def slugify(text: str, *, max_length: int = 30) -> str:
    """Return a filesystem-safe slug derived from ``text``.

    Defensive: clip descriptions come from a VLM and could contain anything
    (path separators, NUL bytes, unicode arrows). We keep only ASCII letters,
    digits, and hyphens.
    """
    if not text:
        return "clip"
    cleaned = _SLUG_BAD.sub("-", text.strip())
    cleaned = _SLUG_SQUEEZE.sub("-", cleaned).strip("-")
    cleaned = cleaned[:max_length].rstrip("-")
    return cleaned or "clip"
