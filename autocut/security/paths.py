"""Path validation utilities.

AutoCut writes outputs under ``./CLIPS/`` in the user's cwd and reads input
videos from arbitrary paths the user provides. We must defend against:
- Symlink games that point inside ``CLIPS/`` to a system folder.
- ``..`` traversal in filenames coming from the VLM (the slug builder uses the
  clip description; a malicious model could craft a name like ``../etc/passwd``).
- Absolute paths in places where only relative is expected.

The two main entry points:
- ``safe_resolve(path)`` — resolve symlinks and normalise.
- ``ensure_inside(child, parent)`` — assert ``child`` resolves inside ``parent``.
"""

from __future__ import annotations

from pathlib import Path


class PathValidationError(ValueError):
    """Raised when a path fails a security check."""


def safe_resolve(path: str | Path) -> Path:
    """Resolve a path: expand ``~``, expand env vars, resolve symlinks.

    Unlike ``Path.resolve(strict=True)``, this returns even if the path does
    not exist yet — useful for output paths we are about to create.
    """
    p = Path(path).expanduser()
    return p.resolve()


def ensure_inside(child: str | Path, parent: str | Path) -> Path:
    """Return ``child`` resolved, guaranteed to be inside ``parent``.

    Raises ``PathValidationError`` if ``child`` (after symlink resolution)
    escapes ``parent``. This is the canonical guard used by output writers
    before opening any file under ``CLIPS/``.
    """
    parent_resolved = safe_resolve(parent)
    child_resolved = safe_resolve(child)
    try:
        child_resolved.relative_to(parent_resolved)
    except ValueError as exc:
        raise PathValidationError(
            f"path {child_resolved} escapes allowed parent {parent_resolved}"
        ) from exc
    return child_resolved
