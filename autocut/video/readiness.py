"""Wait until an input file has finished being written before we touch it.

When a user drops a video into a watched folder (e.g. a phone transfer over the
network), the file *exists* the instant the copy starts but its bytes keep
arriving for seconds afterwards. A naive ``exists()`` check passes immediately,
so the pipeline can probe/cut a half-written file: ffprobe usually errors on a
truncated MP4 (no ``moov`` atom), but a partially-copied faststart MOV can probe
"successfully" and then yield a truncated cut. Either way we never want to start.

The robust, OS-portable signal is *stability*: while a file is still being
written its ``(size, mtime)`` keeps changing; once the copy completes the pair
stops changing. We poll that pair and only declare the file ready once it has
been identical for ``stable_for`` seconds AND we can open it for reading (which
also catches the exclusive lock a Windows copy may hold mid-transfer).

This is deliberately deterministic plumbing — no model judgement — so it lives
in code and is exposed as ``autocut wait-ready`` for the orchestrating agent to
call as the FIRST step of both the host and the cloud recipe.
"""

from __future__ import annotations

import stat
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# A point-in-time fingerprint of the file: (size_bytes, mtime_ns).
Signature = tuple[int, int]


class ReadinessTimeout(RuntimeError):  # noqa: N818
    """Raised when the file did not become stable within the timeout window."""


@dataclass(frozen=True)
class ReadyResult:
    """Outcome of a successful wait."""

    size_bytes: int
    waited_sec: float


def file_signature(path: Path) -> Signature | None:
    """Return ``(size_bytes, mtime_ns)`` for ``path``, or ``None`` if not usable.

    ``None`` means "not ready to read yet": the path is missing, is not a regular
    file, or cannot currently be opened for reading (e.g. an in-progress copy
    holds an exclusive lock on Windows). Callers treat ``None`` exactly like a
    still-changing file — keep waiting.
    """
    try:
        st = path.stat()
    except OSError:
        return None
    if not stat.S_ISREG(st.st_mode):
        return None
    # Confirm the bytes are actually readable right now, not just that an entry
    # exists. A copy still holding the file open exclusively trips this.
    try:
        with path.open("rb") as fh:
            fh.read(1)
    except OSError:
        return None
    return (st.st_size, st.st_mtime_ns)


def wait_until_ready(
    path: str | Path,
    *,
    stable_for: float = 2.0,
    timeout: float = 900.0,
    poll: float = 1.0,
    signature: Callable[[Path], Signature | None] = file_signature,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> ReadyResult:
    """Block until ``path`` is a non-empty file whose size+mtime are stable.

    Parameters
    ----------
    stable_for:
        How many seconds the ``(size, mtime)`` pair must stay identical before
        the file is declared ready.
    timeout:
        Give up (raise :class:`ReadinessTimeout`) after this many seconds. Also
        covers the case where the file never appears at all.
    poll:
        Seconds between checks.
    signature, sleep, monotonic:
        Injection seams for tests; defaults use the real filesystem and clock.

    Returns
    -------
    ReadyResult
        The final size and how long we waited.

    Raises
    ------
    ReadinessTimeout
        If stability was not reached within ``timeout`` seconds.
    """
    target = Path(path)
    start = monotonic()
    last_sig: Signature | None = None
    stable_since = start

    while True:
        sig = signature(target)
        now = monotonic()
        is_candidate = sig is not None and sig[0] > 0

        if not is_candidate:
            # Missing, empty, or locked: reset the stability window.
            last_sig = None
            stable_since = now
        elif sig != last_sig:
            # First sighting or the file changed since last poll: restart timing.
            last_sig = sig
            stable_since = now
        elif now - stable_since >= stable_for:
            assert sig is not None  # narrowed by is_candidate
            return ReadyResult(size_bytes=sig[0], waited_sec=now - start)

        if now - start >= timeout:
            raise ReadinessTimeout(
                f"{target} did not stabilise within {timeout:g}s "
                "(still copying, locked, or never arrived)"
            )
        sleep(poll)
