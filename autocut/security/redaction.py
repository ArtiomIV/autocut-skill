"""Log redaction filter.

Defence-in-depth against accidentally logging API keys. Patterns cover the
common shapes (Anthropic, OpenAI, OpenRouter, Google, generic Bearer tokens).
The primary safeguard is to never pass keys into log calls at all; this filter
exists for the cases where a deeply nested exception trace or a misbehaving
third-party library leaks them.
"""

from __future__ import annotations

import logging
import re

# Each pattern is broad enough to catch realistic keys without flagging arbitrary
# code identifiers. Order does not matter — we apply them all.
_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Anthropic: sk-ant-api03-...
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    # OpenRouter: sk-or-v1-...
    re.compile(r"sk-or-[A-Za-z0-9_\-]{20,}"),
    # Generic OpenAI-style: sk-...
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    # Google API keys: AIza...
    re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    # Bearer tokens (covers many SDKs that put the key in an Authorization header).
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]{20,}", re.IGNORECASE),
)

_REPLACEMENT = "[REDACTED]"


def redact(text: str) -> str:
    """Return ``text`` with all known key-shaped substrings replaced."""
    for pattern in _PATTERNS:
        text = pattern.sub(_REPLACEMENT, text)
    return text


class RedactionFilter(logging.Filter):
    """A logging filter that scrubs key-shaped strings from every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            # Logging must never crash the program itself.
            return True
        redacted = redact(msg)
        if redacted != msg:
            record.msg = redacted
            record.args = ()
        return True


def install_redaction_filter(logger: logging.Logger | None = None) -> None:
    """Attach a ``RedactionFilter`` to ``logger`` (root logger by default).

    Idempotent: a second call is a no-op.
    """
    target = logger or logging.getLogger()
    if any(isinstance(f, RedactionFilter) for f in target.filters):
        return
    target.addFilter(RedactionFilter())
