"""Tests for ``autocut.security.redaction`` — log key-scrubbing."""

from __future__ import annotations

import logging

import pytest

from autocut.security.redaction import RedactionFilter, install_redaction_filter, redact


@pytest.mark.parametrize(
    "raw",
    [
        "sk-ant-api03-aBcDeFgHiJkLmNoPqRsTuVwXyZ1234567890_AbCdEf",
        "sk-or-v1-aBcDeFgHiJkLmNoPqRsTuVwXyZ1234567890",
        "sk-aBcDeFgHiJkLmNoPqRsTuVwXyZ1234567890",
        # Real Google API keys are exactly "AIza" + 35 chars (39 total).
        "AIzaSyA1234567890_aBcDeFgHiJkLmNoPqRsTu",
        "Bearer abc123def456ghi789jkl012mno345pqr678",
        "bearer abc123def456ghi789jkl012mno345pqr678",  # case-insensitive
    ],
)
def test_redact_replaces_known_key_shapes(raw: str) -> None:
    out = redact(f"log line containing {raw} and more text")
    assert "[REDACTED]" in out
    assert raw not in out


def test_redact_leaves_clean_text_untouched() -> None:
    msg = "this line has no secrets, just normal log output"
    assert redact(msg) == msg


def test_redact_handles_multiple_secrets_in_one_line() -> None:
    # Real Google API keys are exactly "AIza" + 35 chars.
    raw = "first sk-ant-api03-aaaaaaaaaaaaaaaaaaaa then AIzaSyA1234567890_aBcDeFgHiJkLmNoPqRsTu"
    out = redact(raw)
    assert "sk-ant-" not in out
    assert "AIzaSy" not in out
    assert out.count("[REDACTED]") == 2


def test_redact_is_idempotent() -> None:
    msg = "before sk-or-v1-12345678901234567890 after"
    once = redact(msg)
    twice = redact(once)
    assert once == twice


# ---------------------------------------------------------------------------
# RedactionFilter integration with logging
# ---------------------------------------------------------------------------


def test_filter_scrubs_log_records(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("autocut.test_filter")
    logger.setLevel(logging.INFO)
    f = RedactionFilter()
    logger.addFilter(f)
    try:
        with caplog.at_level(logging.INFO, logger="autocut.test_filter"):
            logger.info("leaked: sk-ant-api03-aaaaaaaaaaaaaaaaaaaa")
        assert any("[REDACTED]" in r.message for r in caplog.records)
        assert not any("sk-ant-api03" in r.message for r in caplog.records)
    finally:
        logger.removeFilter(f)


def test_install_redaction_filter_is_idempotent() -> None:
    logger = logging.getLogger("autocut.test_install")
    install_redaction_filter(logger)
    install_redaction_filter(logger)
    assert sum(isinstance(f, RedactionFilter) for f in logger.filters) == 1
