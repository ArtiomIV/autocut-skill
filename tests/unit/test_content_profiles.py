"""Unit tests for ``autocut.content.profiles`` — hint → profile mapping."""

from __future__ import annotations

import pytest

from autocut.content.profiles import (
    HIGHLIGHTS_PROFILE,
    HYBRID_PROFILE,
    TALK_PROFILE,
    ContentProfile,
    profile_for,
)
from autocut.models import ContentHint

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("hint", "expected"),
    [
        (ContentHint.highlights, HIGHLIGHTS_PROFILE),
        (ContentHint.talk, TALK_PROFILE),
        (ContentHint.hybrid, HYBRID_PROFILE),
        (ContentHint.auto, HYBRID_PROFILE),
    ],
)
def test_profile_for_maps_each_hint(hint: ContentHint, expected: ContentProfile) -> None:
    assert profile_for(hint) is expected


def test_profile_for_falls_back_to_hybrid_for_unknown_hint() -> None:
    # Should never happen at runtime (Pydantic enforces enum), but defensive
    # programming: an unmapped hint must not crash the pipeline.
    class _FakeHint:
        pass

    assert profile_for(_FakeHint()) is HYBRID_PROFILE  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Built-in profile shape
# ---------------------------------------------------------------------------


def test_highlights_profile_has_short_min_duration_and_high_keyframe_floor() -> None:
    # Highlight clips are typically 3-20 s and need dense sampling for
    # multi-frame confirmation per the project's conservative-scoring rule.
    assert HIGHLIGHTS_PROFILE.min_duration_sec == 3.0
    assert HIGHLIGHTS_PROFILE.max_duration_sec == 20.0
    assert HIGHLIGHTS_PROFILE.min_keyframes >= 8
    assert HIGHLIGHTS_PROFILE.prompt_template == "highlights"


def test_talk_profile_has_tight_social_clip_window() -> None:
    # A verbal moment needs >= ~8s of setup + payoff, but the upper bound is
    # kept TIGHT (30s) so a clip does not sprawl into the next speaker's filler.
    # Talk targets social platforms where short clips win (user feedback 2026-05-30).
    assert TALK_PROFILE.min_duration_sec >= 8.0
    assert TALK_PROFILE.max_duration_sec == 30.0
    assert TALK_PROFILE.prompt_template == "talk"


def test_hybrid_profile_sits_between_highlights_and_talk() -> None:
    # Min duration still rises highlights -> hybrid -> talk (talk needs the most
    # lead-in). Max duration no longer follows that order: talk is deliberately
    # capped tight for social, so hybrid is now the widest catch-all window.
    assert (
        HIGHLIGHTS_PROFILE.min_duration_sec
        <= HYBRID_PROFILE.min_duration_sec
        <= TALK_PROFILE.min_duration_sec
    )
    assert (
        HIGHLIGHTS_PROFILE.max_duration_sec
        <= TALK_PROFILE.max_duration_sec
        <= HYBRID_PROFILE.max_duration_sec
    )
    assert HYBRID_PROFILE.prompt_template == "hybrid"


def test_all_builtin_profiles_default_to_zero_padding() -> None:
    # Per user feedback the cutter should produce tight cuts by default.
    # Profiles ship with 0 padding; the plumbing exists to flip the number later.
    for profile in (HIGHLIGHTS_PROFILE, TALK_PROFILE, HYBRID_PROFILE):
        assert profile.pre_roll_sec == 0.0
        assert profile.post_roll_sec == 0.0


def test_all_builtin_profiles_have_valid_sampling_strategy() -> None:
    for profile in (HIGHLIGHTS_PROFILE, TALK_PROFILE, HYBRID_PROFILE):
        assert profile.sampling_strategy in {"scene", "uniform", "hybrid"}
