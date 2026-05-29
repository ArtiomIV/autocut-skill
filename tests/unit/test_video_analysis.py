"""Unit tests for the L2 video-analysis engine (ffmpeg stubbed out)."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from autocut import video_analysis
from autocut.models import AnalysisHints, Clip, ClipPlan, ClipPlanMetadata
from autocut.video_analysis import VideoAnalysisError, analyze_video


def _one_clip_plan(video_id: str, duration: float, *, cost_usd: float | None = None) -> ClipPlan:
    """A plan with a single clip at a fixed CLIP-RELATIVE 5-10s window."""
    return ClipPlan(
        video_id=video_id,
        duration_sec=duration,
        clips=[
            Clip(
                id="c",
                start=timedelta(seconds=5),
                end=timedelta(seconds=10),
                category="highlight",
                description="stub clip",
                score=8,
                rationale="stub",
                tags=["t"],
            )
        ],
        metadata=ClipPlanMetadata(vlm_provider="openrouter", vlm_model="m", cost_usd=cost_usd),
    )


class _StubProvider:
    """Records each call and returns a deterministic clip-relative plan."""

    def __init__(self, cost_per_call: float | None = None) -> None:
        self.call_durations: list[float] = []
        self._cost_per_call = cost_per_call

    async def analyze_video_clip(
        self,
        clip_path: Path,
        hints: AnalysisHints,
        *,
        video_id: str,
        clip_duration_sec: float,
        timeout_sec: int = 300,
    ) -> ClipPlan:
        self.call_durations.append(clip_duration_sec)
        return _one_clip_plan(video_id, clip_duration_sec, cost_usd=self._cost_per_call)


@pytest.fixture
def _stub_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the ffmpeg-backed helpers so the engine runs without ffmpeg."""
    monkeypatch.setattr(video_analysis, "compress_for_vlm", lambda src, out, **kw: out)
    monkeypatch.setattr(
        video_analysis, "cut_clip", lambda video, request, **kw: request.output_path
    )


@pytest.mark.asyncio
async def test_short_video_is_single_pass(_stub_ffmpeg: None, tmp_path: Path) -> None:
    provider = _StubProvider()
    hints = AnalysisHints(min_duration_sec=3, max_duration_sec=20)
    plan = await analyze_video(
        tmp_path / "src.mp4",
        hints,
        provider,
        video_id="vid",
        duration_sec=100.0,  # < 5 min -> one pass
        cost_cap_usd=1.0,
        work_dir=tmp_path,
    )
    assert len(provider.call_durations) == 1
    assert len(plan.clips) == 1
    # single pass: offset 0, so timestamps stay clip-relative (5-10s).
    assert plan.clips[0].start == timedelta(seconds=5)
    assert plan.clips[0].end == timedelta(seconds=10)


@pytest.mark.asyncio
async def test_long_video_batches_and_offsets(_stub_ffmpeg: None, tmp_path: Path) -> None:
    provider = _StubProvider()
    hints = AnalysisHints(min_duration_sec=3, max_duration_sec=20)  # overlap = 10s
    plan = await analyze_video(
        tmp_path / "src.mp4",
        hints,
        provider,
        video_id="vid",
        duration_sec=700.0,  # -> batches [0-310], [300-610], [600-700]
        cost_cap_usd=1.0,
        work_dir=tmp_path,
    )
    assert len(provider.call_durations) == 3
    # Each stub clip is 5-10s relative; offset by each batch start (0/300/600).
    starts = sorted(c.start.total_seconds() for c in plan.clips)
    assert starts == [5.0, 305.0, 605.0]


@pytest.mark.asyncio
async def test_real_cost_is_summed_across_batches(_stub_ffmpeg: None, tmp_path: Path) -> None:
    provider = _StubProvider(cost_per_call=0.02)
    hints = AnalysisHints(min_duration_sec=3, max_duration_sec=20)
    plan = await analyze_video(
        tmp_path / "src.mp4",
        hints,
        provider,
        video_id="vid",
        duration_sec=700.0,  # 3 batches -> 3 * 0.02
        cost_cap_usd=1.0,
        work_dir=tmp_path,
    )
    assert plan.metadata.cost_usd == pytest.approx(0.06)


@pytest.mark.asyncio
async def test_cost_cap_aborts_without_confirmation(_stub_ffmpeg: None, tmp_path: Path) -> None:
    provider = _StubProvider()
    hints = AnalysisHints(min_duration_sec=3, max_duration_sec=20)
    with pytest.raises(VideoAnalysisError, match="exceeds cap"):
        await analyze_video(
            tmp_path / "src.mp4",
            hints,
            provider,
            video_id="vid",
            duration_sec=10_000.0,  # ~$5 estimate, over the $1 cap
            cost_cap_usd=1.0,
            confirm_cost=None,
        )
    assert provider.call_durations == []  # aborted before any call
