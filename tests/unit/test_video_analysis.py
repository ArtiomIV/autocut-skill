"""Unit tests for the L2 video-analysis engine (ffmpeg stubbed out)."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from autocut import video_analysis
from autocut.models import AnalysisHints, Clip, ClipPlan, ClipPlanMetadata, Keyframe
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

    def _fake_extract(video: Path, specs: list, out_dir: Path, **kw: object) -> list[Keyframe]:
        # Mirror the requested specs as Keyframe records without touching ffmpeg,
        # so the fine-frames pass yields one keyframe per sampled timestamp.
        return [
            Keyframe(scene_index=None, timestamp=s.timestamp, path=Path(out_dir) / f"f{i}.jpg")
            for i, s in enumerate(specs)
        ]

    monkeypatch.setattr(video_analysis, "extract_keyframes", _fake_extract)


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


# ---------------------------------------------------------------------------
# Two-pass (coarse -> fine)
# ---------------------------------------------------------------------------


def _tight_fine_plan(video_id: str, duration: float) -> ClipPlan:
    """One tight clip 5-12s into a (fine) candidate segment."""
    return ClipPlan(
        video_id=video_id,
        duration_sec=duration,
        clips=[
            Clip(
                id="f",
                start=timedelta(seconds=5),
                end=timedelta(seconds=12),
                category="highlight",
                description="tight",
                score=9,
                rationale="fine",
            )
        ],
        metadata=ClipPlanMetadata(vlm_provider="openrouter", vlm_model="m"),
    )


class _CoarseThenFineProvider:
    """Coarse (video) returns 2 wide regions; the fine STILLS pass tightens each.

    The coarse pass goes through ``analyze_video_clip`` (video payload); the
    two-pass fine pass goes through ``analyze`` (dense labelled stills), so the
    test can assert the locate→refine flow AND that the fine pass uses frames,
    not a video clip, without a real model.
    """

    def __init__(self) -> None:
        self.coarse_calls = 0
        self.video_clip_calls = 0
        self.fine_durations: list[float] = []
        self.fine_keyframe_counts: list[int] = []

    async def analyze_video_clip(
        self,
        clip_path: Path,
        hints: AnalysisHints,
        *,
        video_id: str,
        clip_duration_sec: float,
        timeout_sec: int = 300,
    ) -> ClipPlan:
        self.video_clip_calls += 1
        if hints.prompt_template == "coarse":
            self.coarse_calls += 1
            # Two coarse candidate regions, clip-relative to the whole source.
            return ClipPlan(
                video_id=video_id,
                duration_sec=clip_duration_sec,
                clips=[
                    Clip(
                        id="r1",
                        start=timedelta(seconds=30),
                        end=timedelta(seconds=50),
                        category="highlight",
                        description="region 1",
                        score=8,
                        rationale="coarse",
                    ),
                    Clip(
                        id="r2",
                        start=timedelta(seconds=120),
                        end=timedelta(seconds=140),
                        category="highlight",
                        description="region 2",
                        score=7,
                        rationale="coarse",
                    ),
                ],
                metadata=ClipPlanMetadata(vlm_provider="openrouter", vlm_model="m"),
            )
        # Single-pass fallback (no two-pass): one neutral clip.
        return _one_clip_plan(video_id, clip_duration_sec)

    async def analyze(
        self,
        keyframes: list[Keyframe],
        hints: AnalysisHints,
        *,
        video_id: str,
        duration_sec: float,
        timeout_sec: int = 300,
    ) -> ClipPlan:
        # The fine pass: dense stills + their timestamps.
        self.fine_durations.append(duration_sec)
        self.fine_keyframe_counts.append(len(keyframes))
        return _tight_fine_plan(video_id, duration_sec)


@pytest.mark.asyncio
async def test_two_pass_locates_then_refines(_stub_ffmpeg: None, tmp_path: Path) -> None:
    provider = _CoarseThenFineProvider()
    hints = AnalysisHints(min_duration_sec=3, max_duration_sec=20)
    plan = await analyze_video(
        tmp_path / "src.mp4",
        hints,
        provider,
        video_id="vid",
        duration_sec=200.0,  # > 60s trigger, single coarse batch (< 5 min)
        cost_cap_usd=1.0,
        work_dir=tmp_path,
        two_pass=True,
    )
    # One coarse locate call, then one fine call per candidate region (2).
    assert provider.coarse_calls == 1
    assert len(provider.fine_durations) == 2
    # Final clips are re-based to absolute with the stills fine pad ±10s:
    # candidate 1 padded start = max(0, 30-10) = 20, plus the fine clip's 5s
    # relative start -> 25s; candidate 2 -> max(0, 120-10) + 5 = 115s.
    starts = sorted(c.start.total_seconds() for c in plan.clips)
    assert starts == [25.0, 115.0]


class _CoarseThenEmptyProvider:
    """Pass 1 finds one region; pass 2 rejects it (returns NO clip).

    Models the high-recall coarse pass over-including a region that the precise
    fine pass then finds worthless — the candidate must yield zero output.
    """

    def __init__(self) -> None:
        self.fine_calls = 0

    async def analyze_video_clip(
        self,
        clip_path: Path,
        hints: AnalysisHints,
        *,
        video_id: str,
        clip_duration_sec: float,
        timeout_sec: int = 300,
    ) -> ClipPlan:
        return ClipPlan(
            video_id=video_id,
            duration_sec=clip_duration_sec,
            clips=[
                Clip(
                    id="r1",
                    start=timedelta(seconds=40),
                    end=timedelta(seconds=60),
                    category="highlight",
                    description="maybe something",
                    score=6,
                    rationale="coarse over-include",
                )
            ],
            metadata=ClipPlanMetadata(vlm_provider="openrouter", vlm_model="m"),
        )

    async def analyze(
        self,
        keyframes: list[Keyframe],
        hints: AnalysisHints,
        *,
        video_id: str,
        duration_sec: float,
        timeout_sec: int = 300,
    ) -> ClipPlan:
        # Fine pass: nothing valid here -> empty.
        self.fine_calls += 1
        return ClipPlan(
            video_id=video_id,
            duration_sec=duration_sec,
            clips=[],
            metadata=ClipPlanMetadata(vlm_provider="openrouter", vlm_model="m"),
        )


@pytest.mark.asyncio
async def test_two_pass_rejected_candidate_yields_no_clip(
    _stub_ffmpeg: None, tmp_path: Path
) -> None:
    provider = _CoarseThenEmptyProvider()
    hints = AnalysisHints(min_duration_sec=3, max_duration_sec=20)
    plan = await analyze_video(
        tmp_path / "src.mp4",
        hints,
        provider,
        video_id="vid",
        duration_sec=200.0,
        cost_cap_usd=1.0,
        work_dir=tmp_path,
        two_pass=True,
    )
    # The fine pass was consulted and rejected the candidate -> zero clips out.
    assert provider.fine_calls == 1
    assert plan.clips == []


class _CoarseThenFlakyProvider:
    """Coarse finds 2 regions; the fine pass raises for the FIRST candidate."""

    def __init__(self) -> None:
        self.fine_calls = 0

    async def analyze_video_clip(
        self,
        clip_path: Path,
        hints: AnalysisHints,
        *,
        video_id: str,
        clip_duration_sec: float,
        timeout_sec: int = 300,
    ) -> ClipPlan:
        return ClipPlan(
            video_id=video_id,
            duration_sec=clip_duration_sec,
            clips=[
                Clip(
                    id=f"r{i}",
                    start=timedelta(seconds=30 + i * 60),
                    end=timedelta(seconds=45 + i * 60),
                    category="highlight",
                    description=f"region {i}",
                    score=8,
                    rationale="coarse",
                )
                for i in range(2)
            ],
            metadata=ClipPlanMetadata(vlm_provider="openrouter", vlm_model="m"),
        )

    async def analyze(
        self,
        keyframes: list[Keyframe],
        hints: AnalysisHints,
        *,
        video_id: str,
        duration_sec: float,
        timeout_sec: int = 300,
    ) -> ClipPlan:
        self.fine_calls += 1
        if self.fine_calls == 1:
            raise TimeoutError("simulated provider timeout on candidate 1")
        return _one_clip_plan(video_id, duration_sec)


@pytest.mark.asyncio
async def test_two_pass_skips_failing_candidate(_stub_ffmpeg: None, tmp_path: Path) -> None:
    # One candidate's fine call times out; the run survives with the other.
    provider = _CoarseThenFlakyProvider()
    hints = AnalysisHints(min_duration_sec=3, max_duration_sec=20)
    plan = await analyze_video(
        tmp_path / "src.mp4",
        hints,
        provider,
        video_id="vid",
        duration_sec=200.0,
        cost_cap_usd=1.0,
        work_dir=tmp_path,
        two_pass=True,
    )
    assert provider.fine_calls == 2  # both attempted
    assert len(plan.clips) == 1  # one failed (skipped), one survived


class _AlwaysFailsProvider:
    async def analyze_video_clip(
        self,
        clip_path: Path,
        hints: AnalysisHints,
        *,
        video_id: str,
        clip_duration_sec: float,
        timeout_sec: int = 300,
    ) -> ClipPlan:
        raise TimeoutError("provider down")


@pytest.mark.asyncio
async def test_all_batches_failing_raises(_stub_ffmpeg: None, tmp_path: Path) -> None:
    # If every batch fails (systemic provider/network error), surface it.
    provider = _AlwaysFailsProvider()
    hints = AnalysisHints(min_duration_sec=3, max_duration_sec=20)
    with pytest.raises(VideoAnalysisError, match=r"all .* batch"):
        await analyze_video(
            tmp_path / "src.mp4",
            hints,
            provider,
            video_id="vid",
            duration_sec=100.0,  # single pass, one batch -> it fails -> raise
            cost_cap_usd=1.0,
            work_dir=tmp_path,
        )


@pytest.mark.asyncio
async def test_two_pass_off_by_default_uses_single_pass(_stub_ffmpeg: None, tmp_path: Path) -> None:
    # Without two_pass, a 200s video is a single direct pass (no coarse call).
    provider = _CoarseThenFineProvider()
    hints = AnalysisHints(min_duration_sec=3, max_duration_sec=20)
    await analyze_video(
        tmp_path / "src.mp4",
        hints,
        provider,
        video_id="vid",
        duration_sec=200.0,
        cost_cap_usd=1.0,
        work_dir=tmp_path,
    )
    assert provider.coarse_calls == 0


@pytest.mark.asyncio
async def test_two_pass_long_video_stays_under_default_cap(
    _stub_ffmpeg: None, tmp_path: Path
) -> None:
    # Regression: the cost estimate must be calibrated so a normal long (20-min)
    # two-pass run does NOT spuriously trip the default $1 cap now that two-pass
    # is the default for long videos. 1200s * 0.00025 * 2 = $0.60 < $1.
    provider = _CoarseThenFineProvider()
    hints = AnalysisHints(min_duration_sec=3, max_duration_sec=20)
    plan = await analyze_video(
        tmp_path / "src.mp4",
        hints,
        provider,
        video_id="vid",
        duration_sec=1200.0,
        cost_cap_usd=1.0,
        confirm_cost=None,  # would abort if the gate tripped
        work_dir=tmp_path,
        two_pass=True,
    )
    assert provider.coarse_calls >= 1
    assert len(plan.clips) >= 1


@pytest.mark.asyncio
async def test_two_pass_skipped_for_short_video(_stub_ffmpeg: None, tmp_path: Path) -> None:
    # two_pass requested but the source is <= 60s -> stays single pass.
    provider = _CoarseThenFineProvider()
    hints = AnalysisHints(min_duration_sec=3, max_duration_sec=20)
    await analyze_video(
        tmp_path / "src.mp4",
        hints,
        provider,
        video_id="vid",
        duration_sec=40.0,
        cost_cap_usd=1.0,
        work_dir=tmp_path,
        two_pass=True,
    )
    assert provider.coarse_calls == 0


@pytest.mark.asyncio
async def test_two_pass_fine_uses_dense_stills_not_video(
    _stub_ffmpeg: None, tmp_path: Path
) -> None:
    # The fine pass must go through the STILLS path (provider.analyze), NOT a
    # second video clip, and feed ~2 FPS frames over each padded candidate window.
    provider = _CoarseThenFineProvider()
    hints = AnalysisHints(min_duration_sec=3, max_duration_sec=20)
    await analyze_video(
        tmp_path / "src.mp4",
        hints,
        provider,
        video_id="vid",
        duration_sec=200.0,
        cost_cap_usd=1.0,
        work_dir=tmp_path,
        two_pass=True,
    )
    # Only the coarse locate touched the video payload; both fine passes used stills.
    assert provider.video_clip_calls == 1
    assert provider.coarse_calls == 1
    assert len(provider.fine_durations) == 2
    # Each candidate is a 20s region padded ±10s -> a 40s window. At 2 FPS that
    # would be 80 frames, over the cap, so the interval widens and the frame count
    # is clamped to _FINE_FRAME_MAX_FRAMES — the payload never blows up.
    for n in provider.fine_keyframe_counts:
        assert 0 < n <= video_analysis._FINE_FRAME_MAX_FRAMES
    assert max(provider.fine_keyframe_counts) == video_analysis._FINE_FRAME_MAX_FRAMES


class _ShortCoarseFineProvider:
    """One in-bounds coarse region (video) + a stills fine pass, for short clips."""

    def __init__(self) -> None:
        self.coarse_calls = 0
        self.fine_calls = 0
        self.fine_keyframe_counts: list[int] = []

    async def analyze_video_clip(
        self,
        clip_path: Path,
        hints: AnalysisHints,
        *,
        video_id: str,
        clip_duration_sec: float,
        timeout_sec: int = 300,
    ) -> ClipPlan:
        self.coarse_calls += 1
        return ClipPlan(
            video_id=video_id,
            duration_sec=clip_duration_sec,
            clips=[
                Clip(
                    id="r1",
                    start=timedelta(seconds=10),
                    end=timedelta(seconds=13),
                    category="highlight",
                    description="region 1",
                    score=8,
                    rationale="coarse",
                )
            ],
            metadata=ClipPlanMetadata(vlm_provider="openrouter", vlm_model="m"),
        )

    async def analyze(
        self,
        keyframes: list[Keyframe],
        hints: AnalysisHints,
        *,
        video_id: str,
        duration_sec: float,
        timeout_sec: int = 300,
    ) -> ClipPlan:
        self.fine_calls += 1
        self.fine_keyframe_counts.append(len(keyframes))
        return _tight_fine_plan(video_id, duration_sec)


@pytest.mark.asyncio
async def test_force_two_pass_runs_on_short_video(_stub_ffmpeg: None, tmp_path: Path) -> None:
    # force_two_pass lifts the >60s gate -> two-pass runs on a 40s clip too.
    provider = _ShortCoarseFineProvider()
    hints = AnalysisHints(min_duration_sec=3, max_duration_sec=20)
    plan = await analyze_video(
        tmp_path / "src.mp4",
        hints,
        provider,
        video_id="vid",
        duration_sec=40.0,
        cost_cap_usd=1.0,
        work_dir=tmp_path,
        two_pass=True,
        force_two_pass=True,
    )
    assert provider.coarse_calls == 1
    assert provider.fine_calls == 1
    assert len(plan.clips) >= 1
    # Small window (3s region padded ±10s -> ~23s) stays under the cap, so it keeps
    # the full 2 FPS density (interval not widened): ~2 frames per second.
    (n,) = provider.fine_keyframe_counts
    assert n <= video_analysis._FINE_FRAME_MAX_FRAMES
    assert n == pytest.approx(23 * 2, abs=2)
