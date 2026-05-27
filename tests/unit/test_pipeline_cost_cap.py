"""Unit tests for ``autocut.pipeline`` — cost-cap gate behaviour.

These tests stub out ffmpeg/ffprobe/PySceneDetect by monkey-patching the
helpers the pipeline imports, so we can run them headlessly. The focus is
purely on the cost-cap branch.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import ClassVar

import pytest

from autocut.config import AutoCutConfig
from autocut.models import (
    AnalysisHints,
    Category,
    Clip,
    ClipPlan,
    ClipPlanMetadata,
    ContentHint,
    DetectionResult,
    Keyframe,
    Scene,
    VideoMetadata,
)
from autocut.pipeline import CostCapExceeded, run_analysis
from autocut.video.frame_sampler import FrameSpec
from autocut.vlm import CostEstimate, VLMProvider

# ---------------------------------------------------------------------------
# Stub provider — we control estimate_cost() so we can drive both branches.
# ---------------------------------------------------------------------------


class _StubProvider(VLMProvider):
    name: ClassVar[str] = "stub"
    model: ClassVar[str] = "stub-model"

    def __init__(self, *, cost_usd: float) -> None:
        self._cost_usd = cost_usd
        self.analyze_called = False

    async def analyze(
        self,
        keyframes: list[Keyframe],
        hints: AnalysisHints,
        *,
        video_id: str,
        duration_sec: float,
        timeout_sec: int = 300,
    ) -> ClipPlan:
        del keyframes, hints, timeout_sec
        self.analyze_called = True
        return ClipPlan(
            video_id=video_id,
            duration_sec=duration_sec,
            clips=[
                Clip(
                    id="c1",
                    start=timedelta(seconds=0),
                    end=timedelta(seconds=15),
                    category=Category.highlight,
                    description="ok",
                    score=8,
                    rationale="because",
                )
            ],
            metadata=ClipPlanMetadata(vlm_provider=self.name, vlm_model=self.model),
        )

    async def detect_content(
        self,
        keyframes: list[Keyframe],
        audio_description: str,
        *,
        video_id: str,
        duration_sec: float,
        timeout_sec: int = 120,
        transcript_text: str | None = None,
        audio_clip_path: Path | None = None,
        video_clip_paths: list[Path] | None = None,
    ) -> DetectionResult:
        del (
            keyframes,
            audio_description,
            video_id,
            duration_sec,
            timeout_sec,
            transcript_text,
            audio_clip_path,
            video_clip_paths,
        )
        return DetectionResult(
            content_hint=ContentHint.other,
            confidence=0.0,
            reasoning="stub provider — cost-cap test fixture",
        )

    def estimate_cost(self, n_keyframes: int) -> CostEstimate:
        return CostEstimate(
            provider=self.name,
            model=self.model,
            n_input_images=n_keyframes,
            estimated_input_tokens=0,
            estimated_output_tokens=0,
            estimated_total_usd=self._cost_usd,
        )

    def health_check(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Patch helpers — replace the video stack with deterministic stubs.
# ---------------------------------------------------------------------------


@pytest.fixture
def _stub_video_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_meta = VideoMetadata(
        path=Path("/fake/video.mp4"),
        duration_sec=60.0,
        width=640,
        height=480,
        fps=30.0,
        video_codec="h264",
        audio_codec="aac",
        container="mp4",
        size_bytes=1024,
    )
    fake_scenes = [Scene(index=0, start=timedelta(0), end=timedelta(seconds=60))]
    fake_specs = [FrameSpec(timestamp=timedelta(seconds=t), scene_index=0) for t in (10, 30, 50)]
    fake_keyframes = [
        Keyframe(scene_index=0, timestamp=spec.timestamp, path=Path(f"/fake/kf_{i}.jpg"))
        for i, spec in enumerate(fake_specs)
    ]

    monkeypatch.setattr("autocut.pipeline.probe_video", lambda _path: fake_meta)
    monkeypatch.setattr("autocut.pipeline.detect_scenes", lambda _v, threshold=27.0: fake_scenes)
    monkeypatch.setattr(
        "autocut.pipeline.build_sampler",
        lambda strategy, scenes, duration, **_kwargs: fake_specs,
    )
    monkeypatch.setattr(
        "autocut.pipeline.extract_keyframes",
        lambda video, specs, out_dir, long_edge_px: fake_keyframes,
    )

    # Phase E: bypass auto-detect entirely so cost-cap tests stay focused on
    # the cost gate. Return a low-confidence ``other`` so the pipeline falls
    # back to HYBRID without running ffmpeg on a fake path.
    async def _stub_auto_detect(*_args: object, **_kwargs: object) -> DetectionResult:
        return DetectionResult(
            content_hint=ContentHint.other,
            confidence=0.0,
            reasoning="cost-cap test fixture bypassing detection",
        )

    monkeypatch.setattr("autocut.pipeline._run_auto_detect", _stub_auto_detect)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_cost_cap_raises_when_estimate_exceeds_cap_and_no_confirm_hook(
    _stub_video_stack: None,
    tmp_path: Path,
) -> None:
    config = AutoCutConfig()
    config.security.cost_cap_usd = 0.10
    provider = _StubProvider(cost_usd=5.0)

    with pytest.raises(CostCapExceeded, match="exceeds cap"):
        await run_analysis(
            Path("/fake/video.mp4"),
            provider,
            config=config,
            output_root=tmp_path / "CLIPS",
            write_outputs=False,
        )
    assert provider.analyze_called is False


async def test_cost_cap_raises_when_confirm_hook_denies(
    _stub_video_stack: None,
    tmp_path: Path,
) -> None:
    config = AutoCutConfig()
    config.security.cost_cap_usd = 0.10
    provider = _StubProvider(cost_usd=5.0)
    seen: dict[str, float] = {}

    def deny(estimate: CostEstimate, cap: float) -> bool:
        seen["estimate"] = estimate.estimated_total_usd
        seen["cap"] = cap
        return False

    with pytest.raises(CostCapExceeded):
        await run_analysis(
            Path("/fake/video.mp4"),
            provider,
            config=config,
            output_root=tmp_path / "CLIPS",
            write_outputs=False,
            confirm_cost=deny,
        )
    assert seen == {"estimate": 5.0, "cap": 0.10}
    assert provider.analyze_called is False


async def test_cost_cap_proceeds_when_confirm_hook_accepts(
    _stub_video_stack: None,
    tmp_path: Path,
) -> None:
    config = AutoCutConfig()
    config.security.cost_cap_usd = 0.10
    provider = _StubProvider(cost_usd=5.0)

    result = await run_analysis(
        Path("/fake/video.mp4"),
        provider,
        config=config,
        output_root=tmp_path / "CLIPS",
        write_outputs=False,
        confirm_cost=lambda est, cap: True,
    )
    assert provider.analyze_called is True
    assert len(result.plan.clips) == 1


async def test_free_provider_skips_cost_cap(
    _stub_video_stack: None,
    tmp_path: Path,
) -> None:
    # cost_usd=0 means is_free=True; the cap branch must be bypassed even when
    # the cap is set absurdly low.
    config = AutoCutConfig()
    config.security.cost_cap_usd = 0.0
    provider = _StubProvider(cost_usd=0.0)

    result = await run_analysis(
        Path("/fake/video.mp4"),
        provider,
        config=config,
        output_root=tmp_path / "CLIPS",
        write_outputs=False,
    )
    assert provider.analyze_called is True
    assert result.dispatch is None  # write_outputs=False
