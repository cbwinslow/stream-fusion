"""Tests for Dynamic Moment Discovery & Significance-Gated Short Production (Spec 31)."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from stream_fusion.models.schemas import (
    AudioSegment,
    ChatMessage,
    DynamicMomentThresholds,
    FusionSlice,
    StreamAnalysisResult,
)
from stream_fusion.production.director import DirectorAgent
from stream_fusion.production.orchestrator import ShortProductionOrchestrator


@pytest.fixture
def mock_stream_analysis():
    """Builds a synthetic multi-hour analysis result with multiple distinct highlight spikes."""
    highlights = [
        {"timestamp_sec": 100.0, "score": 0.85},  # Spike 1
        {"timestamp_sec": 110.0, "score": 0.90},  # Close to Spike 1 (< 60s, should merge)
        {"timestamp_sec": 300.0, "score": 0.75},  # Spike 2
        {"timestamp_sec": 600.0, "score": 0.80},  # Spike 3
        {"timestamp_sec": 900.0, "score": 0.70},  # Spike 4
        {"timestamp_sec": 1200.0, "score": 0.95}, # Spike 5
        {"timestamp_sec": 1500.0, "score": 0.68}, # Spike 6
        {"timestamp_sec": 1800.0, "score": 0.88}, # Spike 7
        {"timestamp_sec": 2100.0, "score": 0.72}, # Spike 8
        {"timestamp_sec": 2400.0, "score": 0.40}, # Below threshold (< 0.65)
    ]
    return StreamAnalysisResult(
        stream_id="test-stream-dynamic",
        duration_sec=3600.0,
        slices=[],
        fusion_slices=[],
        highlights=highlights,
        summary="Synthetic test stream with 8 qualifying highlights",
    )


def test_legacy_top_k_mode(mock_stream_analysis):
    """Verifies backward compatibility: top_k truncates yield when dynamic=False."""
    director = DirectorAgent()
    candidates = director.select_candidates(
        analysis=mock_stream_analysis,
        top_k=3,
        dynamic=False,
    )
    assert len(candidates) == 3
    # Verify sorted by score
    assert candidates[0].highlight_score >= candidates[1].highlight_score


def test_dynamic_moment_discovery_captures_all_qualifying(mock_stream_analysis):
    """Verifies that dynamic discovery captures all 7 distinct moments exceeding threshold."""
    director = DirectorAgent()
    thresholds = DynamicMomentThresholds(
        min_highlight_score=0.65,
        min_separation_sec=60.0,
        safety_max_shorts=50,
    )
    candidates = director.select_candidates(
        analysis=mock_stream_analysis,
        dynamic=True,
        thresholds=thresholds,
    )
    # Total qualifying: 100/110 (merged into 1), 300, 600, 900, 1200, 1500, 1800, 2100 = 8 distinct
    assert len(candidates) == 8
    # 2400s had score 0.40, which is below 0.65 and must NOT be included
    timestamps = [c.peak_timestamp_sec for c in candidates]
    assert 2400.0 not in timestamps


def test_dynamic_temporal_separation(mock_stream_analysis):
    """Verifies that adjacent moments within min_separation_sec are properly merged."""
    director = DirectorAgent()
    thresholds = DynamicMomentThresholds(
        min_highlight_score=0.65,
        min_separation_sec=120.0,  # 2 minute separation
        safety_max_shorts=50,
    )
    candidates = director.select_candidates(
        analysis=mock_stream_analysis,
        dynamic=True,
        thresholds=thresholds,
    )
    # Check that no two candidates are closer than 120 seconds
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            assert abs(candidates[i].peak_timestamp_sec - candidates[j].peak_timestamp_sec) >= 120.0


def test_safety_max_shorts_capping():
    """Verifies that safety_max_shorts prevents runaway candidate generation."""
    highlights = [{"timestamp_sec": float(i * 100), "score": 0.9} for i in range(40)]
    analysis = StreamAnalysisResult(
        stream_id="test-runaway",
        duration_sec=5000.0,
        highlights=highlights,
        slices=[],
    )
    director = DirectorAgent()
    thresholds = DynamicMomentThresholds(
        min_highlight_score=0.65,
        min_separation_sec=60.0,
        safety_max_shorts=10,
    )
    candidates = director.select_candidates(
        analysis=analysis,
        dynamic=True,
        thresholds=thresholds,
    )
    assert len(candidates) == 10


def test_short_orchestrator_dynamic_production(tmp_path, mock_stream_analysis):
    """Verifies ShortProductionOrchestrator runs end-to-end with dynamic=True."""
    orchestrator = ShortProductionOrchestrator()
    thresholds = DynamicMomentThresholds(
        min_highlight_score=0.70,
        min_separation_sec=60.0,
        safety_max_shorts=15,
    )
    packages_and_envs = orchestrator.produce_shorts(
        analysis=mock_stream_analysis,
        output_dir=tmp_path / "shorts",
        dry_run=True,
        dynamic=True,
        thresholds=thresholds,
    )
    # Qualifying >= 0.70: 110s (0.90), 300s (0.75), 600s (0.80), 900s (0.70), 1200s (0.95), 1800s (0.88), 2100s (0.72) = 7
    assert len(packages_and_envs) == 7
    for pkg, env in packages_and_envs:
        assert pkg.package_id is not None
        assert env.event_type.value == "SHORT_PRODUCED"
