"""Unit tests for Continuous Benchmarking, Telemetry & Auditing Framework (Spec 15)."""

from pathlib import Path
import time
import pytest

from stream_fusion.monitoring.audit import (
    AuditRunRecord,
    AuditStore,
    RegressionComparator,
    StageTelemetry,
    get_git_info,
)
from stream_fusion.monitoring.telemetry import StageTimer, TelemetryCollector


def test_stage_timer_and_collector():
    collector = TelemetryCollector(run_id="test_run_001")

    with collector.stage("demuxing"):
        time.sleep(0.05)

    with collector.stage("whisper"):
        time.sleep(0.05)

    collector.record_quality_metric("ocr_blocks", 15)
    collector.record_quality_metric("chat_messages", 250)

    assert "demuxing" in collector.stages
    assert "whisper" in collector.stages
    assert collector.stages["demuxing"]["duration_sec"] >= 0.04
    assert collector.stages["demuxing"]["status"] == "SUCCESS"
    assert collector.quality_metrics["ocr_blocks"] == 15
    assert collector.get_total_duration() >= 0.08


def test_audit_store_persistence(tmp_path: Path):
    db_file = tmp_path / "audit.db"
    store = AuditStore(db_path=db_file)

    stages = {
        "demux": StageTelemetry(
            stage_name="demux",
            duration_sec=2.5,
            start_time_iso="2026-09-27T16:00:00Z",
            end_time_iso="2026-09-27T16:00:02Z",
            ram_mb_peak=450.0,
            vram_mb_peak=0.0,
        )
    }

    record = AuditRunRecord(
        run_id="run_baseline",
        git_commit="abc1234",
        git_dirty=False,
        timestamp="2026-09-27T16:00:00Z",
        stream_id="test_stream",
        media_duration_sec=60.0,
        total_pipeline_duration_sec=10.0,
        overall_real_time_factor=0.166,
        stages=stages,
        quality_metrics={"chat_messages": 100},
    )

    # Save as baseline
    store.record_run(record, set_as_baseline=True)

    loaded_baseline = store.get_baseline()
    assert loaded_baseline is not None
    assert loaded_baseline.run_id == "run_baseline"
    assert loaded_baseline.total_pipeline_duration_sec == 10.0

    recent = store.get_recent_runs(5)
    assert len(recent) == 1
    assert recent[0].run_id == "run_baseline"


def test_regression_comparator_alerts():
    comparator = RegressionComparator()

    base_stages = {
        "transcription": StageTelemetry(
            stage_name="transcription",
            duration_sec=10.0,
            start_time_iso="", end_time_iso="",
            ram_mb_peak=1000.0, vram_mb_peak=2000.0,
        )
    }
    baseline = AuditRunRecord(
        run_id="base",
        stream_id="stream_01",
        media_duration_sec=60.0,
        total_pipeline_duration_sec=20.0,
        overall_real_time_factor=0.33,
        stages=base_stages,
        quality_metrics={"ocr_blocks": 50, "claims": 10},
    )

    # 1. Regressed run: total duration +50%, transcription stage +60%, ram +50%, ocr dropped -30%
    reg_stages = {
        "transcription": StageTelemetry(
            stage_name="transcription",
            duration_sec=16.0,  # +60%
            start_time_iso="", end_time_iso="",
            ram_mb_peak=1500.0,  # +50%
            vram_mb_peak=2000.0,
        )
    }
    current = AuditRunRecord(
        run_id="regressed_run",
        stream_id="stream_01",
        media_duration_sec=60.0,
        total_pipeline_duration_sec=30.0,  # +50%
        overall_real_time_factor=0.50,
        stages=reg_stages,
        quality_metrics={"ocr_blocks": 30, "claims": 10},  # -40% ocr
    )

    alerts, recommendations = comparator.compare_runs(current, baseline)
    assert len(alerts) >= 3
    metric_names = [a.metric_name for a in alerts]
    assert "total_pipeline_duration_sec" in metric_names
    assert "stage_duration_sec" in metric_names
    assert "ram_mb_peak" in metric_names
    assert "ocr_blocks" in metric_names
    assert len(recommendations) > 0


def test_git_info_retrieval():
    commit, dirty = get_git_info()
    assert commit != ""
    assert isinstance(dirty, bool)
