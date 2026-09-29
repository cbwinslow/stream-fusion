"""Unit tests for the StreamFusion Typer CLI."""

from pathlib import Path
from typer.testing import CliRunner

from stream_fusion.cli import app

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Multimodal Livestream VOD & Chat Grounding and Analysis CLI" in result.output
    assert "process" in result.output
    assert "clip" in result.output
    assert "demo" in result.output


def test_cli_process_help():
    result = runner.invoke(app, ["process", "--help"])
    assert result.exit_code == 0
    assert "--latency-offset" in result.output
    assert "--auto-latency" in result.output
    assert "--chunk-duration" in result.output


def test_cli_clip_help():
    result = runner.invoke(app, ["clip", "--help"])
    assert result.exit_code == 0
    assert "--start" in result.output
    assert "--end" in result.output
    assert "--out" in result.output


def test_cli_demo_execution(tmp_path: Path):
    out_html = tmp_path / "demo_report.html"
    result = runner.invoke(app, ["demo", "--out", str(out_html)])
    assert result.exit_code == 0
    assert "StreamFusion Demonstration Mode" in result.output
    assert out_html.exists()
    assert out_html.stat().st_size > 500


def test_cli_chat_nlp():
    fixture_chat = Path(__file__).parent / "fixtures" / "sample_twitch_chat.json"
    result = runner.invoke(app, ["chat-nlp", str(fixture_chat), "--min-authors", "2"])
    assert result.exit_code == 0
    assert "Chat NLP" in result.output
    assert "Community Emotional Intent Distribution" in result.output


def test_cli_sponsor():
    fixture_chat = Path(__file__).parent / "fixtures" / "sample_twitch_chat.json"
    result = runner.invoke(app, ["sponsor", "TestBrand", "--chat", str(fixture_chat), "--start", "0", "--end", "20"])
    assert result.exit_code == 0
    assert "Sponsor Impact Analysis" in result.output
    assert "Brand Attention Score" in result.output


def test_cli_audit_commands(tmp_path: Path):
    db_file = tmp_path / "test_audit.db"

    # 1. Audit help
    res_help = runner.invoke(app, ["audit", "--help"])
    assert res_help.exit_code == 0
    assert "benchmark" in res_help.output
    assert "history" in res_help.output
    assert "compare" in res_help.output

    # 2. History on empty db
    res_hist_empty = runner.invoke(app, ["audit", "history", "--db", str(db_file)])
    assert res_hist_empty.exit_code == 0
    assert "No audit" in res_hist_empty.output

    # 3. Populate store with two test runs and test history + compare
    from stream_fusion.monitoring.audit import AuditStore, AuditRunRecord, StageTelemetry
    store = AuditStore(db_path=db_file)

    st_base = {"demux": StageTelemetry(stage_name="demux", duration_sec=2.0, start_time_iso="", end_time_iso="", ram_mb_peak=100.0, vram_mb_peak=0.0)}
    rec_base = AuditRunRecord(
        run_id="run_base",
        git_commit="abc123",
        git_dirty=False,
        timestamp="2026-09-27T16:00:00Z",
        stream_id="stream_test",
        media_duration_sec=60.0,
        total_pipeline_duration_sec=10.0,
        overall_real_time_factor=0.166,
        stages=st_base,
        quality_metrics={"chat_count": 100},
        status="SUCCESS",
    )
    store.record_run(rec_base, set_as_baseline=True)

    st_cur = {"demux": StageTelemetry(stage_name="demux", duration_sec=3.5, start_time_iso="", end_time_iso="", ram_mb_peak=150.0, vram_mb_peak=0.0)}
    rec_cur = AuditRunRecord(
        run_id="run_current",
        git_commit="def456",
        git_dirty=True,
        timestamp="2026-09-27T16:10:00Z",
        stream_id="stream_test",
        media_duration_sec=60.0,
        total_pipeline_duration_sec=14.0,
        overall_real_time_factor=0.233,
        stages=st_cur,
        quality_metrics={"chat_count": 100},
        status="SUCCESS",
    )
    store.record_run(rec_cur, set_as_baseline=False)

    # Test history
    res_hist = runner.invoke(app, ["audit", "history", "--db", str(db_file)])
    assert res_hist.exit_code == 0
    assert "run_base" in res_hist.output
    assert "run_current" in res_hist.output
    assert "BASELINE" in res_hist.output

    # Test compare
    res_comp = runner.invoke(app, ["audit", "compare", "baseline", "run_current", "--db", str(db_file)])
    assert res_comp.exit_code == 0
    assert "Audit Run Comparison" in res_comp.output
    assert "Total Pipeline Duration" in res_comp.output

    # Test compare with invalid ID
    res_comp_err = runner.invoke(app, ["audit", "compare", "non_existent_run", "--db", str(db_file)])
    assert res_comp_err.exit_code == 1
    assert "not found" in res_comp_err.output


def test_cli_query_claims():
    res = runner.invoke(app, ["query-claims", "Godzilla", "--creator", "asmongold"])
    assert res.exit_code == 0
    assert "Knowledge Query Results" in res.output
    assert "Synthesized Entity Stances" in res.output


def test_cli_clip_execution(tmp_path: Path):
    fixture_video = Path(__file__).parent / "fixtures" / "sample_test_vod.mp4"
    out_short = tmp_path / "test_short.mp4"
    res = runner.invoke(app, ["clip", str(fixture_video), "--start", "0.0", "--end", "2.0", "--out", str(out_short)])
    assert res.exit_code == 0
    assert "Vertical short exported" in res.output
    assert out_short.exists()


def test_cli_audit_benchmark_missing_video():
    res = runner.invoke(app, ["audit", "benchmark", "--video", "missing_video_12345.mp4"])
    assert res.exit_code == 1
    assert "not found" in res.output


def test_config_centralization_and_homelab(tmp_path: Path, monkeypatch):
    from stream_fusion.config import StreamFusionConfig, load_config, save_config

    cfg = StreamFusionConfig()
    assert cfg.homelab.host == "cbwdellr720"
    assert cfg.homelab.port == 5432
    assert "cbwdellr720" in cfg.homelab.storage_root

    # Test saving and loading from YAML
    yaml_file = tmp_path / "streamfusion.yaml"
    cfg.homelab.host = "10.147.17.5"
    save_config(cfg, yaml_file)
    assert yaml_file.exists()

    loaded = load_config(yaml_file)
    assert loaded.homelab.host == "10.147.17.5"

    # Test environment variable override
    monkeypatch.setenv("STREAMFUSION_HOMELAB_HOST", "custom-homelab-host")
    override_cfg = load_config(yaml_file)
    assert override_cfg.homelab.host == "custom-homelab-host"




