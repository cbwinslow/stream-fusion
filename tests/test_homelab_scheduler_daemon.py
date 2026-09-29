"""Unit and integration tests for Spec 26: Homelab 24/7 Scheduler Daemon & Service Orchestration."""

from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Dict

import pytest
from typer.testing import CliRunner

from stream_fusion.cli import app
from stream_fusion.harvester.catalog import HarvestCatalog
from stream_fusion.harvester.crawler import ChannelVodCrawler
from stream_fusion.harvester.daemon import HomelabDaemon, is_process_running
from stream_fusion.harvester.engine import HomelabHarvester
from stream_fusion.harvester.scheduler import HomelabScheduler, run_storage_retention_cleanup
from stream_fusion.models.schemas import (
    DaemonConfig,
    DaemonJobRecord,
    DaemonState,
    DaemonStatusReport,
    DaemonTaskType,
    FullSpectrumManifest,
    HarvestedVodRecord,
    HarvestStatus,
    StreamAnalysisResult,
    StreamerTargetRecord,
)
from stream_fusion.schema.agent_rpc import AgentRpcDispatcher

runner = CliRunner()


@pytest.fixture
def temp_daemon_env():
    """Sets up an isolated filesystem and database environment for daemon testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        homelab_root = root / "homelab_storage"
        homelab_root.mkdir(parents=True, exist_ok=True)
        db_path = homelab_root / "catalog.db"
        catalog = HarvestCatalog(database_url=db_path)
        pid_file = root / "daemon.pid"
        log_file = root / "logs" / "daemon.log"

        config = DaemonConfig(
            crawl_interval_minutes=1,
            download_poll_interval_seconds=1,
            auto_analyze=True,
            max_concurrent_downloads=2,
            max_concurrent_pipelines=1,
            min_free_disk_gb=1.0,
            pid_file=str(pid_file),
            log_file=str(log_file),
            homelab_root=str(homelab_root),
            catalog_db_url=f"sqlite:///{db_path}",
        )

        yield {
            "root": root,
            "homelab_root": homelab_root,
            "catalog": catalog,
            "config": config,
            "pid_file": pid_file,
            "log_file": log_file,
        }
        catalog.close()


def test_daemon_config_serialization():
    """Validates DaemonConfig defaults, serialization and deserialization."""
    cfg = DaemonConfig(
        crawl_interval_minutes=45,
        download_poll_interval_seconds=20,
        max_concurrent_downloads=4,
        retention_days=14,
    )
    data = cfg.model_dump(mode="json")
    assert data["crawl_interval_minutes"] == 45
    assert data["download_poll_interval_seconds"] == 20
    assert data["max_concurrent_downloads"] == 4
    assert data["retention_days"] == 14
    assert data["auto_analyze"] is True

    # Reconstruct
    cfg2 = DaemonConfig.model_validate(data)
    assert cfg2.crawl_interval_minutes == 45
    assert cfg2.retention_days == 14


def test_scheduler_task_registration_and_tick():
    """Tests HomelabScheduler registering tasks, tick execution, and manual trigger."""
    scheduler = HomelabScheduler()
    counter = {"ticks": 0, "triggers": 0}

    def _tick_action():
        counter["ticks"] += 1
        return "ticked"

    def _trigger_action():
        counter["triggers"] += 1
        return "triggered"

    scheduler.add_task(
        name="test_periodic",
        task_type=DaemonTaskType.DOWNLOAD,
        interval_seconds=0.1,
        action=_tick_action,
        initial_delay_seconds=0.0,
    )
    scheduler.add_task(
        name="test_manual",
        task_type=DaemonTaskType.CRAWL,
        interval_seconds=100.0,
        action=_trigger_action,
        initial_delay_seconds=100.0,
    )

    # Initial tick executes test_periodic because initial delay was 0
    executed = scheduler.tick()
    assert "test_periodic" in executed
    assert counter["ticks"] == 1
    assert "test_manual" not in executed

    # Trigger test_manual explicitly
    res = scheduler.trigger_task("test_manual")
    assert res == "triggered"
    assert counter["triggers"] == 1


def test_retention_housekeeping(temp_daemon_env):
    """Tests run_storage_retention_cleanup pruning old analyzed raw video files."""
    catalog = temp_daemon_env["catalog"]
    homelab_root = temp_daemon_env["homelab_root"]

    # Target record required for foreign key constraint
    target = StreamerTargetRecord(
        streamer_id="streamerA",
        display_name="Streamer A",
        channel_urls=["https://twitch.tv/streamerA"],
    )
    catalog.add_target(target)

    # Create dummy VOD directory and files
    vod_dir = homelab_root / "vods" / "streamerA" / "2026-09-01_vod_old"
    vod_dir.mkdir(parents=True, exist_ok=True)
    media_file = vod_dir / "media.mp4"
    chat_file = vod_dir / "chat.json"
    manifest_file = vod_dir / "ingest_manifest.json"

    media_file.write_text("dummy raw media content")
    chat_file.write_text("{}")
    manifest_file.write_text("{}")

    # Register in catalog with analyzed_at 20 days ago
    old_time = datetime.now(timezone.utc) - timedelta(days=20)
    vod = HarvestedVodRecord(
        vod_id="vod_old",
        streamer_id="streamerA",
        platform="TWITCH",
        status=HarvestStatus.ANALYZED,
        video_path=str(media_file),
        chat_path=str(chat_file),
        analyzed_at=old_time,
    )
    catalog.add_vod(vod)

    # Run cleanup with 14-day retention
    pruned = run_storage_retention_cleanup(catalog=catalog, homelab_root=homelab_root, retention_days=14)
    assert pruned == 1
    assert not media_file.exists()  # Raw video was pruned
    assert chat_file.exists()  # Chat replay preserved
    assert manifest_file.exists()  # Manifest preserved


def test_pid_lockfile_lifecycle(temp_daemon_env):
    """Tests PID lock acquisition, duplicate guard, and clean release."""
    config = temp_daemon_env["config"]
    pid_file = temp_daemon_env["pid_file"]

    daemon = HomelabDaemon(config=config, catalog=temp_daemon_env["catalog"])
    try:
        assert not pid_file.exists()

        daemon.acquire_pid_lock()
        assert pid_file.exists()
        with open(pid_file, "r") as f:
            assert int(f.read().strip()) == os.getpid()

        # Attempting to acquire again with same running process ID should succeed or not crash
        daemon.acquire_pid_lock()

        # Release
        daemon.release_pid_lock()
        assert not pid_file.exists()
    finally:
        daemon.close()


def test_daemon_lifecycle_pause_resume_stop(temp_daemon_env):
    """Tests starting daemon in background, pausing, resuming, and graceful stop."""
    config = temp_daemon_env["config"]
    catalog = temp_daemon_env["catalog"]

    daemon = HomelabDaemon(config=config, catalog=catalog)
    try:
        daemon.start(foreground=False)

        time.sleep(0.2)
        assert daemon.state == DaemonState.RUNNING

        status = daemon.get_status()
        assert status.state == DaemonState.RUNNING
        assert status.pid == os.getpid()
        assert status.uptime_seconds >= 0.0

        # Pause
        daemon.pause()
        assert daemon.state == DaemonState.PAUSED

        # Resume
        daemon.resume()
        assert daemon.state == DaemonState.RUNNING

        # Stop
        daemon.stop()
        time.sleep(0.3)
        assert daemon.state == DaemonState.STOPPED
    finally:
        daemon.close()


def test_daemon_auto_pipeline_handoff(temp_daemon_env):
    """Tests that harvested VODs are automatically picked up by pipeline executor."""
    config = temp_daemon_env["config"]
    catalog = temp_daemon_env["catalog"]
    homelab_root = temp_daemon_env["homelab_root"]

    # Target streamer
    target = StreamerTargetRecord(
        streamer_id="streamerX",
        display_name="Streamer X",
        channel_urls=["https://twitch.tv/streamerX"],
    )
    catalog.add_target(target)

    # Create dummy VOD directory and files
    vod_dir = homelab_root / "vods" / "streamerX" / "2026-09-29_vod_auto"
    vod_dir.mkdir(parents=True, exist_ok=True)
    media_file = vod_dir / "media.mp4"
    media_file.write_text("video dummy")

    # Add VOD in HARVESTED status
    vod = HarvestedVodRecord(
        vod_id="vod_auto",
        streamer_id="streamerX",
        platform="TWITCH",
        status=HarvestStatus.HARVESTED,
        video_path=str(media_file),
    )
    catalog.add_vod(vod)

    executed_vods = []

    def mock_pipeline_executor(vod_id, catalog, config=None, **kwargs):
        executed_vods.append(vod_id)
        catalog.update_vod_status(vod_id, HarvestStatus.ANALYZED)
        manifest = FullSpectrumManifest(
            stream_id=vod_id,
            duration_sec=60.0,
            stages_executed=["phase_1_demuxer", "phase_9_packaging"],
        )
        analysis = StreamAnalysisResult(stream_id=vod_id, duration_sec=60.0)
        return analysis, manifest

    daemon = HomelabDaemon(
        config=config,
        catalog=catalog,
        pipeline_executor=mock_pipeline_executor,
    )
    try:
        # Manually execute scheduled pipeline tick
        res = daemon._scheduled_pipeline()
        assert res.get("launched") == "vod_auto"

        # Wait briefly for thread to complete
        time.sleep(0.3)
        assert "vod_auto" in executed_vods
        updated = catalog.get_vod("vod_auto")
        assert updated.status == HarvestStatus.ANALYZED
    finally:
        daemon.close()


def test_daemon_json_rpc_handlers(temp_daemon_env):
    """Tests JSON-RPC 2.0 endpoints for daemon control."""
    config = temp_daemon_env["config"]
    catalog = temp_daemon_env["catalog"]

    daemon = HomelabDaemon(config=config, catalog=catalog)
    daemon.start(foreground=False)

    dispatcher = AgentRpcDispatcher()
    dispatcher.set_daemon(daemon)

    try:
        # 1. getStatus
        req = {"jsonrpc": "2.0", "id": 1, "method": "streamfusion.daemon.getStatus", "params": {}}
        resp = dispatcher.handle_request(req)
        assert resp["result"]["state"] == "RUNNING"
        assert resp["result"]["pid"] == os.getpid()

        # 2. pause
        req = {"jsonrpc": "2.0", "id": 2, "method": "streamfusion.daemon.pause", "params": {}}
        resp = dispatcher.handle_request(req)
        assert resp["result"]["status"] == "PAUSED"
        assert daemon.state == DaemonState.PAUSED

        # 3. resume
        req = {"jsonrpc": "2.0", "id": 3, "method": "streamfusion.daemon.resume", "params": {}}
        resp = dispatcher.handle_request(req)
        assert resp["result"]["status"] == "RUNNING"
        assert daemon.state == DaemonState.RUNNING

        # 4. triggerCrawl
        req = {"jsonrpc": "2.0", "id": 4, "method": "streamfusion.daemon.triggerCrawl", "params": {}}
        resp = dispatcher.handle_request(req)
        assert resp["result"]["status"] == "TRIGGERED"

        # 5. stop
        req = {"jsonrpc": "2.0", "id": 5, "method": "streamfusion.daemon.stop", "params": {}}
        resp = dispatcher.handle_request(req)
        assert resp["result"]["status"] == "STOPPING"
    finally:
        daemon.stop()
        daemon.close()
        time.sleep(0.2)


def test_daemon_cli_commands(temp_daemon_env):
    """Tests CLI commands: init-config and status."""
    root = temp_daemon_env["root"]
    cfg_out = root / "test_config.yaml"

    # Test init-config
    res = runner.invoke(app, ["daemon", "init-config", "--output", str(cfg_out)])
    assert res.exit_code == 0
    assert cfg_out.exists()
    assert "crawl_interval_minutes" in cfg_out.read_text()

    # Test status
    res = runner.invoke(app, ["daemon", "status", "--pid-file", str(root / "nonexistent.pid")])
    assert res.exit_code == 0
    assert "Daemon State" in res.stdout or "STOPPED" in res.stdout
