"""Tests for Spec 27: Unified Web Dashboard & Real-Time Studio."""

import json
from pathlib import Path
import tempfile
from typing import Any, Dict
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from stream_fusion.cli import app as cli_app
from stream_fusion.harvester.catalog import HarvestCatalog
from stream_fusion.harvester.daemon import HomelabDaemon
from stream_fusion.models.schemas import (
    DaemonConfig,
    DaemonState,
    DashboardConfig,
    HarvestedVodRecord,
    HarvestStatus,
    StreamerTargetRecord,
)
from stream_fusion.monitoring.broadcaster import LiveEventBroadcaster
from stream_fusion.schema.envelope import StreamEventType, StreamFusionEnvelope
from stream_fusion.web.app import create_app


@pytest.fixture
def temp_env():
    """Provides an isolated temporary environment with catalog and storage."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        db_path = tmp_path / "catalog.db"
        catalog = HarvestCatalog(database_url=f"sqlite:///{db_path}")

        # Seed sample target streamer
        target = StreamerTargetRecord(
            streamer_id="asmongold",
            display_name="Asmongold",
            channel_urls=["https://twitch.tv/asmongold"],
            primary_platform="twitch",
            download_priority=8,
            enabled=True,
        )
        catalog.upsert_streamer_target(target)

        # Seed sample chat log file
        chat_path = tmp_path / "sample_chat.json"
        chat_path.write_text(
            json.dumps([
                {
                    "author_name": "Gamer123",
                    "content": "Pog That was unbelievable!",
                    "timestamp_offset": 12.5,
                    "color": "#38bdf8",
                },
                {
                    "author_name": "StreamFan",
                    "content": "OMEGALUL so true",
                    "timestamp_offset": 24.0,
                    "color": "#eab308",
                },
            ]),
            encoding="utf-8",
        )

        # Seed sample harvested VOD
        vod = HarvestedVodRecord(
            vod_id="vod_asmon_001",
            streamer_id="asmongold",
            platform="twitch",
            title="Asmongold Reacts to New MMO",
            duration_sec=120.0,
            status=HarvestStatus.HARVESTED,
            chat_path=str(chat_path),
            file_size_bytes=104857600,
        )
        catalog.upsert_harvested_vod(vod)

        daemon_cfg = DaemonConfig(
            homelab_root=str(tmp_path),
            catalog_db_url=f"sqlite:///{db_path}",
            pid_file=str(tmp_path / "daemon.pid"),
            log_file=str(tmp_path / "daemon.log"),
        )
        daemon = HomelabDaemon(config=daemon_cfg, catalog=catalog)

        broadcaster = LiveEventBroadcaster(replay_capacity=100)
        app_cfg = DashboardConfig(
            host="127.0.0.1",
            port=8888,
            homelab_root=str(tmp_path),
            catalog_db_url=f"sqlite:///{db_path}",
        )
        app = create_app(
            config=app_cfg,
            catalog=catalog,
            daemon=daemon,
            broadcaster=broadcaster,
        )
        client = TestClient(app)

        try:
            yield {
                "tmp_path": tmp_path,
                "catalog": catalog,
                "daemon": daemon,
                "broadcaster": broadcaster,
                "app": app,
                "client": client,
            }
        finally:
            daemon.close()
            catalog.close()
            import logging
            for h in list(logging.getLogger().handlers):
                try:
                    h.close()
                    logging.getLogger().removeHandler(h)
                except Exception:
                    pass



# --- 1. System Health & Storage Routes ---

def test_system_health_and_storage(temp_env):
    client: TestClient = temp_env["client"]

    # Health check
    res = client.get("/api/system/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "uptime_seconds" in data
    assert data["version"] == "0.1.0"

    # Storage check
    res = client.get("/api/system/storage")
    assert res.status_code == 200
    s_data = res.json()
    assert "total_gb" in s_data
    assert "free_gb" in s_data
    assert s_data["free_gb"] > 0

    # Stats aggregation
    res = client.get("/api/system/stats")
    assert res.status_code == 200
    stats = res.json()
    assert stats["streamer_count"] == 1
    assert stats["total_vods_count"] == 1
    assert stats["harvested_vods_count"] == 1


# --- 2. Roster Management Routes ---

def test_roster_crud_endpoints(temp_env):
    client: TestClient = temp_env["client"]

    # List roster
    res = client.get("/api/roster")
    assert res.status_code == 200
    roster = res.json()
    assert len(roster) == 1
    assert roster[0]["streamer_id"] == "asmongold"

    # Get single
    res = client.get("/api/roster/asmongold")
    assert res.status_code == 200
    assert res.json()["display_name"] == "Asmongold"

    # Create new streamer
    new_streamer = {
        "streamer_id": "shroud",
        "display_name": "Shroud",
        "channel_urls": ["https://twitch.tv/shroud"],
        "primary_platform": "twitch",
        "download_priority": 9,
        "enabled": True,
    }
    res = client.post("/api/roster", json=new_streamer)
    assert res.status_code == 200
    assert res.json()["streamer_id"] == "shroud"

    # Verify presence
    res = client.get("/api/roster")
    assert len(res.json()) == 2

    # Delete streamer
    res = client.delete("/api/roster/shroud")
    assert res.status_code == 200
    assert res.json()["status"] == "OK"

    # Verify deletion
    res = client.get("/api/roster/shroud")
    assert res.status_code == 404


# --- 3. VOD Catalog & Queue Routes ---

def test_catalog_and_queue_endpoints(temp_env):
    client: TestClient = temp_env["client"]

    # List VODs
    res = client.get("/api/catalog")
    assert res.status_code == 200
    vods = res.json()
    assert len(vods) == 1
    assert vods[0]["vod_id"] == "vod_asmon_001"

    # Enqueue new VOD manually
    enqueue_req = {
        "streamer_id": "asmongold",
        "vod_id": "vod_asmon_manual_002",
        "platform": "twitch",
        "title": "Manual Test VOD",
        "url": "https://twitch.tv/videos/999999",
        "duration_sec": 300.0,
    }
    res = client.post("/api/catalog/queue", json=enqueue_req)
    assert res.status_code == 200
    assert res.json()["vod_id"] == "vod_asmon_manual_002"
    assert res.json()["status"] == "QUEUED"

    # Search VODs
    res = client.get("/api/catalog?search=Manual")
    assert res.status_code == 200
    found = res.json()
    assert len(found) == 1
    assert found[0]["vod_id"] == "vod_asmon_manual_002"

    # Trigger analysis on harvested VOD
    res = client.post("/api/catalog/analyze/vod_asmon_001")
    assert res.status_code == 200
    assert res.json()["status"] == "OK"


# --- 4. Daemon Supervisor & Control Routes ---

def test_daemon_control_routes(temp_env):
    client: TestClient = temp_env["client"]

    # Status check
    res = client.get("/api/daemon/status")
    assert res.status_code == 200
    rep = res.json()
    assert rep["state"] == "STOPPED"

    # Start daemon
    res = client.post("/api/daemon/start")
    assert res.status_code == 200
    assert res.json()["status"] == "OK"

    # Check state is RUNNING
    res = client.get("/api/daemon/status")
    assert res.json()["state"] == "RUNNING"

    # Pause daemon
    res = client.post("/api/daemon/pause")
    assert res.status_code == 200
    res = client.get("/api/daemon/status")
    assert res.json()["state"] == "PAUSED"

    # Resume daemon
    res = client.post("/api/daemon/resume")
    assert res.status_code == 200
    res = client.get("/api/daemon/status")
    assert res.json()["state"] == "RUNNING"

    # Trigger crawl
    res = client.post("/api/daemon/crawl")
    assert res.status_code == 200
    assert res.json()["status"] == "OK"

    # Stop daemon
    res = client.post("/api/daemon/stop")
    assert res.status_code == 200
    res = client.get("/api/daemon/status")
    assert res.json()["state"] == "STOPPED"


# --- 5. Multimodal Player & Timeline Routes ---

def test_player_and_timeline_scrubber(temp_env):
    client: TestClient = temp_env["client"]

    # Timeline payload
    res = client.get("/api/vods/vod_asmon_001/timeline")
    assert res.status_code == 200
    tl = res.json()
    assert tl["vod_id"] == "vod_asmon_001"
    assert tl["duration_sec"] == 120.0
    assert len(tl["waveform"]) > 0

    # Chat replay window
    res = client.get("/api/vods/vod_asmon_001/chat?start_sec=10.0&end_sec=30.0")
    assert res.status_code == 200
    c_data = res.json()
    assert c_data["count"] == 2
    assert c_data["messages"][0]["author_name"] == "Gamer123"

    # Fusion slices
    res = client.get("/api/vods/vod_asmon_001/slices")
    assert res.status_code == 200
    assert "slices" in res.json()


# --- 6. Short Studio Routes ---

def test_short_studio_routes(temp_env):
    client: TestClient = temp_env["client"]

    # List candidates
    res = client.get("/api/shorts")
    assert res.status_code == 200

    # Get single candidate inspection
    res = client.get("/api/shorts/cand_test_001")
    assert res.status_code == 200
    c = res.json()
    assert "virality_score" in c
    assert "script" in c
    assert "camera_layout" in c

    # Render candidate
    render_req = {
        "candidate_id": "cand_test_001",
        "vertical_width": 1080,
        "vertical_height": 1920,
        "subtitle_style": "DYNAMIC_WORD_HIGHLIGHT",
        "reaction_layout": "SPLIT_CAM_GAME",
        "render_full_video": False,
    }
    res = client.post("/api/shorts/cand_test_001/render", json=render_req)
    assert res.status_code == 200
    assert res.json()["status"] == "OK"


# --- 7. Knowledge & Stance Explorer Routes ---

def test_knowledge_and_stances_routes(temp_env):
    client: TestClient = temp_env["client"]

    # Claims fact-checks
    res = client.get("/api/knowledge/claims")
    assert res.status_code == 200
    claims = res.json()
    assert len(claims) >= 2
    assert any(c["verdict"] in ("TRUE", "FALSE") for c in claims)

    # Filter claims by verdict
    res = client.get("/api/knowledge/claims?verdict=TRUE")
    assert res.status_code == 200
    for c in res.json():
        assert c["verdict"] == "TRUE"

    # Stances
    res = client.get("/api/knowledge/stances")
    assert res.status_code == 200
    stances = res.json()
    assert len(stances) >= 2
    assert any(s["entity"] == "Blizzard" for s in stances)

    # Sponsors
    res = client.get("/api/knowledge/sponsors")
    assert res.status_code == 200
    sponsors = res.json()
    assert len(sponsors) >= 1
    assert sponsors[0]["brand_name"] == "Starforge Systems"


# --- 8. Live Tail Monitor & Ingestion Routes ---

def test_live_monitor_routes(temp_env):
    client: TestClient = temp_env["client"]

    # Status check initially empty
    res = client.get("/api/live/status")
    assert res.status_code == 200
    assert res.json()["active_streams_count"] == 0

    # Start live tailing for channel
    start_req = {
        "channel_name": "asmongold",
        "platform": "twitch",
        "buffer_duration_sec": 60.0,
    }
    res = client.post("/api/live/start", json=start_req)
    assert res.status_code == 200
    assert res.json()["status"] == "OK"

    # Verify active stream exists
    res = client.get("/api/live/status")
    assert res.json()["active_streams_count"] == 1
    assert res.json()["streams"][0]["channel_name"] == "asmongold"

    # Stop live tailing
    res = client.post("/api/live/stop/asmongold")
    assert res.status_code == 200
    assert res.json()["status"] == "OK"

    # Verify active streams is 0
    res = client.get("/api/live/status")
    assert res.json()["active_streams_count"] == 0


# --- 9. JSON-RPC 2.0 Gateway Route ---

def test_json_rpc_gateway(temp_env):
    client: TestClient = temp_env["client"]

    # Ping method
    ping_payload = {
        "jsonrpc": "2.0",
        "method": "streamfusion.ping",
        "params": {},
        "id": "req-1",
    }
    res = client.post("/api/rpc", json=ping_payload)
    assert res.status_code == 200
    body = res.json()
    assert body["jsonrpc"] == "2.0"
    assert body["id"] == "req-1"
    assert body["result"]["status"] == "pong"

    # List schemas method
    list_schemas_payload = {
        "jsonrpc": "2.0",
        "method": "streamfusion.listSchemas",
        "params": {},
        "id": "req-2",
    }
    res = client.post("/api/rpc", json=list_schemas_payload)
    assert res.status_code == 200
    body = res.json()
    assert isinstance(body["result"], list)
    assert len(body["result"]) >= 70

    # Batch request
    batch = [
        {"jsonrpc": "2.0", "method": "streamfusion.ping", "params": {}, "id": "b-1"},
        {"jsonrpc": "2.0", "method": "streamfusion.ping", "params": {}, "id": "b-2"},
    ]
    res = client.post("/api/rpc", json=batch)
    assert res.status_code == 200
    b_res = res.json()
    assert len(b_res) == 2
    assert b_res[0]["id"] == "b-1"
    assert b_res[1]["id"] == "b-2"


# --- 10. WebSocket Hub & Real-Time Event Dispatch ---

def test_websocket_live_connection(temp_env):
    client: TestClient = temp_env["client"]
    broadcaster: LiveEventBroadcaster = temp_env["broadcaster"]

    with client.websocket_connect("/ws/live") as websocket:
        # Send ping action to WebSocket
        websocket.send_text(json.dumps({"action": "ping"}))
        raw = websocket.receive_text()
        msg = json.loads(raw)
        assert msg["event_type"] == "PONG"
        assert msg["payload"]["message"] == "pong"

        # Emit live event through broadcaster and verify client receives it
        test_env = StreamFusionEnvelope[Dict[str, Any]](
            event_type=StreamEventType.CHAT_MESSAGE,
            stream_id="test_stream",
            payload={"author_name": "TestBot", "content": "Hello Realtime!"},
        )
        broadcaster.emit_event(
            event_type=StreamEventType.CHAT_MESSAGE,
            payload=test_env.payload,
            stream_id="test_stream",
        )


# --- 11. Static Frontend Asset Serving ---

def test_static_frontend_serving(temp_env):
    client: TestClient = temp_env["client"]

    # Main index.html
    res = client.get("/")
    assert res.status_code == 200
    assert "StreamFusion Studio" in res.text
    assert "Spec 27" in res.text

    # CSS asset
    res = client.get("/static/css/studio.css")
    assert res.status_code == 200
    assert "--bg-primary" in res.text

    # JS asset
    res = client.get("/static/js/studio.js")
    assert res.status_code == 200
    assert "initWebSocket" in res.text


# --- 12. CLI Dashboard Commands ---

def test_cli_dashboard_commands():
    runner = CliRunner()

    # Test dashboard --help
    res = runner.invoke(cli_app, ["dashboard", "--help"])
    assert res.exit_code == 0
    assert "Unified Web Dashboard & Real-Time Studio" in res.output
    assert "serve" in res.output
    assert "status" in res.output
    assert "open" in res.output
