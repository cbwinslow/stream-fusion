"""Unit and Integration Tests for Spec 16: Unified JSON Schema & Agent Communication."""

import json
from pathlib import Path
import pytest
from typer.testing import CliRunner

from stream_fusion.models.schemas import (
    FusionSlice,
    StreamerClaim,
    ChatMessage,
)
from stream_fusion.schema.envelope import (
    StreamEventType,
    EnvelopeTelemetry,
    StreamFusionEnvelope,
)
from stream_fusion.schema.registry import SchemaRegistry, default_schema_registry
from stream_fusion.schema.adapters import (
    JsonlStreamAdapter,
    SqliteJsonStore,
    ParquetJsonBridge,
)
from stream_fusion.schema.agent_rpc import AgentRpcDispatcher
from stream_fusion.cli import app


@pytest.fixture
def runner():
    return CliRunner()


def test_stream_fusion_envelope_serialization():
    """Verify envelope serialization and deserialization."""
    telemetry = EnvelopeTelemetry(duration_ms=45.2, stage="diarization", ram_mb=120.0)
    claim = StreamerClaim(
        claim_id="c_001",
        creator_id="theburntpeanut",
        vod_id="vod_101",
        timestamp_sec=10.0,
        end_sec=15.0,
        subject_entity="Palworld",
        predicate="is",
        object_value="addictive",
        stance="APPROVAL",
        polarity=0.85,
        raw_quote="Palworld is honestly so addictive man",
    )

    envelope = StreamFusionEnvelope.create(
        stream_id="vod_101",
        event_type=StreamEventType.CLAIM_EXTRACTED,
        payload=claim,
        producer="stream_fusion.knowledge.claims",
        trace_id="trace-test-123",
        telemetry=telemetry,
        metadata={"priority": "high"},
    )

    assert envelope.version == "1.0"
    assert envelope.stream_id == "vod_101"
    assert envelope.event_type == StreamEventType.CLAIM_EXTRACTED
    assert envelope.payload["subject_entity"] == "Palworld"
    assert envelope.telemetry.duration_ms == 45.2

    # JSON roundtrip
    json_str = envelope.to_json()
    assert "CLAIM_EXTRACTED" in json_str

    deserialized = StreamFusionEnvelope.from_json(json_str)
    assert deserialized.message_id == envelope.message_id
    assert deserialized.event_type == StreamEventType.CLAIM_EXTRACTED
    assert deserialized.payload["polarity"] == 0.85
    assert deserialized.telemetry.stage == "diarization"


def test_schema_registry(tmp_path: Path):
    """Test schema registry listing, JSON-schema export, and validation."""
    registry = SchemaRegistry()
    schemas = registry.list_schemas()
    assert len(schemas) >= 30

    names = [s["schema_name"] for s in schemas]
    assert "FusionSlice" in names
    assert "StreamerClaim" in names
    assert "StreamFusionEnvelope" in names

    # Single model export
    schema_dict = registry.export_json_schema("FusionSlice")
    assert "properties" in schema_dict
    assert "bucket_index" in schema_dict["properties"]

    with pytest.raises(KeyError):
        registry.export_json_schema("NonExistentModel")

    # Export all schemas to disk
    out_dir = tmp_path / "schemas_export"
    exported_json = registry.export_all_schemas(output_dir=out_dir, fmt="json")
    assert len(exported_json) >= 30
    assert (out_dir / "FusionSlice.schema.json").exists()

    exported_yaml = registry.export_all_schemas(output_dir=out_dir, fmt="yaml")
    assert (out_dir / "FusionSlice.schema.yaml").exists()

    # Validation
    valid_data = {
        "message_id": "m1",
        "timestamp_offset": 5.2,
        "user_id": "u1",
        "author_name": "gamer1",
        "content": "KEKW",
    }
    is_valid, err, inst = registry.validate_data("ChatMessage", valid_data)
    assert is_valid is True
    assert err is None
    assert isinstance(inst, ChatMessage)

    # Invalid data (missing required field)
    invalid_data = {"message_id": "m2"}
    is_valid, err, inst = registry.validate_data("ChatMessage", invalid_data)
    assert is_valid is False
    assert err is not None
    assert inst is None


def test_jsonl_stream_adapter(tmp_path: Path):
    """Test JSONL streaming reader and writer with filtering."""
    file_path = tmp_path / "stream_events.jsonl"
    adapter = JsonlStreamAdapter()

    env1 = StreamFusionEnvelope.create(
        stream_id="stream_a",
        event_type=StreamEventType.STAGE_START,
        payload={"stage": "demux"},
    )
    env2 = StreamFusionEnvelope.create(
        stream_id="stream_a",
        event_type=StreamEventType.HIGHLIGHT_MOMENT,
        payload={"score": 0.95},
    )
    env3 = StreamFusionEnvelope.create(
        stream_id="stream_b",
        event_type=StreamEventType.HIGHLIGHT_MOMENT,
        payload={"score": 0.72},
    )

    adapter.write_envelope(file_path, env1)
    written = adapter.write_envelopes(file_path, [env2, env3])
    assert written == 2

    # Read all
    all_read = list(adapter.read_envelopes(file_path))
    assert len(all_read) == 3

    # Filter by stream_id
    stream_a_only = list(adapter.read_envelopes(file_path, stream_id="stream_a"))
    assert len(stream_a_only) == 2

    # Filter by event_type
    highlights_only = list(
        adapter.read_envelopes(file_path, event_type=StreamEventType.HIGHLIGHT_MOMENT)
    )
    assert len(highlights_only) == 2

    # Non-existent file yields nothing
    empty_read = list(adapter.read_envelopes(tmp_path / "non_existent.jsonl"))
    assert len(empty_read) == 0


def test_sqlite_json_store(tmp_path: Path):
    """Test SQLite JSON1 store insert, batch insert, and JSON attribute queries."""
    db_path = tmp_path / "events.db"
    store = SqliteJsonStore(db_path=db_path)

    env1 = StreamFusionEnvelope.create(
        stream_id="vod_1",
        event_type=StreamEventType.CLAIM_EXTRACTED,
        payload={
            "claim_id": "c1",
            "subject_entity": "Elden Ring",
            "stance": "APPROVAL",
            "polarity": 0.9,
        },
        producer="claim_extractor",
    )
    env2 = StreamFusionEnvelope.create(
        stream_id="vod_1",
        event_type=StreamEventType.CLAIM_EXTRACTED,
        payload={
            "claim_id": "c2",
            "subject_entity": "Overwatch 2",
            "stance": "DISAPPROVAL",
            "polarity": -0.8,
        },
        producer="claim_extractor",
    )
    env3 = StreamFusionEnvelope.create(
        stream_id="vod_2",
        event_type=StreamEventType.HIGHLIGHT_MOMENT,
        payload={"score": 0.88, "reason": "Chat explosion"},
    )

    store.insert_envelope(env1)
    batch_count = store.insert_envelopes([env2, env3])
    assert batch_count == 2

    # Total counts
    assert store.count_events() == 3
    assert store.count_events(stream_id="vod_1") == 2
    assert store.count_events(event_type=StreamEventType.HIGHLIGHT_MOMENT) == 1

    # Query with JSON1 attribute filtering
    approval_claims = store.query_events(
        stream_id="vod_1",
        event_type=StreamEventType.CLAIM_EXTRACTED,
        json_filters={"stance": "APPROVAL"},
    )
    assert len(approval_claims) == 1
    assert approval_claims[0].payload["subject_entity"] == "Elden Ring"

    disapproval_claims = store.query_events(
        stream_id="vod_1",
        json_filters={"stance": "DISAPPROVAL"},
    )
    assert len(disapproval_claims) == 1
    assert disapproval_claims[0].payload["subject_entity"] == "Overwatch 2"


def test_parquet_json_bridge(tmp_path: Path):
    """Test bidirectional Parquet-to-JSON bridge."""
    bridge = ParquetJsonBridge()
    parquet_file = tmp_path / "test_matrix.parquet"

    slices = [
        FusionSlice(
            bucket_index=0,
            start_sec=0.0,
            end_sec=5.0,
            active_speakers=["STREAMER"],
            streamer_transcript="Hello chat!",
            chat_message_count=15,
            chat_velocity_per_sec=3.0,
        ),
        FusionSlice(
            bucket_index=1,
            start_sec=5.0,
            end_sec=10.0,
            active_speakers=["STREAMER", "CO_STREAMER"],
            streamer_transcript="What a play!",
            chat_message_count=45,
            chat_velocity_per_sec=9.0,
            is_spike_moment=True,
        ),
    ]

    envelopes = bridge.slices_to_envelopes(slices, stream_id="test_stream")
    assert len(envelopes) == 2
    assert envelopes[0].event_type == StreamEventType.FUSION_SLICE

    # Convert back to slices
    recovered_slices = bridge.envelopes_to_slices(envelopes)
    assert len(recovered_slices) == 2
    assert recovered_slices[0].streamer_transcript == "Hello chat!"
    assert recovered_slices[1].is_spike_moment is True

    # Parquet export and import
    bridge.export_envelopes_to_parquet(envelopes, parquet_file)
    assert parquet_file.exists()

    loaded_envelopes = bridge.load_envelopes_from_parquet(parquet_file, stream_id="test_stream")
    assert len(loaded_envelopes) == 2
    assert loaded_envelopes[1].payload["chat_message_count"] == 45


def test_agent_rpc_dispatcher(tmp_path: Path):
    """Test JSON-RPC 2.0 requests, summary, queries, and error handling."""
    db_path = tmp_path / "agent_rpc.db"
    store = SqliteJsonStore(db_path=db_path)

    # Seed data
    store.insert_envelopes([
        StreamFusionEnvelope.create(
            stream_id="stream_rpc",
            event_type=StreamEventType.FUSION_SLICE,
            payload={"bucket_index": 0},
        ),
        StreamFusionEnvelope.create(
            stream_id="stream_rpc",
            event_type=StreamEventType.CLAIM_EXTRACTED,
            payload={
                "claim_id": "c1",
                "subject_entity": "Starfield",
                "stance": "DISAPPROVAL",
                "polarity": -0.6,
            },
        ),
        StreamFusionEnvelope.create(
            stream_id="stream_rpc",
            event_type=StreamEventType.HIGHLIGHT_MOMENT,
            payload={"highlight_score": 0.92, "reason": "Insane Boss Fight"},
        ),
        StreamFusionEnvelope.create(
            stream_id="stream_rpc",
            event_type=StreamEventType.HIGHLIGHT_MOMENT,
            payload={"highlight_score": 0.45, "reason": "Minor chuckle"},
        ),
        StreamFusionEnvelope.create(
            stream_id="stream_rpc",
            event_type=StreamEventType.SAFETY_ALERT,
            payload={"user_id": "troll_99", "verdict": "BAD_FAITH_GRIEFING"},
        ),
    ])

    dispatcher = AgentRpcDispatcher(store=store)

    # 1. Ping
    ping_resp = dispatcher.handle_request({
        "jsonrpc": "2.0",
        "method": "streamfusion.ping",
        "id": 1,
    })
    assert ping_resp["result"]["status"] == "pong"

    # 2. List schemas
    list_resp = dispatcher.handle_request({
        "jsonrpc": "2.0",
        "method": "streamfusion.listSchemas",
        "id": 2,
    })
    assert len(list_resp["result"]) >= 30

    # 3. Stream Summary
    summary_resp = dispatcher.handle_request({
        "jsonrpc": "2.0",
        "method": "streamfusion.getStreamSummary",
        "params": {"stream_id": "stream_rpc"},
        "id": 3,
    })
    res = summary_resp["result"]
    assert res["total_events"] == 5
    assert res["counts"]["claims"] == 1
    assert res["counts"]["highlights"] == 2
    assert res["counts"]["safety_alerts"] == 1

    # 4. Query Claims
    claims_resp = dispatcher.handle_request({
        "jsonrpc": "2.0",
        "method": "streamfusion.queryClaims",
        "params": {"stream_id": "stream_rpc", "stance": "DISAPPROVAL"},
        "id": 4,
    })
    assert len(claims_resp["result"]) == 1
    assert claims_resp["result"][0]["subject_entity"] == "Starfield"

    # 5. Query Highlights with score threshold
    high_resp = dispatcher.handle_request({
        "jsonrpc": "2.0",
        "method": "streamfusion.queryHighlights",
        "params": {"stream_id": "stream_rpc", "min_score": 0.8},
        "id": 5,
    })
    assert len(high_resp["result"]) == 1
    assert high_resp["result"][0]["highlight_score"] == 0.92

    # 6. Query Chatters / Safety
    safety_resp = dispatcher.handle_request({
        "jsonrpc": "2.0",
        "method": "streamfusion.queryChatters",
        "params": {"stream_id": "stream_rpc"},
        "id": 6,
    })
    assert len(safety_resp["result"]) == 1
    assert safety_resp["result"][0]["user_id"] == "troll_99"

    # 7. Error handling
    err_method = dispatcher.handle_request({
        "jsonrpc": "2.0",
        "method": "streamfusion.unknownMethod",
        "id": 7,
    })
    assert "error" in err_method
    assert err_method["error"]["code"] == -32601

    err_invalid = dispatcher.handle_request({"not": "valid"})
    assert err_invalid["error"]["code"] == -32600

    # Line handling
    line_out = dispatcher.handle_line('{"jsonrpc": "2.0", "method": "streamfusion.ping", "id": 10}')
    assert '"status": "pong"' in line_out

    bad_line_out = dispatcher.handle_line('not valid json {')
    assert '"code": -32700' in bad_line_out


def test_schema_and_agent_cli(runner, tmp_path: Path):
    """Test CLI commands for schema listing, export, validate, and agent query."""
    # 1. schema list
    res = runner.invoke(app, ["schema", "list"])
    assert res.exit_code == 0
    assert "FusionSlice" in res.stdout

    # 2. schema export
    exp_dir = tmp_path / "cli_schemas"
    res = runner.invoke(app, ["schema", "export", "--out-dir", str(exp_dir), "--model", "FusionSlice"])
    assert res.exit_code == 0
    assert (exp_dir / "FusionSlice.schema.json").exists()

    # 3. schema validate
    valid_file = tmp_path / "valid_msg.json"
    valid_file.write_text(
        json.dumps({
            "message_id": "m_test",
            "timestamp_offset": 2.5,
            "user_id": "user_a",
            "author_name": "chatter_1",
            "content": "POGGERS",
        }),
        encoding="utf-8",
    )
    res = runner.invoke(app, ["schema", "validate", "ChatMessage", str(valid_file)])
    assert res.exit_code == 0
    assert "conforms strictly" in res.stdout

    # 4. agent query & summary
    db_file = tmp_path / "cli_events.db"
    store = SqliteJsonStore(db_path=db_file)
    store.insert_envelope(
        StreamFusionEnvelope.create(
            stream_id="test_stream_cli",
            event_type=StreamEventType.HIGHLIGHT_MOMENT,
            payload={"score": 0.99, "reason": "World First"},
        )
    )

    res = runner.invoke(app, ["agent", "query", "--db", str(db_file), "--stream-id", "test_stream_cli"])
    assert res.exit_code == 0
    assert "HIGHLIGHT_MOMENT" in res.stdout

    res_json = runner.invoke(app, ["agent", "query", "--db", str(db_file), "--json"])
    assert res_json.exit_code == 0
    assert '"event_type": "HIGHLIGHT_MOMENT"' in res_json.stdout

    res_sum = runner.invoke(app, ["agent", "summary", "test_stream_cli", "--db", str(db_file)])
    assert res_sum.exit_code == 0
    assert "Total Envelopes" in res_sum.stdout
