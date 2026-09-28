"""Tests for Real-Time Live Ingestion & WebSocket Stream Tailing (Spec 19)."""

import asyncio
import json
import os
from pathlib import Path
import tempfile
import time
import pytest
from typer.testing import CliRunner

from stream_fusion.chat.live_irc import TwitchIrcMessage, TwitchIrcParser, LiveChatTailer
from stream_fusion.cli import app
from stream_fusion.ingest.live_buffer import CircularSegmentBuffer, LiveMediaSegment
from stream_fusion.ingest.live_coordinator import LiveStreamCoordinator
from stream_fusion.ingest.live_tailer import LiveStreamIngestor
from stream_fusion.models.schemas import (
    LivePlatform,
    LiveState,
    LiveStreamConfig,
    LiveStreamStatus,
    LiveTailHealthMetrics,
)
from stream_fusion.monitoring.broadcaster import LiveEventBroadcaster
from stream_fusion.nlp.adaptive_slang import RollingBurstDetector
from stream_fusion.schema.agent_rpc import AgentRpcDispatcher
from stream_fusion.schema.envelope import StreamEventType, StreamFusionEnvelope


@pytest.fixture
def cli_runner():
    return CliRunner()


# --- 1. Twitch IRC Parser Tests ---

def test_twitch_irc_parser_privmsg():
    line = (
        "@badge-info=subscriber/6;badges=subscriber/6,premium/1;color=#1E90FF;"
        "display-name=PogGamer;emotes=25:0-4;id=123-abc;mod=0;room-id=12345;"
        "subscriber=1;tmi-sent-ts=1727500000000;user-id=999 :poggamer!poggamer@poggamer.tmi.twitch.tv PRIVMSG #asmongold :Kappa that was crazy!"
    )
    msg = TwitchIrcParser.parse_line(line)
    assert msg is not None
    assert msg.command == "PRIVMSG"
    assert msg.params == ["#asmongold"]
    assert msg.trailing == "Kappa that was crazy!"
    assert msg.tags["display-name"] == "PogGamer"
    assert msg.tags["user-id"] == "999"

    # Convert to ChatMessage
    chat_msg = TwitchIrcParser.to_chat_message(msg, stream_start_ms=1727500000000)
    assert chat_msg is not None
    assert chat_msg.author_name == "PogGamer"
    assert chat_msg.user_id == "999"
    assert chat_msg.content == "Kappa that was crazy!"
    assert chat_msg.timestamp_offset == 0.0
    assert len(chat_msg.badges) == 2
    assert "subscriber/6" in chat_msg.badges
    assert len(chat_msg.emotes) == 1
    assert chat_msg.emotes[0].id == "25"
    assert chat_msg.emotes[0].name == "Kappa"


def test_twitch_irc_parser_escaped_characters():
    line = r"@badges=;display-name=User\sName;custom-tag=hello\:world\\test :user!user@user.tmi.twitch.tv PRIVMSG #channel :hi"
    msg = TwitchIrcParser.parse_line(line)
    assert msg is not None
    assert msg.tags["display-name"] == "User Name"
    assert msg.tags["custom-tag"] == r"hello;world\test"


def test_twitch_irc_parser_ping():
    line = "PING :tmi.twitch.tv"
    msg = TwitchIrcParser.parse_line(line)
    assert msg is not None
    assert msg.command == "PING"
    assert msg.trailing == "tmi.twitch.tv"
    assert TwitchIrcParser.to_chat_message(msg) is None


def test_twitch_irc_parser_malformed():
    assert TwitchIrcParser.parse_line("") is None
    assert TwitchIrcParser.parse_line("   \r\n") is None


# --- 2. Circular Segment Buffer Tests ---

def test_circular_segment_buffer_eviction():
    with tempfile.TemporaryDirectory() as tmp_dir:
        buf = CircularSegmentBuffer(buffer_duration_sec=30.0, auto_unlink=True)

        # Create dummy segment files
        f1 = Path(tmp_dir) / "seg1.mp4"
        f1.write_bytes(b"A" * 100)
        f2 = Path(tmp_dir) / "seg2.mp4"
        f2.write_bytes(b"B" * 100)
        f3 = Path(tmp_dir) / "seg3.mp4"
        f3.write_bytes(b"C" * 100)

        # Add seg 1: 0 - 15s
        buf.add_segment(1, 0.0, 15.0, video_path=str(f1))
        assert buf.segment_count == 1
        assert buf.get_buffered_duration() == 15.0

        # Add seg 2: 15 - 30s
        buf.add_segment(2, 15.0, 30.0, video_path=str(f2))
        assert buf.segment_count == 2
        assert buf.get_buffered_duration() == 30.0
        assert f1.exists()

        # Add seg 3: 30 - 45s (span = 45s > 30s) -> seg 1 must be evicted and unlinked
        buf.add_segment(3, 30.0, 45.0, video_path=str(f3))
        assert buf.segment_count == 2
        assert buf.get_buffered_duration() == 30.0
        assert not f1.exists(), "Evicted segment file should be automatically unlinked"
        assert f2.exists()
        assert f3.exists()

        # Range query
        segs = buf.get_segments_in_range(20.0, 35.0)
        assert len(segs) == 2
        assert segs[0].segment_id == 2
        assert segs[1].segment_id == 3

        buf.clear()
        assert buf.segment_count == 0
        assert not f2.exists()
        assert not f3.exists()


# --- 3. Live Event Broadcaster & Dual Transport Tests ---

def test_live_broadcaster_event_filtering_and_replay():
    async def _run():
        broadcaster = LiveEventBroadcaster(replay_capacity=10)

        # Register client 1 for ALL events
        sub1, q1 = await broadcaster.register_client(transport="WEBSOCKET", event_types=["*"])

        # Register client 2 for CHAT_BURST only
        sub2, q2 = await broadcaster.register_client(transport="SSE", event_types=["CHAT_BURST"])

        # Broadcast event 1: HIGHLIGHT_MOMENT
        env1 = StreamFusionEnvelope[dict](
            stream_id="s1",
            event_type=StreamEventType.HIGHLIGHT_MOMENT,
            payload={"score": 0.95},
        )
        delivered = await broadcaster.broadcast(env1)
        assert delivered == 1  # Only sub1 received it

        # Broadcast event 2: CHAT_BURST
        env2 = StreamFusionEnvelope[dict](
            stream_id="s1",
            event_type=StreamEventType.CHAT_BURST,
            payload={"token": "KEKW", "z_score": 4.5},
        )
        delivered2 = await broadcaster.broadcast(env2)
        assert delivered2 == 2  # Both received it

        assert broadcaster.replay_buffer_size == 2

        # Register client 3 with replay
        sub3, q3 = await broadcaster.register_client(
            transport="WEBSOCKET",
            event_types=["*"],
            replay_count=5,
        )
        assert q3.qsize() == 2, "Replay buffer should have queued 2 events for client 3"

        await broadcaster.unregister_client(sub1.client_id)
        await broadcaster.unregister_client(sub2.client_id)
        await broadcaster.unregister_client(sub3.client_id)
        assert broadcaster.active_subscribers_count == 0

    asyncio.run(_run())


def test_ws_frame_encoding_and_decoding():
    message = "Hello StreamFusion WebSocket!"
    encoded = LiveEventBroadcaster.encode_ws_frame(message)
    assert len(encoded) > len(message)
    assert encoded[0] == 0x81  # FIN + Text

    # Decode simulated masked client frame
    mask = b"\x12\x34\x56\x78"
    msg_bytes = message.encode("utf-8")
    masked_payload = bytearray(len(msg_bytes))
    for i in range(len(msg_bytes)):
        masked_payload[i] = msg_bytes[i] ^ mask[i % 4]

    header = bytearray([0x81, 0x80 | len(msg_bytes)]) + mask
    full_client_frame = bytes(header + masked_payload)

    decoded_text, opcode, remaining = LiveEventBroadcaster.decode_ws_frame(full_client_frame)
    assert decoded_text == message
    assert opcode == 0x1
    assert remaining == b""


# --- 4. Live Chat Tailer & Burst Linkage Tests ---

def test_live_chat_tailer_ingestion_and_bursts():
    async def _run():
        detector = RollingBurstDetector(window_sec=10.0, z_threshold=2.0)
        tailer = LiveChatTailer(
            channel_name="asmongold",
            burst_detector=detector,
            anonymous=True,
            stream_start_ms=1727500000000,
        )

        received_messages = []
        tailer.register_callback(lambda m: received_messages.append(m))

        # Ingest 15 chat lines
        for i in range(15):
            line = (
                f"@display-name=Viewer{i};user-id={i};tmi-sent-ts={1727500000000 + i * 500} "
                f":viewer{i}!v@v.tmi.twitch.tv PRIVMSG #asmongold :KEKW this is funny {i}"
            )
            await tailer.ingest_raw_line(line)

        assert tailer.total_messages == 15
        assert len(received_messages) == 15
        assert received_messages[0].author_name == "Viewer0"
        assert tailer.get_message_velocity() > 0.0

    asyncio.run(_run())


# --- 5. Live Stream Coordinator Lifecycle Tests ---

def test_live_stream_coordinator_lifecycle():
    async def _run():
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = LiveStreamConfig(
                channel_name="teststreamer",
                platform=LivePlatform.TWITCH,
                buffer_duration_sec=60.0,
                temp_dir=tmp_dir,
                ws_port=8788,
                sse_port=8789,
            )
            coord = LiveStreamCoordinator(config=cfg)
            assert coord.state == LiveState.STOPPED

            # Check status before start
            st = coord.get_status()
            assert st.state == LiveState.STOPPED
            assert st.channel_name == "teststreamer"

            # Start coordinator
            await coord.start()
            assert coord.state == LiveState.RUNNING

            # Inject synthetic media segment
            seg = coord.ingestor.ingest_synthetic_segment(
                duration_sec=10.0,
                video_content=b"test-video-bytes",
                audio_content=b"test-audio-bytes",
            )
            assert seg.duration_sec == 10.0
            assert coord.ingestor.total_bytes > 0

            # Inject chat lines
            await coord.chat_tailer.ingest_raw_line(
                "@display-name=Fan1;user-id=1 :fan1!f@f.tmi.twitch.tv PRIVMSG #teststreamer :PogChamp let's go"
            )
            assert coord.chat_tailer.total_messages == 1

            # Check status while running
            running_st = coord.get_status()
            assert running_st.state == LiveState.RUNNING
            assert running_st.total_chat_messages == 1
            assert running_st.total_bytes_ingested > 0

            # Stop coordinator
            await coord.stop()
            assert coord.state == LiveState.STOPPED

    asyncio.run(_run())


# --- 6. JSON-RPC 2.0 Integration Tests ---

def test_agent_rpc_live_tail_methods():
    dispatcher = AgentRpcDispatcher()

    # 1. startLiveTail
    req_start = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "streamfusion.startLiveTail",
        "params": {
            "channel_name": "rpc_channel",
            "platform": "TWITCH",
            "buffer_duration_sec": 60.0,
        },
    }
    resp_start = dispatcher.handle_request(req_start)
    assert "result" in resp_start
    stream_id = resp_start["result"]["stream_id"]
    assert resp_start["result"]["channel_name"] == "rpc_channel"

    # 2. getLiveStatus
    req_status = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "streamfusion.getLiveStatus",
        "params": {"stream_id": stream_id},
    }
    resp_status = dispatcher.handle_request(req_status)
    assert resp_status["result"]["stream_id"] == stream_id

    # 3. stopLiveTail
    req_stop = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "streamfusion.stopLiveTail",
        "params": {"stream_id": stream_id},
    }
    resp_stop = dispatcher.handle_request(req_stop)
    assert resp_stop["result"]["state"] in ["STOPPED", "STOPPING"]


# --- 7. CLI Subcommand Tests ---

def test_cli_live_help(cli_runner):
    result = cli_runner.invoke(app, ["live", "--help"])
    assert result.exit_code == 0
    assert "Real-Time Live Stream Ingestion" in result.output
    assert "tail" in result.output
    assert "status" in result.output


def test_cli_live_status(cli_runner):
    result = cli_runner.invoke(app, ["live", "status"])
    assert result.exit_code == 0
    assert "No local background daemon running" in result.output
