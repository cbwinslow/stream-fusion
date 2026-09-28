"""Comprehensive Unit Test Suite for Spec 23: Co-Stream & Cross-Platform Alignment Subsystem."""

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import pytest
from typer.testing import CliRunner

from stream_fusion.cli import app
from stream_fusion.connectors.base import BaseChatConnector, ConnectorState, PlatformCapabilities
from stream_fusion.costream.audience_comparator import CrossAudienceComparator
from stream_fusion.costream.coordinator import MultiStreamCoordinator, MultiStreamSupervisor
from stream_fusion.costream.debate_analyzer import CoStreamDebateAnalyzer
from stream_fusion.costream.exceptions import CoStreamError, StreamSynchronizationError
from stream_fusion.costream.short_composer import MultiAngleShortComposer
from stream_fusion.costream.sync_engine import CrossStreamSyncEngine
from stream_fusion.models.schemas import (
    AudioSegment,
    ChatEmote,
    ChatMessage,
    CoStreamChannelConfig,
    CoStreamSessionConfig,
    LivePlatform,
    LiveStreamConfig,
)
from stream_fusion.schema.agent_rpc import AgentRpcDispatcher


# --- Mock Connector for Isolated Testing ---

class MockCoStreamConnector(BaseChatConnector):
    def __init__(self, channel_name: str, platform: LivePlatform = LivePlatform.TWITCH):
        super().__init__(channel_name=channel_name, platform=platform)
        self._caps = PlatformCapabilities(platform=platform)
        self.connected = False

    @property
    def capabilities(self) -> PlatformCapabilities:
        return self._caps

    async def _connect_and_listen(self) -> None:
        self.connected = True
        self._state = ConnectorState.RUNNING
        while self._running:
            await asyncio.sleep(0.05)

    async def _disconnect(self) -> None:
        self.connected = False
        self._state = ConnectorState.STOPPED

    async def push_mock_message(self, text: str, offset_sec: float, user: str = "viewer1", emotes=None) -> ChatMessage:
        msg = ChatMessage(
            message_id=f"msg-{self.channel_name}-{len(self._rolling_messages)}",
            timestamp_offset=offset_sec,
            user_id=f"u-{user}",
            author_name=user,
            content=text,
            emotes=emotes or [],
            metadata={"platform": self.platform.value},
        )
        await self.ingest_normalized_message(msg)
        return msg


# --- Test 1: CrossStreamSyncEngine Latency Calibration & Transformation ---

def test_sync_engine_manual_offsets_and_transform():
    engine = CrossStreamSyncEngine(reference_channel_id="asmongold")
    engine.set_manual_offset("xqc_kick", 3.5, confidence=0.95)
    engine.set_manual_offset("tim_yt", -2.0, confidence=0.90)

    assert engine.get_offset("asmongold") == 0.0
    assert engine.get_offset("xqc_kick") == 3.5
    assert engine.get_offset("tim_yt") == -2.0
    assert engine.get_confidence("xqc_kick") == 0.95

    # Test timestamp transformations: T_unified = T_local - offset
    # If xqc stream is 3.5s ahead, its local 10.0s corresponds to unified 6.5s
    assert engine.transform_timestamp("xqc_kick", 10.0) == 6.5
    assert engine.transform_timestamp("asmongold", 10.0) == 10.0
    assert engine.transform_timestamp("tim_yt", 10.0) == 12.0


def test_sync_engine_cross_correlation_synthetic():
    engine = CrossStreamSyncEngine()

    # Generate synthetic signals with a deliberate 2.0 second offset (step 0.5s -> 4 sample shift)
    base_sig = [0.0] * 5 + [10.0, 25.0, 50.0, 25.0, 10.0] + [0.0] * 20
    # Lagged signal shifted right by 4 bins (2.0s)
    lagged_sig = [0.0] * 9 + [10.0, 25.0, 50.0, 25.0, 10.0] + [0.0] * 16

    signals = {
        "stream_ref": base_sig,
        "stream_target": lagged_sig,
    }

    result = engine.calibrate_offsets(
        signals,
        reference_channel_id="stream_ref",
        sample_step_sec=0.5,
        max_lag_sec=10.0,
    )

    assert result.reference_channel_id == "stream_ref"
    assert result.channel_offsets["stream_ref"] == 0.0
    # Expected target offset should be +2.0 seconds
    assert pytest.approx(result.channel_offsets["stream_target"], abs=0.1) == 2.0
    assert result.confidence_scores["stream_target"] > 0.85


def test_sync_engine_trigger_event_calibration():
    engine = CrossStreamSyncEngine()

    # Shared trigger events occurred at 10.0, 25.0, 50.0 on reference
    # Target stream was delayed by 3.2 seconds
    triggers = {
        "twitch_ref": [10.0, 25.0, 50.0],
        "kick_target": [13.2, 28.2, 53.2],
    }

    res = engine.calibrate_from_trigger_events(triggers, reference_channel_id="twitch_ref")
    assert res.reference_channel_id == "twitch_ref"
    assert pytest.approx(res.channel_offsets["kick_target"], abs=0.1) == 3.2
    assert res.confidence_scores["kick_target"] >= 0.90


def test_sync_engine_align_messages_and_audio():
    engine = CrossStreamSyncEngine(reference_channel_id="ch_a")
    engine.set_manual_offset("ch_b", 4.0)

    # Channel A: local msgs at t=5.0, t=15.0 -> unified 5.0, 15.0
    # Channel B: local msgs at t=8.0, t=16.0 -> unified 4.0, 12.0
    msg_a1 = ChatMessage(message_id="a1", timestamp_offset=5.0, user_id="u1", author_name="a", content="hello")
    msg_a2 = ChatMessage(message_id="a2", timestamp_offset=15.0, user_id="u1", author_name="a", content="world")
    msg_b1 = ChatMessage(message_id="b1", timestamp_offset=8.0, user_id="u2", author_name="b", content="pog")
    msg_b2 = ChatMessage(message_id="b2", timestamp_offset=16.0, user_id="u2", author_name="b", content="based")

    aligned = engine.align_messages({
        "ch_a": [msg_a1, msg_a2],
        "ch_b": [msg_b1, msg_b2],
    })

    assert len(aligned) == 4
    # Chronological unified order:
    # 1. b1 (4.0s)
    # 2. a1 (5.0s)
    # 3. b2 (12.0s)
    # 4. a2 (15.0s)
    assert aligned[0][0] == 4.0 and aligned[0][1] == "ch_b"
    assert aligned[1][0] == 5.0 and aligned[1][1] == "ch_a"
    assert aligned[2][0] == 12.0 and aligned[2][1] == "ch_b"
    assert aligned[3][0] == 15.0 and aligned[3][1] == "ch_a"

    # Audio alignment
    seg_a = AudioSegment(segment_id=1, start_sec=10.0, end_sec=15.0, speaker_label="SPK1", transcript="test a")
    seg_b = AudioSegment(segment_id=2, start_sec=12.0, end_sec=17.0, speaker_label="SPK2", transcript="test b")
    aligned_audio = engine.align_audio_segments({
        "ch_a": [seg_a],
        "ch_b": [seg_b],
    })
    # seg_b start 12.0 - 4.0 = 8.0s -> comes before seg_a start 10.0s
    assert aligned_audio[0][0] == 8.0 and aligned_audio[0][2] == "ch_b"
    assert aligned_audio[1][0] == 10.0 and aligned_audio[1][2] == "ch_a"


# --- Test 2: CrossAudienceComparator Sentiment & Divergence ---

def test_audience_comparator_consensus():
    comparator = CrossAudienceComparator(bucket_window_sec=2.0)

    # Both Twitch and Kick audiences react with high hype / positive sentiment
    msg1 = ChatMessage(
        message_id="1", timestamp_offset=2.0, user_id="u1", author_name="u1",
        content="POGGERS this game is so based", emotes=[ChatEmote(id="e1", name="POGGERS")]
    )
    msg2 = ChatMessage(
        message_id="2", timestamp_offset=2.2, user_id="u2", author_name="u2",
        content="W W W letsgo gigachad", emotes=[]
    )

    aligned = [
        (2.0, "twitch_ch", msg1),
        (2.2, "kick_ch", msg2),
    ]
    ch_plat = {"twitch_ch": "TWITCH", "kick_ch": "KICK"}

    timeline = comparator.compute_aligned_sentiment_timeline(aligned, channel_to_platform=ch_plat)
    assert len(timeline) >= 1
    point = timeline[0]

    assert point.platform_sentiments["TWITCH"] > 0.6
    assert point.platform_sentiments["KICK"] > 0.6
    assert point.cross_platform_agreement >= 0.85
    assert not point.divergence_detected


def test_audience_comparator_divergence():
    comparator = CrossAudienceComparator(bucket_window_sec=2.0, divergence_threshold=0.75)

    # YouTube loves it (positive), Twitch despises it (cringe/cooked)
    yt_msg = ChatMessage(
        message_id="yt1", timestamp_offset=4.0, user_id="y1", author_name="yt_user",
        content="Awesome update W W W based game", emotes=[]
    )
    tw_msg = ChatMessage(
        message_id="tw1", timestamp_offset=4.5, user_id="t1", author_name="tw_user",
        content="L cringe cooked trash awful", emotes=[ChatEmote(id="e2", name="cringe")]
    )

    aligned = [
        (4.0, "youtube_ch", yt_msg),
        (4.5, "twitch_ch", tw_msg),
    ]
    ch_plat = {"youtube_ch": "YOUTUBE_LIVE", "twitch_ch": "TWITCH"}

    timeline = comparator.compute_aligned_sentiment_timeline(aligned, channel_to_platform=ch_plat)
    assert len(timeline) >= 1
    point = timeline[0]

    assert point.platform_sentiments["YOUTUBE_LIVE"] > 0.5
    assert point.platform_sentiments["TWITCH"] < -0.5
    assert point.divergence_detected is True
    assert "Audience divergence detected" in (point.divergence_description or "")
    assert point.cross_platform_agreement < 0.55


def test_track_meme_cascade_propagation():
    comparator = CrossAudienceComparator()

    # Emote OMEGALUL first bursts on Twitch at t=10.0s, then Kick at t=12.5s, then YouTube at t=15.0s
    aligned = [
        (10.0, "twitch_ch", ChatMessage(message_id="1", timestamp_offset=10.0, user_id="1", author_name="u", content="OMEGALUL")),
        (10.2, "twitch_ch", ChatMessage(message_id="2", timestamp_offset=10.2, user_id="2", author_name="u", content="OMEGALUL so funny")),
        (12.5, "kick_ch", ChatMessage(message_id="3", timestamp_offset=12.5, user_id="3", author_name="u", content="OMEGALUL")),
        (15.0, "yt_ch", ChatMessage(message_id="4", timestamp_offset=15.0, user_id="4", author_name="u", content="OMEGALUL in chat")),
    ]
    ch_plat = {"twitch_ch": "TWITCH", "kick_ch": "KICK", "yt_ch": "YOUTUBE_LIVE"}

    cascade = comparator.track_meme_cascade(
        aligned,
        target_token="OMEGALUL",
        channel_to_platform=ch_plat,
        burst_threshold=1,
    )

    assert cascade is not None
    assert cascade.token == "OMEGALUL"
    assert cascade.origin_platform == "TWITCH"
    assert cascade.origin_channel == "twitch_ch"
    assert cascade.origin_timestamp == 10.0
    assert cascade.cascade_timeline["KICK:kick_ch"] == 2.5
    assert cascade.cascade_timeline["YOUTUBE_LIVE:yt_ch"] == 5.0
    assert cascade.cascade_velocity_sec > 0.0


# --- Test 3: CoStreamDebateAnalyzer Turn-Taking & Debate ---

def test_debate_analyzer_turn_taking_and_interrupts():
    analyzer = CoStreamDebateAnalyzer(min_turn_duration_sec=0.5, interrupt_overlap_sec=0.3)

    # Segment 1: Streamer A speaks from 1.0s to 5.0s
    # Segment 2: Streamer B interrupts at 4.2s (while A is still speaking) to 8.0s
    # Segment 3: Streamer A replies from 8.2s to 12.0s
    seg1 = AudioSegment(segment_id=1, start_sec=1.0, end_sec=5.0, speaker_label="Asmon", transcript="This is completely based and awesome W")
    seg2 = AudioSegment(segment_id=2, start_sec=4.2, end_sec=8.0, speaker_label="xQc", transcript="No shot man that is trash and cooked L")
    seg3 = AudioSegment(segment_id=3, start_sec=8.2, end_sec=12.0, speaker_label="Asmon", transcript="I disagree it is good")

    aligned_audio = [
        (1.0, 5.0, "asmongold", seg1),
        (4.2, 8.0, "xqc", seg2),
        (8.2, 12.0, "asmongold", seg3),
    ]

    analysis = analyzer.analyze_debate(aligned_audio)
    assert len(analysis["turns"]) == 3
    # Turn 2 should have interrupt flagged
    assert analysis["turns"][1]["interrupts_previous"] is True
    assert analysis["turns"][0]["interrupts_previous"] is False

    # Talk time
    assert "Asmon" in analysis["talk_time_sec"]
    assert "xQc" in analysis["talk_time_sec"]
    assert analysis["talk_time_sec"]["Asmon"] == 7.8
    assert analysis["talk_time_sec"]["xQc"] == 3.8

    # Debate moments detected (Asmon positive -> xQc negative)
    assert len(analysis["debate_moments"]) >= 1
    moment = analysis["debate_moments"][0]
    assert moment["speaker_1"] == "Asmon"
    assert moment["speaker_2"] == "xQc"
    assert moment["delta"] > 0.6


# --- Test 4: MultiAngleShortComposer Climax Detection ---

def test_multi_angle_short_composer():
    composer = MultiAngleShortComposer(min_duration_sec=15.0)

    # Construct aligned messages and points
    msgs = []
    for t in [10.0, 10.5, 11.0, 11.5, 12.0]:
        msgs.append((t, "ch_a", ChatMessage(message_id=f"a_{t}", timestamp_offset=t, user_id="u1", author_name="u", content="W POGGERS")))
        msgs.append((t, "ch_b", ChatMessage(message_id=f"b_{t}", timestamp_offset=t, user_id="u2", author_name="u", content="W POGGERS")))

    comparator = CrossAudienceComparator(bucket_window_sec=2.0)
    timeline = comparator.compute_aligned_sentiment_timeline(msgs, channel_to_platform={"ch_a": "TWITCH", "ch_b": "KICK"})

    candidates = composer.find_multi_angle_candidates(timeline, msgs, top_k=2)
    assert len(candidates) >= 1
    cand = candidates[0]

    assert cand.duration_sec >= 15.0
    assert cand.layout_preset == "STACKED_SPLIT"
    assert cand.virality_score > 50.0
    assert len(cand.hooks) > 0


# --- Test 5: MultiStreamCoordinator Lifecycle & Resource Management ---

@pytest.mark.anyio
async def test_multistream_coordinator_lifecycle_and_bounded_resources():
    ch_cfg1 = CoStreamChannelConfig(
        channel_id="asmongold",
        stream_config=LiveStreamConfig(channel_name="asmongold", platform=LivePlatform.TWITCH),
        is_reference_stream=True,
    )
    ch_cfg2 = CoStreamChannelConfig(
        channel_id="xqc",
        stream_config=LiveStreamConfig(channel_name="xqc", platform=LivePlatform.KICK),
        manual_latency_offset=2.5,
    )

    session_cfg = CoStreamSessionConfig(
        session_id="test-session-001",
        session_title="Gaming Awards Co-Stream",
        channels=[ch_cfg1, ch_cfg2],
        auto_sync=False,
    )

    coordinator = MultiStreamCoordinator(config=session_cfg)

    # Inject mock connectors into supervisors
    mock_conn1 = MockCoStreamConnector("asmongold", LivePlatform.TWITCH)
    mock_conn2 = MockCoStreamConnector("xqc", LivePlatform.KICK)
    coordinator.supervisors["asmongold"].chat_connector = mock_conn1
    coordinator.supervisors["xqc"].chat_connector = mock_conn2

    # Start session
    await coordinator.start()
    assert coordinator.is_active is True
    assert coordinator.supervisors["asmongold"].state == ConnectorState.RUNNING
    assert coordinator.supervisors["xqc"].state == ConnectorState.RUNNING

    # Push messages through mock connectors
    msg1 = await mock_conn1.push_mock_message("POGGERS game reveal!", offset_sec=10.0, user="viewer_a")
    msg2 = await mock_conn2.push_mock_message("W update let's go", offset_sec=12.5, user="viewer_b")

    # Give async event loop a moment to fanout
    await asyncio.sleep(0.05)

    # Check status
    status = coordinator.get_status()
    assert status.session_id == "test-session-001"
    assert status.is_active is True
    assert status.total_messages == 2
    assert "asmongold" in status.channels
    assert "xqc" in status.channels
    assert status.channels["xqc"].calibrated_latency_offset == 2.5

    # Check bounded message queue limit
    sup = coordinator.supervisors["asmongold"]
    sup.message_queue = asyncio.Queue(maxsize=3)
    for i in range(10):
        await mock_conn1.push_mock_message(f"Spam message {i}", offset_sec=20.0 + i)

    # Queue should be safely bounded at maxsize without throwing QueueFull
    assert sup.message_queue.qsize() <= 3

    # Teardown
    await coordinator.stop()
    assert coordinator.is_active is False
    assert coordinator.supervisors["asmongold"].state == ConnectorState.STOPPED
    assert coordinator.supervisors["xqc"].state == ConnectorState.STOPPED


# --- Test 6: AgentRpcDispatcher Co-Stream Methods ---

def test_agent_rpc_costream_handlers():
    dispatcher = AgentRpcDispatcher()

    # 1. Align co-streams via RPC
    req_align = {
        "jsonrpc": "2.0",
        "method": "streamfusion.alignCoStreams",
        "params": {
            "reference_channel_id": "ref",
            "channel_triggers": {
                "ref": [10.0, 20.0, 30.0],
                "guest": [14.0, 24.0, 34.0],
            },
        },
        "id": 101,
    }
    resp_align_str = dispatcher.handle_line(json.dumps(req_align))
    resp_align = json.loads(resp_align_str)

    assert "result" in resp_align
    assert resp_align["result"]["reference_channel_id"] == "ref"
    assert pytest.approx(resp_align["result"]["channel_offsets"]["guest"], abs=0.1) == 4.0

    # 2. Compare cross-audience sentiment via RPC
    raw_msgs = [
        [5.0, "ch1", {"message_id": "m1", "timestamp_offset": 5.0, "user_id": "u1", "author_name": "a", "content": "POG W"}],
        [5.2, "ch2", {"message_id": "m2", "timestamp_offset": 5.2, "user_id": "u2", "author_name": "b", "content": "POG W"}],
    ]
    req_compare = {
        "jsonrpc": "2.0",
        "method": "streamfusion.compareCrossAudience",
        "params": {
            "aligned_messages": raw_msgs,
            "channel_to_platform": {"ch1": "TWITCH", "ch2": "KICK"},
            "bucket_window_sec": 2.0,
        },
        "id": 102,
    }
    resp_compare_str = dispatcher.handle_line(json.dumps(req_compare))
    resp_compare = json.loads(resp_compare_str)

    assert "result" in resp_compare
    assert len(resp_compare["result"]["timeline"]) >= 1
    assert resp_compare["result"]["summary"]["overall_agreement_index"] >= 0.8


# --- Test 7: CLI Subcommands Verification ---

def test_cli_costream_help():
    runner = CliRunner()
    result = runner.invoke(app, ["costream", "--help"])
    assert result.exit_code == 0
    assert "align" in result.output
    assert "compare" in result.output
    assert "start" in result.output


def test_cli_costream_align_and_compare(tmp_path: Path):
    runner = CliRunner()

    # Create dummy stream analysis JSON files
    s1_data = {
        "chat_messages": [
            {"message_id": "s1_1", "timestamp_offset": 10.0, "user_id": "u1", "author_name": "a", "content": "OMEGALUL so funny", "emotes": [{"id": "e1", "name": "OMEGALUL", "count": 1}]},
            {"message_id": "s1_2", "timestamp_offset": 25.0, "user_id": "u1", "author_name": "a", "content": "POGGERS huge update", "emotes": [{"id": "e2", "name": "POGGERS", "count": 1}]},
        ]
    }
    s2_data = {
        "chat_messages": [
            {"message_id": "s2_1", "timestamp_offset": 13.0, "user_id": "u2", "author_name": "b", "content": "OMEGALUL funny", "emotes": [{"id": "e1", "name": "OMEGALUL", "count": 1}]},
            {"message_id": "s2_2", "timestamp_offset": 28.0, "user_id": "u2", "author_name": "b", "content": "POGGERS based", "emotes": [{"id": "e2", "name": "POGGERS", "count": 1}]},
        ]
    }

    f1 = tmp_path / "asmon_analysis.json"
    f2 = tmp_path / "xqc_analysis.json"
    f1.write_text(json.dumps(s1_data), encoding="utf-8")
    f2.write_text(json.dumps(s2_data), encoding="utf-8")

    out_aligned = tmp_path / "costream_aligned.json"

    # Run CLI align
    res_align = runner.invoke(app, [
        "costream", "align",
        "-s", str(f1),
        "-s", str(f2),
        "-r", "asmon",
        "-o", str(out_aligned),
    ])
    assert res_align.exit_code == 0
    assert out_aligned.exists()

    # Run CLI compare
    res_compare = runner.invoke(app, [
        "costream", "compare",
        "-a", str(out_aligned),
        "-w", "5.0",
    ])
    assert res_compare.exit_code == 0
    assert "Cross-Platform Audience Timeline" in res_compare.output
