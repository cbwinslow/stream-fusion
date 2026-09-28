"""Unit tests for Multi-Platform Live Stream & Chat Connectors (Spec 22)."""

import asyncio
from datetime import datetime, timezone
import json
import time
from typing import Any, Dict

import pytest

from stream_fusion.connectors.base import BaseChatConnector
from stream_fusion.connectors.kick.pusher_connector import KickChatConnector
from stream_fusion.connectors.kick.resolver import KickChannelResolver
from stream_fusion.connectors.kick.webhook_receiver import KickWebhookReceiver
from stream_fusion.connectors.normalizer import MessageNormalizer
from stream_fusion.connectors.registry import ConnectorRegistry
from stream_fusion.connectors.twitch.irc_connector import TwitchChatConnector
from stream_fusion.connectors.youtube.chat_downloader import YouTubeChatConnector
from stream_fusion.connectors.youtube.data_api import YouTubeDataApiConnector
from stream_fusion.ingest.live_coordinator import LiveStreamCoordinator
from stream_fusion.models.schemas import (
    ChatMessage,
    ConnectorState,
    LivePlatform,
    LiveState,
    LiveStreamConfig,
)
from stream_fusion.nlp.adaptive_slang import RollingBurstDetector


# --- 1. Message Normalizer Tests ---

def test_message_normalizer_kick_pusher():
    payload = {
        "id": "kick-msg-1234",
        "chatroom_id": 668,
        "content": "Hey chat [emote:2506823:azzzjh] what a play [emote:2506823:azzzjh] Pog",
        "created_at": "2026-09-28T16:00:00Z",
        "sender": {
            "id": 99999,
            "username": "KickFan42",
            "slug": "kickfan42",
            "identity": {
                "badges": [
                    {"type": "moderator"},
                    {"type": "subscriber", "count": 6},
                ]
            },
        },
    }

    start_ms = datetime(2026, 9, 28, 15, 59, 30, tzinfo=timezone.utc).timestamp() * 1000.0
    msg = MessageNormalizer.from_kick_pusher(payload, stream_start_ms=start_ms)

    assert msg is not None
    assert msg.message_id == "kick-msg-1234"
    assert msg.author_name == "KickFan42"
    assert msg.user_id == "99999"
    assert msg.timestamp_offset == 30.0  # 16:00:00 - 15:59:30
    assert "moderator" in msg.badges
    assert "subscriber/6" in msg.badges
    assert len(msg.emotes) == 1
    assert msg.emotes[0].id == "2506823"
    assert msg.emotes[0].name == "azzzjh"
    assert msg.emotes[0].count == 2
    assert msg.metadata.get("platform") == "KICK"
    assert msg.metadata.get("chatroom_id") == 668


def test_message_normalizer_youtube_chat_downloader_item():
    item = {
        "message_id": "yt-msg-777",
        "time_in_seconds": 45.5,
        "author": {
            "id": "UC_TEST_CHANNEL_ID",
            "name": "SuperFan",
            "badges": [{"title": "Member"}, "Moderator"],
        },
        "message": [
            {"text": "Huge moment! "},
            {"emoji": {"emoji_id": "emoji_fire", "shortcuts": [":fire:"]}},
        ],
        "money": {
            "amount": 25.0,
            "currency": "USD",
            "text": "$25.00",
        },
    }

    msg = MessageNormalizer.from_youtube_item(item)
    assert msg is not None
    assert msg.message_id == "yt-msg-777"
    assert msg.timestamp_offset == 45.5
    assert msg.author_name == "SuperFan"
    assert msg.user_id == "UC_TEST_CHANNEL_ID"
    assert "Member" in msg.badges
    assert "Moderator" in msg.badges
    assert msg.content == "Huge moment! :fire:"
    assert len(msg.emotes) == 1
    assert msg.emotes[0].name == ":fire:"
    assert msg.metadata.get("platform") == "YOUTUBE_LIVE"
    assert msg.metadata.get("monetization", {}).get("type") == "SUPER_CHAT"
    assert msg.metadata["monetization"]["amount"] == 25.0


def test_message_normalizer_youtube_innertube_renderer():
    renderer = {
        "id": "innertube-msg-1",
        "authorName": {"simpleText": "VODReviewer"},
        "authorExternalChannelId": "UC_REVIEWER",
        "authorBadges": [
            {"liveChatAuthorBadgeRenderer": {"tooltip": "Verified Channel"}}
        ],
        "message": {
            "runs": [
                {"text": "Let's go "},
                {"emoji": {"emojiId": "clap_icon", "shortcuts": [":clap:"]}},
            ]
        },
        "timestampUsec": "1700000050000000",
        "purchaseAmountText": {"simpleText": "$5.00"},
        "headerBackgroundColor": 4278190080,
    }

    raw_item = {"liveChatPaidMessageRenderer": renderer}
    start_ms = (1700000000.0) * 1000.0
    msg = MessageNormalizer.from_youtube_item(raw_item, stream_start_ms=start_ms)

    assert msg is not None
    assert msg.message_id == "innertube-msg-1"
    assert msg.author_name == "VODReviewer"
    assert msg.user_id == "UC_REVIEWER"
    assert "Verified Channel" in msg.badges
    assert msg.content == "Let's go :clap:"
    assert round(msg.timestamp_offset, 1) == 50.0
    assert msg.metadata["monetization"]["text"] == "$5.00"


def test_message_normalizer_twitch_irc():
    tags = {
        "id": "twitch-msg-888",
        "display-name": "TwitchGamer",
        "user-id": "112233",
        "badges": "broadcaster/1,subscriber/12",
        "emotes": "25:0-4",
        "bits": "100",
        "tmi-sent-ts": "1700000010000",
    }
    content = "Kappa clutch round!"
    start_ms = 1700000000.0 * 1000.0

    msg = MessageNormalizer.from_twitch_irc(tags, prefix="tg!tg@tmi.twitch.tv", content=content, stream_start_ms=start_ms)
    assert msg.message_id == "twitch-msg-888"
    assert msg.author_name == "TwitchGamer"
    assert msg.timestamp_offset == 10.0
    assert "broadcaster/1" in msg.badges
    assert len(msg.emotes) == 1
    assert msg.emotes[0].id == "25"
    assert msg.emotes[0].name == "Kappa"
    assert msg.metadata["platform"] == "TWITCH"
    assert msg.metadata["monetization"]["type"] == "BITS"
    assert msg.metadata["monetization"]["amount"] == 100.0


# --- 2. Kick Channel Resolver Tests ---

def test_kick_channel_resolver():
    resolver = KickChannelResolver()

    # 1. Direct numeric ID string
    assert resolver.resolve("123456") == 123456

    # 2. Known cached streamer
    assert resolver.resolve("xqc") == 668
    assert resolver.resolve("adinross") == 182

    # 3. Manual override
    assert resolver.resolve("xqc", manual_override=9999) == 9999

    # 4. Deterministic fallback for unknown offline streamer
    res1 = resolver.resolve("unknown_streamer_alpha")
    res2 = resolver.resolve("unknown_streamer_alpha")
    assert res1 > 0
    assert res1 == res2


# --- 3. Kick Pusher Connector Tests ---

def test_kick_pusher_connector_lifecycle():
    async def _test():
        burst_detector = RollingBurstDetector(window_sec=5.0, z_threshold=2.0)
        connector = KickChatConnector(
            channel_name="test_channel",
            chatroom_id=123456,
            burst_detector=burst_detector,
        )

        assert connector.capabilities.platform == LivePlatform.KICK
        assert connector.capabilities.supports_emotes is True
        assert connector.state == ConnectorState.STOPPED

        received_messages = []
        connector.register_callback(lambda m: received_messages.append(m))

        bursts_detected = []
        connector.register_burst_callback(lambda b: bursts_detected.append(b))

        # 1. Test Pusher handshake events
        await connector.ingest_raw_event("pusher:connection_established", {})
        await connector.ingest_raw_event("pusher_internal:subscription_succeeded", {})
        assert connector.is_subscribed is True

        # 2. Test keepalive ping/pong
        await connector.ingest_raw_event("pusher:ping", {})

        # 3. Test incoming chat messages
        for i in range(5):
            await connector.ingest_raw_event(
                "App\\Events\\ChatMessageEvent",
                {
                    "id": f"k-{i}",
                    "chatroom_id": 123456,
                    "content": "POGGERS [emote:99:pog] incredible",
                    "sender": {"id": 100 + i, "username": f"Viewer{i}"},
                },
            )

        assert len(received_messages) == 5
        assert connector.total_messages == 5
        assert connector.get_message_velocity(window_sec=10.0) > 0.0

        await connector.stop()
        assert connector.state == ConnectorState.STOPPED

    asyncio.run(_test())


# --- 4. Kick Webhook Receiver Tests ---

def test_kick_webhook_receiver():
    secret = "my_webhook_secret_key"
    receiver = KickWebhookReceiver(webhook_secret=secret)

    payload = json.dumps({
        "event": "chat.message.sent",
        "data": {
            "id": "wh-msg-1",
            "chatroom_id": 555,
            "content": "Hello from official webhook!",
            "sender": {"id": 12, "username": "ApiBot"},
        }
    }).encode("utf-8")

    # 1. Invalid signature
    assert receiver.parse_webhook_payload(payload, signature_header="invalid_sig") is None

    # 2. Valid signature
    import hmac
    import hashlib
    valid_sig = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()

    msg = receiver.parse_webhook_payload(payload, signature_header=valid_sig)
    assert msg is not None
    assert msg.message_id == "wh-msg-1"
    assert msg.author_name == "ApiBot"
    assert msg.content == "Hello from official webhook!"


# --- 5. YouTube Live Chat Connector Tests ---

def test_youtube_chat_downloader_connector():
    async def _test():
        connector = YouTubeChatConnector(channel_or_url="@Asmongold")
        assert connector.target_url == "https://www.youtube.com/@Asmongold/live"
        assert connector.capabilities.platform == LivePlatform.YOUTUBE_LIVE
        assert connector.capabilities.supports_superchats is True

        received = []
        connector.register_callback(lambda m: received.append(m))

        # Ingest mock raw items
        for i in range(3):
            await connector.ingest_raw_item({
                "message_id": f"yt-{i}",
                "author": {"name": f"User{i}", "id": f"UC{i}"},
                "message": "OMEGALUL so funny",
                "time_in_seconds": float(i * 2),
            })

        assert len(received) == 3
        assert connector.total_messages == 3
        assert connector.get_message_velocity() > 0.0

        await connector.stop()
        assert connector.state == ConnectorState.STOPPED

    asyncio.run(_test())


def test_youtube_data_api_connector_normalization():
    connector = YouTubeDataApiConnector(live_chat_id="LIVEChat123", api_key="dummy_key")
    item = {
        "id": "api-msg-101",
        "snippet": {
            "displayMessage": "Official API chat test",
            "publishedAt": "2026-09-28T16:05:00Z",
            "superChatDetails": {
                "amountMicros": 10000000,
                "currency": "USD",
                "amountDisplayString": "$10.00",
            },
        },
        "authorDetails": {
            "displayName": "VIPViewer",
            "channelId": "UC_VIP",
            "isChatSponsor": True,
            "isVerified": True,
        },
    }

    msg = connector.normalize_api_item(item)
    assert msg is not None
    assert msg.message_id == "api-msg-101"
    assert msg.author_name == "VIPViewer"
    assert "Member" in msg.badges
    assert "Verified" in msg.badges
    assert msg.metadata["monetization"]["amount"] == 10.0
    assert msg.metadata["monetization"]["currency"] == "USD"


# --- 6. Twitch Chat Connector Tests ---

def test_twitch_chat_connector():
    async def _test():
        connector = TwitchChatConnector(channel_name="#twitchstreamer")
        assert connector.channel_name == "twitchstreamer"
        assert connector.capabilities.platform == LivePlatform.TWITCH
        assert connector.capabilities.supports_bits is True

        received = []
        connector.register_callback(lambda m: received.append(m))

        line = "@display-name=TwitchUser;user-id=4455;emotes= :twitchuser!t@t.tmi.twitch.tv PRIVMSG #twitchstreamer :hello world"
        msg = await connector.ingest_raw_line(line)
        assert msg is not None
        assert msg.author_name == "TwitchUser"
        assert msg.content == "hello world"
        assert len(received) == 1
        assert connector.total_messages == 1

        await connector.stop()
        assert connector.state == ConnectorState.STOPPED

    asyncio.run(_test())


# --- 7. Connector Registry Factory Tests ---

def test_connector_registry_factory():
    # 1. Twitch
    cfg_twitch = LiveStreamConfig(channel_name="asmongold", platform=LivePlatform.TWITCH)
    conn_twitch = ConnectorRegistry.create_chat_connector(cfg_twitch)
    assert isinstance(conn_twitch, TwitchChatConnector)

    # 2. Kick
    cfg_kick = LiveStreamConfig(channel_name="xqc", platform=LivePlatform.KICK, chatroom_id=668)
    conn_kick = ConnectorRegistry.create_chat_connector(cfg_kick)
    assert isinstance(conn_kick, KickChatConnector)
    assert conn_kick.chatroom_id == 668

    # 3. YouTube Chat Downloader
    cfg_yt = LiveStreamConfig(channel_name="@LofiGirl", platform=LivePlatform.YOUTUBE_LIVE)
    conn_yt = ConnectorRegistry.create_chat_connector(cfg_yt)
    assert isinstance(conn_yt, YouTubeChatConnector)

    # 4. YouTube Data API v3
    cfg_yt_api = LiveStreamConfig(channel_name="chat123", platform=LivePlatform.YOUTUBE_LIVE, youtube_api_key="AIzaSy...")
    conn_yt_api = ConnectorRegistry.create_chat_connector(cfg_yt_api)
    assert isinstance(conn_yt_api, YouTubeDataApiConnector)


# --- 8. Multi-Platform LiveStreamCoordinator End-to-End ---

def test_live_stream_coordinator_with_kick_and_youtube(tmp_path):
    async def _test():
        # A. Coordinator with Kick
        cfg_kick = LiveStreamConfig(
            channel_name="xqc",
            platform=LivePlatform.KICK,
            chatroom_id=668,
            temp_dir=str(tmp_path / "kick"),
            ws_port=8791,
            sse_port=8792,
        )
        coord_kick = LiveStreamCoordinator(config=cfg_kick)
        assert isinstance(coord_kick.chat_tailer, KickChatConnector)
        await coord_kick.start()
        assert coord_kick.state == LiveState.RUNNING

        # Ingest Kick event through coordinator's chat connector
        await coord_kick.chat_tailer.ingest_raw_event(
            "App\\Events\\ChatMessageEvent",
            {
                "id": "coord-kick-1",
                "chatroom_id": 668,
                "content": "Kick test [emote:1:kekw]",
                "sender": {"id": 1, "username": "KickUser"},
            },
        )
        status_kick = coord_kick.get_status()
        assert status_kick.total_chat_messages == 1
        assert status_kick.platform == LivePlatform.KICK
        await coord_kick.stop()
        assert coord_kick.state == LiveState.STOPPED

        # B. Coordinator with YouTube
        cfg_yt = LiveStreamConfig(
            channel_name="UC_TEST",
            platform=LivePlatform.YOUTUBE_LIVE,
            temp_dir=str(tmp_path / "yt"),
            ws_port=8793,
            sse_port=8794,
        )
        coord_yt = LiveStreamCoordinator(config=cfg_yt)
        assert isinstance(coord_yt.chat_tailer, YouTubeChatConnector)
        await coord_yt.start()
        assert coord_yt.state == LiveState.RUNNING

        await coord_yt.chat_tailer.ingest_raw_item({
            "message_id": "coord-yt-1",
            "author": {"name": "YTUser", "id": "UC1"},
            "message": "YouTube test",
        })
        status_yt = coord_yt.get_status()
        assert status_yt.total_chat_messages == 1
        assert status_yt.platform == LivePlatform.YOUTUBE_LIVE
        await coord_yt.stop()
        assert coord_yt.state == LiveState.STOPPED

    asyncio.run(_test())
