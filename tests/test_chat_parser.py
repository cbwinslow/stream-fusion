"""Comprehensive tests for the chat parser and analyzer using fixtures."""

from pathlib import Path
from stream_fusion.chat.analyzer import ChatAnalyzer, EMOTE_POLARITY_LEXICON


def test_parse_twitch_downloader_json_fixture():
    fixture_path = Path(__file__).parent / "fixtures" / "sample_twitch_chat.json"
    assert fixture_path.exists(), f"Fixture file not found: {fixture_path}"

    analyzer = ChatAnalyzer(latency_offset_sec=4.0)
    messages = analyzer.parse_twitch_downloader_json(fixture_path)

    # 1. Verify message count and structure
    assert len(messages) == 5
    assert messages[0].author_name == "Gamer123"
    assert messages[0].timestamp_offset == 1.2
    assert messages[0].content == "pepeJAM chill stream"

    # 2. Verify emote extraction
    assert len(messages[2].emotes) == 1
    assert messages[2].emotes[0].name == "OMEGALUL"
    assert messages[2].emotes[0].id == "128054"

    # 3. Verify latency calibration & bucket aggregation
    # Message at 19.1s with 4.0s latency offset calibrates to 15.1s.
    # In 2-second buckets, 15.1s falls into bucket index 7 (14.0s - 16.0s)
    buckets = analyzer.aggregate_into_buckets(messages, stream_duration_sec=30.0, bucket_size_sec=2.0)
    
    b7 = buckets[7]
    assert b7["chat_message_count"] == 2  # messages at 19.1s and 19.5s
    assert "OMEGALUL" in b7["dominant_emotes"]
    assert b7["chat_sentiment_polarity"] > 0.5  # OMEGALUL is positive amusement


def test_empty_chat_handling():
    analyzer = ChatAnalyzer(latency_offset_sec=4.0)
    buckets = analyzer.aggregate_into_buckets([], stream_duration_sec=10.0, bucket_size_sec=2.0)
    assert len(buckets) == 6
    for b in buckets:
        assert b["chat_message_count"] == 0
        assert b["chat_velocity_per_sec"] == 0.0
        assert b["chat_sentiment_polarity"] == 0.0
