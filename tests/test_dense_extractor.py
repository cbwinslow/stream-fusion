"""Unit tests for Dense Frame Extraction & Web/Social Post Grounding (Spec 14)."""

import pytest
from stream_fusion.models.schemas import WordTiming
from stream_fusion.vision.dense_extractor import (
    DenseFrameExtractor,
    GameHUDParser,
    ReadAlongAligner,
    SocialCardParser,
)


def test_social_card_and_url_parser():
    parser = SocialCardParser()

    ocr_blocks = [
        "https://x.com/Zackrawrr/status/18392019283\nZack\n@Zackrawrr\nServers are finally back up. Jumping on to test the new raid boss.",
        "Bookmarks • Lists • Profile",
    ]

    cards = parser.parse_social_cards(ocr_blocks)
    assert len(cards) == 1
    card = cards[0]
    assert card.platform == "X_TWITTER"
    assert card.author_handle == "@Zackrawrr"
    assert card.author_name == "Zack"
    assert "Servers are finally back up" in card.post_text

    url, domain = parser.extract_browser_url(ocr_blocks)
    assert url == "https://x.com/Zackrawrr/status/18392019283"
    assert domain == "x.com"


def test_game_hud_parser():
    parser = GameHUDParser()

    ocr_blocks = [
        "Inventory - 14/30 Slots Used",
        "Asmongold defeated Mythic Raider (Critical Strike)",
    ]

    context = parser.parse_game_context(ocr_blocks, timestamp_sec=14.0)
    assert context.timestamp_sec == 14.0
    assert context.inventory_open is True
    assert len(context.killfeed_entries) == 1
    assert "defeated Mythic Raider" in context.killfeed_entries[0]


def test_read_along_aligner():
    aligner = ReadAlongAligner(min_alignment_threshold=0.6)

    # Streamer speaks the tweet aloud
    spoken = [
        WordTiming(word="Servers", start=10.0, end=10.4),
        WordTiming(word="are", start=10.4, end=10.6),
        WordTiming(word="finally", start=10.6, end=11.0),
        WordTiming(word="back", start=11.0, end=11.3),
        WordTiming(word="up", start=11.3, end=11.6),
    ]

    on_screen = "Servers are finally back up. Jumping on to test the raid."

    seg = aligner.compute_alignment(
        spoken_words=spoken,
        on_screen_text=on_screen,
        start_sec=10.0,
        end_sec=12.0,
        source_handle="@Zackrawrr",
    )

    assert seg is not None
    assert seg.alignment_score >= 0.8
    assert seg.source_handle == "@Zackrawrr"
    assert seg.reading_wpm > 100.0


def test_dense_frame_extractor_end_to_end():
    extractor = DenseFrameExtractor()

    ocr_blocks = [
        "https://x.com/Zackrawrr\n@Zackrawrr\nGood morning everyone!",
    ]
    spoken = [
        WordTiming(word="Good", start=1.0, end=1.3),
        WordTiming(word="morning", start=1.3, end=1.7),
        WordTiming(word="everyone", start=1.7, end=2.2),
    ]

    web, game, read_along = extractor.process_dense_frame(
        ocr_blocks=ocr_blocks,
        timestamp_sec=1.0,
        spoken_words_in_window=spoken,
    )

    assert web.browser_detected is True
    assert web.domain == "x.com"
    assert len(web.social_post_cards) == 1
    assert read_along is not None
    assert read_along.source_handle == "@Zackrawrr"
