"""Unit tests for Sponsor & Brand Performance Quantifier (Spec 09)."""

import pytest
from stream_fusion.analytics.sponsor_quantifier import (
    SponsorDetector,
    SponsorReportGenerator,
    fuzzy_match_token,
)
from stream_fusion.models.schemas import (
    AudioSegment,
    BrandProfile,
    ChatMessage,
    VisualKeyframe,
)


@pytest.fixture
def sample_brand():
    return BrandProfile(
        brand_id="starforge_systems",
        brand_name="Starforge Systems",
        aliases=["starforge", "star forge"],
        promo_codes=["ASMON", "STREAMFUSION"],
        product_keywords=["pc", "computer", "rig", "gpu"],
    )


def test_fuzzy_match_token():
    assert fuzzy_match_token("starforge", "Go check out starforge today!")
    assert fuzzy_match_token("ASMON", "Use code asmon for 10% off")
    assert fuzzy_match_token("starforge", "STARFORGE systems are great")
    assert not fuzzy_match_token("starforge", "completely unrelated phrase")


def test_sponsor_detector_audio_and_ocr(sample_brand):
    detector = SponsorDetector(merge_threshold_sec=30.0)

    # Audio mention at t=10s
    audio_segments = [
        AudioSegment(
            segment_id=1,
            start_sec=10.0,
            end_sec=15.0,
            speaker_label="STREAMER",
            transcript="Shoutout to Starforge Systems for sponsoring this stream!",
        ),
        # Audio mention at t=25s (within 30s gap)
        AudioSegment(
            segment_id=2,
            start_sec=25.0,
            end_sec=30.0,
            speaker_label="STREAMER",
            transcript="Use code ASMON at checkout to get a discount on your custom PC.",
        ),
        # Unrelated audio at t=200s
        AudioSegment(
            segment_id=3,
            start_sec=200.0,
            end_sec=205.0,
            speaker_label="STREAMER",
            transcript="Now let's get back into the boss fight.",
        ),
    ]

    # Visual OCR mention at t=20s
    keyframes = [
        VisualKeyframe(
            frame_index=10,
            timestamp_sec=20.0,
            ocr_text_blocks=["STARFORGE SYSTEMS - VISIT STARFORGESYSTEMS.COM", "USE CODE ASMON"],
        )
    ]

    segments = detector.detect_segments(sample_brand, audio_segments=audio_segments, keyframes=keyframes)
    assert len(segments) == 1
    seg = segments[0]
    assert seg.brand_id == "starforge_systems"
    assert seg.start_sec == 10.0
    assert seg.end_sec == 30.0
    assert len(seg.matched_audio_transcripts) == 2
    assert len(seg.matched_ocr_texts) == 2


def test_sponsor_report_generator(sample_brand):
    detector = SponsorDetector(merge_threshold_sec=30.0)
    report_gen = SponsorReportGenerator(window_buffer_sec=30.0)

    audio = [
        AudioSegment(
            segment_id=1,
            start_sec=50.0,
            end_sec=60.0,
            speaker_label="STREAMER",
            transcript="Thanks to Starforge Systems for sponsoring!",
        )
    ]
    segments = detector.detect_segments(sample_brand, audio_segments=audio)
    assert len(segments) == 1
    segment = segments[0]

    # Create chat messages: baseline vs sponsor window
    messages = [
        # Baseline outside window
        ChatMessage(message_id="b1", timestamp_offset=5.0, user_id="u1", author_name="a1", content="good stream Pog"),
        ChatMessage(message_id="b2", timestamp_offset=10.0, user_id="u2", author_name="a2", content="hello everyone"),
        # Inside sponsor window [20.0, 90.0]
        ChatMessage(message_id="w1", timestamp_offset=52.0, user_id="u3", author_name="a3", content="Starforge Pog W computers!"),
        ChatMessage(message_id="w2", timestamp_offset=54.0, user_id="u4", author_name="a4", content="use code ASMON for real"),
        ChatMessage(message_id="w3", timestamp_offset=56.0, user_id="u5", author_name="a5", content="starforge pcs are so clean"),
        ChatMessage(message_id="w4", timestamp_offset=58.0, user_id="u6", author_name="a6", content="ad cringe sellout"),
    ]

    report = report_gen.generate_report(sample_brand, segment, messages)
    assert report.brand_id == "starforge_systems"
    assert report.chat_mention_count >= 3
    assert report.mention_velocity > 0.0
    assert report.backlash_index > 0.0
    assert 0.0 <= report.brand_attention_score <= 100.0
    assert "Starforge Systems" in report.summary
