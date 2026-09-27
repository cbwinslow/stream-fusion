"""Unit tests for Chatter Profiling, Banter Disambiguation, and Griefer Safety (Spec 12)."""

import pytest
from stream_fusion.chat.profiler import (
    BanterClassifier,
    BrigadeDetector,
    ChatterProfileStore,
)
from stream_fusion.models.schemas import ChatMessage


def test_bot_exclusion():
    classifier = BanterClassifier()

    bot_msg = ChatMessage(
        message_id="1", timestamp_offset=1.0, user_id="b1",
        author_name="Nightbot", content="Follow Asmongold on Twitter @Zackrawrr",
    )
    assert classifier.is_bot_message(bot_msg) is True

    cmd_msg = ChatMessage(
        message_id="2", timestamp_offset=2.0, user_id="u1",
        author_name="viewer_1", content="!uptime",
    )
    assert classifier.is_bot_message(cmd_msg) is True

    human_msg = ChatMessage(
        message_id="3", timestamp_offset=3.0, user_id="u2",
        author_name="real_viewer", content="hello asmon good luck today",
    )
    assert classifier.is_bot_message(human_msg) is False


def test_banter_vs_griefing_contextual_disambiguation():
    classifier = BanterClassifier()

    # Case 1: Streamer died in a video game -> "you're so trash washed lmao" is GOOD_NATURED_BANTER
    gaming_msg = ChatMessage(
        message_id="10", timestamp_offset=10.0, user_id="u10",
        author_name="playful_fan", content="YOU THREW SO HARD WASHED LMAO",
    )
    verdict1 = classifier.classify_message(
        gaming_msg,
        is_streamer_failure_moment=True,
        is_amusement_spike=True,
    )
    assert verdict1.verdict == "GOOD_NATURED_BANTER"

    # Case 2: Running community joke / roast on hair/baldness
    bald_msg = ChatMessage(
        message_id="11", timestamp_offset=20.0, user_id="u11",
        author_name="roaster", content="that hairline is crazy bald Aware",
    )
    verdict2 = classifier.classify_message(bald_msg, streamer_topic_is_serious=False)
    assert verdict2.verdict == "COMMUNITY_ROAST"

    # Case 3: Serious monologue about family/health -> "loser nobody cares" is BAD_FAITH_GRIEFING
    serious_msg = ChatMessage(
        message_id="12", timestamp_offset=30.0, user_id="u12",
        author_name="toxic_griefer", content="shut up loser nobody cares about your problems",
    )
    verdict3 = classifier.classify_message(serious_msg, streamer_topic_is_serious=True)
    assert verdict3.verdict == "BAD_FAITH_GRIEFING"


def test_chatter_profile_store():
    store = ChatterProfileStore(db_path=":memory:")

    # User 1: Regular fan making friendly jokes
    for i in range(5):
        store.ingest_message(
            ChatMessage(
                message_id=f"fan_{i}", timestamp_offset=10.0 + i, user_id="good_fan",
                author_name="GoodFan", content="lmao nice try asmon washed",
            ),
            is_streamer_failure_moment=True,
        )
    fan_profile = store.get_chatter_profile("good_fan")
    assert fan_profile is not None
    assert fan_profile.flagged_status == "CLEAN"
    assert fan_profile.banter_reciprocity > 0.8

    # User 2: Chronic hostile griefer
    for i in range(5):
        store.ingest_message(
            ChatMessage(
                message_id=f"troll_{i}", timestamp_offset=100.0 + i, user_id="bad_troll",
                author_name="BadTroll", content="unsubbed hate this loser fake nobody cares",
            ),
            streamer_topic_is_serious=True,
        )
    troll_profile = store.get_chatter_profile("bad_troll")
    assert troll_profile is not None
    assert troll_profile.flagged_status in ("WATCHLIST", "GRIEFER")
    assert troll_profile.griefer_score > 0.5


def test_brigade_detector():
    detector = BrigadeDetector(window_sec=60.0, similarity_threshold=0.65, min_accounts=5)

    raid_msgs = [
        ChatMessage(
            message_id=f"raid_{i}",
            timestamp_offset=200.0 + i * 2.0,
            user_id=f"raider_{i}",
            author_name=f"bot_account_{i}",
            content="CORRUPT STREAMER UNFOLLOW NOW CANCELLED",
        )
        for i in range(6)
    ]

    clusters = detector.detect_brigades(raid_msgs)
    assert len(clusters) == 1
    assert len(clusters[0].participant_user_ids) >= 5
    assert "CORRUPT STREAMER" in clusters[0].flagged_phrase
