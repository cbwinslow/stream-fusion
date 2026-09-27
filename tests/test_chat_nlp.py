"""Unit tests for Chat NLP, Meme Burst tracking, and Chatter Influence scoring."""

import pytest
from stream_fusion.chat.nlp import (
    ChatNLPAnalyzer,
    MemeBurstTracker,
    ChatterInfluenceScorer,
    normalize_repeated_chars,
    jaccard_similarity,
    tokenize_for_similarity,
)
from stream_fusion.models.schemas import AudioSegment, ChatMessage


def test_repeated_chars_normalization():
    assert normalize_repeated_chars("loooooool") == "lol"
    assert normalize_repeated_chars("pogggg") == "pog"
    assert normalize_repeated_chars("???") == "?"
    assert normalize_repeated_chars("KEKW") == "kekw"


def test_jaccard_similarity():
    s1 = tokenize_for_similarity("I can't believe he actually did that")
    s2 = tokenize_for_similarity("I can't believe he actually did this")
    assert jaccard_similarity(s1, s2) > 0.7

    s3 = tokenize_for_similarity("completely unrelated completely different")
    assert jaccard_similarity(s1, s3) == 0.0


def test_chat_intent_classification():
    analyzer = ChatNLPAnalyzer()

    # Amusement
    msg_amuse = ChatMessage(
        message_id="1", timestamp_offset=10.0, user_id="u1", author_name="a1",
        content="OMEGALUL that was hilarious LOOOOOOL",
    )
    dist = analyzer.classify_message(msg_amuse)
    assert dist.primary_intent == "AMUSEMENT"
    assert dist.valence > 0.5

    # Hype
    msg_hype = ChatMessage(
        message_id="2", timestamp_offset=12.0, user_id="u2", author_name="a2",
        content="POGGERS WHAT A CLUTCH LETSGOOO",
    )
    dist = analyzer.classify_message(msg_hype)
    assert dist.primary_intent == "HYPE"
    assert dist.valence > 0.5

    # Disbelief
    msg_disbelief = ChatMessage(
        message_id="3", timestamp_offset=14.0, user_id="u3", author_name="a3",
        content="HUH ain't no way that happened",
    )
    dist = analyzer.classify_message(msg_disbelief)
    assert dist.primary_intent == "DISBELIEF"

    # Disgust / Cringe
    msg_cringe = ChatMessage(
        message_id="4", timestamp_offset=16.0, user_id="u4", author_name="a4",
        content="cringe weirdchamp aware",
    )
    dist = analyzer.classify_message(msg_cringe)
    assert dist.primary_intent == "DISGUST_CRINGE"
    assert dist.valence < 0.0

    # Agreement
    msg_agree = ChatMessage(
        message_id="5", timestamp_offset=18.0, user_id="u5", author_name="a5",
        content="TRUE based facts Gigachad",
    )
    dist = analyzer.classify_message(msg_agree)
    assert dist.primary_intent == "AGREEMENT"

    # Heart warm
    msg_wholesome = ChatMessage(
        message_id="6", timestamp_offset=20.0, user_id="u6", author_name="a6",
        content="widepeepoHappy BibleThump <3 wholesome",
    )
    dist = analyzer.classify_message(msg_wholesome)
    assert dist.primary_intent == "HEART_WARM"

    # Question
    msg_q = ChatMessage(
        message_id="7", timestamp_offset=22.0, user_id="u7", author_name="a7",
        content="wait why did he do that?",
    )
    dist = analyzer.classify_message(msg_q)
    assert dist.primary_intent == "QUESTION"

    # Neutral
    msg_neutral = ChatMessage(
        message_id="8", timestamp_offset=24.0, user_id="u8", author_name="a8",
        content="hello stream good morning",
    )
    dist = analyzer.classify_message(msg_neutral)
    assert dist.primary_intent == "NEUTRAL"


def test_meme_burst_tracker():
    tracker = MemeBurstTracker(window_sec=10.0, similarity_threshold=0.6, min_distinct_authors=5)

    # Generate a copy-pasta burst from 6 different authors within 8 seconds
    copypasta = "HE DOESN'T KNOW ABOUT THE DRAGON CHAT Aware"
    messages = []
    for i in range(6):
        messages.append(
            ChatMessage(
                message_id=f"msg_{i}",
                timestamp_offset=100.0 + i * 1.2,
                user_id=f"user_{i}",
                author_name=f"chatter_{i}",
                content=copypasta if i % 2 == 0 else "HE DOESN'T KNOW ABOUT THE DRAGON Aware",
            )
        )

    # Add random chatter
    messages.append(
        ChatMessage(
            message_id="rand_1",
            timestamp_offset=102.0,
            user_id="rand_u",
            author_name="random_guy",
            content="just got back from lunch what happened",
        )
    )

    bursts = tracker.detect_meme_bursts(messages)
    assert len(bursts) == 1
    burst = bursts[0]
    assert burst.meme_id == "meme_0"
    assert burst.origin_author_name == "chatter_0"
    assert burst.unique_spreaders == 6
    assert burst.total_occurrences == 6
    assert burst.burst_start_sec == 100.0
    assert burst.burst_end_sec == 106.0
    assert burst.propagation_velocity > 0.0


def test_chatter_influence_scorer():
    scorer = ChatterInfluenceScorer(alpha=0.5, beta=0.4, gamma=0.1)

    # Chatter 1 started a meme burst
    tracker = MemeBurstTracker(window_sec=10.0, min_distinct_authors=3)
    messages = [
        ChatMessage(
            message_id=f"m_{i}",
            timestamp_offset=10.0 + i,
            user_id=f"u_{i}",
            author_name=f"leader" if i == 0 else f"follower_{i}",
            content="PUT ON THE WIZARD HAT ASMON",
        )
        for i in range(4)
    ]

    # Leader also has streamer response in audio 4 seconds later
    streamer_audio = [
        AudioSegment(
            segment_id=1,
            start_sec=14.5,
            end_sec=17.0,
            speaker_label="STREAMER",
            transcript="I am not going to put on the wizard hat chat, stop asking.",
        )
    ]

    bursts = tracker.detect_meme_bursts(messages)
    profiles = scorer.score_chatters(messages, bursts, streamer_audio=streamer_audio)

    assert len(profiles) > 0
    leader_profile = next(p for p in profiles if p.author_name == "leader")
    assert leader_profile.first_meme_origin_count == 1
    assert leader_profile.streamer_response_count == 1
    assert leader_profile.influence_score > 0.8
    # Leader is ranked first
    assert profiles[0].author_name == "leader"
