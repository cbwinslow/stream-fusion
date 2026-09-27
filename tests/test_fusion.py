"""Unit tests for StreamFusion core pipeline."""

from stream_fusion.models.schemas import AudioSegment, VisualKeyframe, ChatMessage
from stream_fusion.chat.analyzer import ChatAnalyzer
from stream_fusion.fusion.matrix import FusionEngine


def test_chat_latency_calibration():
    analyzer = ChatAnalyzer(latency_offset_sec=4.0)
    messages = [
        ChatMessage(
            message_id="1",
            timestamp_offset=14.0,  # Posted at 14.0s -> Calibrated event should be 10.0s
            user_id="u1",
            author_name="User1",
            content="OMEGALUL TRUE",
        )
    ]
    buckets = analyzer.aggregate_into_buckets(messages, stream_duration_sec=20.0, bucket_size_sec=2.0)
    # 10.0s falls into bucket index 5 (10.0 to 12.0s)
    b5 = buckets[5]
    assert b5["chat_message_count"] == 1
    assert "OMEGALUL" in b5["dominant_emotes"]


def test_fusion_matrix_alignment():
    audio = [
        AudioSegment(
            segment_id=1,
            start_sec=10.0,
            end_sec=14.0,
            speaker_label="STREAMER",
            transcript="This game is crazy",
        )
    ]
    visual = [
        VisualKeyframe(
            frame_index=1,
            timestamp_sec=10.0,
            scene_type="GAMEPLAY",
            screen_summary="Streamer fighting boss",
        )
    ]
    analyzer = ChatAnalyzer(latency_offset_sec=0.0)
    messages = [
        ChatMessage(
            message_id="1",
            timestamp_offset=11.0,
            user_id="u1",
            author_name="User1",
            content="Pog",
        )
    ]
    buckets = analyzer.aggregate_into_buckets(messages, stream_duration_sec=20.0, bucket_size_sec=2.0)

    engine = FusionEngine(bucket_size_sec=2.0)
    result = engine.build_matrix(
        stream_id="test_01",
        duration_sec=20.0,
        audio_segments=audio,
        visual_keyframes=visual,
        chat_buckets=buckets,
    )

    assert len(result.slices) == 11
    # Check slice at 10.0s (index 5)
    s5 = result.slices[5]
    assert s5.streamer_transcript == "This game is crazy"
    assert s5.active_scene_type == "GAMEPLAY"
    assert s5.chat_message_count == 1


def test_highlight_audience_alignment_labels():
    from stream_fusion.models.schemas import FusionSlice

    engine = FusionEngine()
    slices = [
        FusionSlice(
            slice_index=1,
            bucket_index=1,
            start_sec=10.0,
            end_sec=12.0,
            is_spike_moment=True,
            chat_velocity_per_sec=8.0,
            agreement_score=-0.75,
            streamer_transcript="I loved this so much",
        ),
        FusionSlice(
            slice_index=2,
            bucket_index=2,
            start_sec=20.0,
            end_sec=22.0,
            is_spike_moment=True,
            chat_velocity_per_sec=5.0,
            agreement_score=0.10,
            streamer_transcript="Maybe okay",
        ),
    ]
    highlights = engine._extract_highlights(slices)
    assert len(highlights) == 2
    assert highlights[0]["audience_alignment"] == "AUDIENCE_REVOLT"
    assert highlights[1]["audience_alignment"] == "DIVIDED"

