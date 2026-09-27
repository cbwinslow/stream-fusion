"""Tests for Dynamic Broadcast LatencyCalibrator."""

from stream_fusion.models.schemas import AudioSegment, ChatMessage
from stream_fusion.chat.calibrator import LatencyCalibrator


def test_compute_optimal_latency():
    calibrator = LatencyCalibrator(min_lag_sec=2.0, max_lag_sec=8.0, default_lag_sec=4.0)

    # Audio event happens at 10.0s - 14.0s
    audio = [
        AudioSegment(
            segment_id=1,
            start_sec=10.0,
            end_sec=14.0,
            speaker_label="STREAMER",
            transcript="Check this out right now!",
        )
    ]

    # True delay: 4.5 seconds. Chatters react at 14.5s - 18.5s
    true_delay = 4.5
    chat = []
    # Background chat (1 message every 2s)
    for s in range(0, 40, 2):
        chat.append(
            ChatMessage(
                message_id=f"bg_{s}",
                timestamp_offset=float(s),
                user_id=f"u_{s}",
                author_name="viewer",
                content="hello",
            )
        )
    # Burst of chat messages at exactly 10.0s + true_delay = 14.5s
    for i in range(25):
        chat.append(
            ChatMessage(
                message_id=f"burst_{i}",
                timestamp_offset=14.5 + (i * 0.1),
                user_id=f"burst_u_{i}",
                author_name="chatter",
                content="OMEGALUL",
            )
        )

    estimated_lag = calibrator.compute_optimal_latency(
        audio_segments=audio,
        chat_messages=chat,
        stream_duration_sec=40.0,
        resolution_sec=0.5,
    )

    # Should detect 4.5s within resolution (4.0s - 5.0s)
    assert abs(estimated_lag - true_delay) <= 0.5


def test_latency_fallback_on_sparse_data():
    calibrator = LatencyCalibrator(default_lag_sec=4.2)
    # Insufficient chat data -> should safely return default lag
    estimated = calibrator.compute_optimal_latency([], [], stream_duration_sec=10.0)
    assert estimated == 4.2
