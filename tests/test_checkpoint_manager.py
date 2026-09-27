"""Unit tests for Stateful Checkpoint Resumption (Spec 10)."""

import pytest
from pathlib import Path
from stream_fusion.checkpoint.manager import CheckpointManager
from stream_fusion.models.schemas import (
    AudioSegment,
    VisualKeyframe,
    BoundingBox,
    FusionSlice,
    StreamAnalysisResult,
)


def test_checkpoint_manager_lifecycle(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    stream_id = "test_vod_001"

    manager = CheckpointManager(
        cache_dir=cache_dir,
        stream_id=stream_id,
        chunk_duration_sec=300.0,
        total_chunks_expected=3,
    )

    assert not manager.is_chunk_completed(0)
    assert manager.get_completed_chunks() == []

    # 1. Save and load audio chunk
    audio_data = [
        AudioSegment(
            segment_id=1,
            start_sec=0.0,
            end_sec=5.0,
            speaker_label="STREAMER",
            transcript="Testing checkpoint audio serialization",
        )
    ]
    manager.save_chunk_audio(0, audio_data)
    loaded_audio = manager.load_chunk_audio(0)
    assert loaded_audio is not None
    assert len(loaded_audio) == 1
    assert loaded_audio[0].transcript == "Testing checkpoint audio serialization"

    # 2. Save and load vision chunk
    vision_data = [
        VisualKeyframe(
            frame_index=1,
            timestamp_sec=2.0,
            scene_type="FULLSCREEN_CAM",
            screen_summary="Streamer talking directly to camera",
            ocr_text_blocks=["Chat rules"],
            detected_objects=[
                BoundingBox(label="facecam", confidence=0.95, box=[0.1, 0.1, 0.4, 0.4])
            ],
        )
    ]
    manager.save_chunk_vision(0, vision_data)
    loaded_vision = manager.load_chunk_vision(0)
    assert loaded_vision is not None
    assert len(loaded_vision) == 1
    assert loaded_vision[0].scene_type == "FULLSCREEN_CAM"

    # 3. Save and load chat chunk
    chat_buckets = [
        {
            "bucket_index": 0,
            "start_sec": 0.0,
            "end_sec": 2.0,
            "chat_message_count": 15,
            "chat_velocity_per_sec": 7.5,
            "dominant_emotes": {"LUL": 10},
            "chat_sentiment_polarity": 0.7,
        }
    ]
    manager.save_chunk_chat(0, chat_buckets)
    loaded_chat = manager.load_chunk_chat(0)
    assert loaded_chat is not None
    assert len(loaded_chat) == 1
    assert loaded_chat[0]["dominant_emotes"]["LUL"] == 10

    # 4. Save and load final result
    mock_result = StreamAnalysisResult(
        stream_id=stream_id,
        duration_sec=300.0,
        total_chat_messages=15,
        slices=[
            FusionSlice(
                bucket_index=0,
                start_sec=0.0,
                end_sec=2.0,
                chat_message_count=15,
            )
        ],
        highlights=[],
    )
    manager.save_final_result(mock_result)
    loaded_result = manager.load_final_result()
    assert loaded_result is not None
    assert loaded_result.stream_id == stream_id

    # 5. Mark chunk completed
    manager.mark_chunk_completed(0)
    assert manager.is_chunk_completed(0)
    assert manager.get_completed_chunks() == [0]

    # Verify reload of new manager instance from same cache_dir
    resumed_manager = CheckpointManager(
        cache_dir=cache_dir,
        stream_id=stream_id,
        chunk_duration_sec=300.0,
        total_chunks_expected=3,
    )
    assert resumed_manager.is_chunk_completed(0)
    assert not resumed_manager.is_chunk_completed(1)
