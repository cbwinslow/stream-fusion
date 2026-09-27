"""Tests for dataset exporters (Parquet and JSONL training triples)."""

import json
from pathlib import Path
import pandas as pd
from stream_fusion.models.schemas import StreamAnalysisResult, FusionSlice
from stream_fusion.export.dataset import export_to_parquet, export_training_triples_jsonl


def test_export_to_parquet_and_jsonl(tmp_path: Path):
    slice_sample = FusionSlice(
        bucket_index=0,
        start_sec=10.0,
        end_sec=12.0,
        active_speakers=["STREAMER"],
        streamer_transcript="This game update is completely cooked.",
        external_audio_transcript=None,
        active_scene_type="REACT_VIDEO",
        visual_description="Streamer leaning forward looking at camera.",
        screen_ocr=["Patch 2.0 Notes"],
        chat_message_count=15,
        chat_velocity_per_sec=7.5,
        dominant_emotes={"OMEGALUL": 8, "COOKED": 5, "TRUE": 2},
        chat_sentiment_polarity=0.75,
        is_spike_moment=True,
        agreement_score=0.8,
    )

    result = StreamAnalysisResult(
        stream_id="test_stream_001",
        duration_sec=12.0,
        total_chat_messages=15,
        slices=[slice_sample],
        highlights=[],
    )

    # 1. Test Parquet Export
    parquet_path = tmp_path / "matrix.parquet"
    res_parquet = export_to_parquet(result, parquet_path)
    assert res_parquet.exists()
    df = pd.read_parquet(res_parquet)
    assert len(df) == 1
    assert df["stream_id"].iloc[0] == "test_stream_001"
    assert df["streamer_transcript"].iloc[0] == "This game update is completely cooked."
    assert df["chat_velocity_per_sec"].iloc[0] == 7.5

    # 2. Test JSONL Export
    jsonl_path = tmp_path / "training_data.jsonl"
    res_jsonl = export_training_triples_jsonl(result, jsonl_path)
    assert res_jsonl.exists()

    with open(res_jsonl, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    assert len(lines) == 1
    triple = lines[0]
    assert "instruction" in triple
    assert triple["input"]["streamer_commentary"] == "This game update is completely cooked."
    assert "OMEGALUL" in triple["output"]["predicted_dominant_emotes"]
    assert "AUDIENCE_ALIGNED" in triple["output"]["audience_alignment_verdict"]
