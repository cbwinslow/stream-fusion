"""Dataset exporters: Parquet and Hugging Face instruction-tuning JSONL."""

import json
from pathlib import Path
from typing import Dict, List
import pandas as pd

from stream_fusion.models.schemas import StreamAnalysisResult


def export_to_parquet(result: StreamAnalysisResult, output_path: Path) -> Path:
    """Exports the complete temporal matrix to an Apache Parquet file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for s in result.slices:
        records.append(
            {
                "stream_id": result.stream_id,
                "bucket_index": s.bucket_index,
                "start_sec": s.start_sec,
                "end_sec": s.end_sec,
                "active_speakers": ",".join(s.active_speakers),
                "streamer_transcript": s.streamer_transcript or "",
                "external_audio_transcript": s.external_audio_transcript or "",
                "scene_type": s.active_scene_type,
                "visual_description": s.visual_description,
                "ocr_text": " | ".join(s.screen_ocr),
                "chat_message_count": s.chat_message_count,
                "chat_velocity_per_sec": s.chat_velocity_per_sec,
                "dominant_emotes_json": json.dumps(s.dominant_emotes),
                "chat_sentiment_polarity": s.chat_sentiment_polarity,
                "is_spike_moment": s.is_spike_moment,
                "agreement_score": s.agreement_score if s.agreement_score is not None else 0.0,
            }
        )

    df = pd.DataFrame(records)
    df.to_parquet(output_path, index=False)
    return output_path


def export_training_triples_jsonl(result: StreamAnalysisResult, output_path: Path) -> Path:
    """Exports fine-tuning training triples: (Visual Context + Streamer Speech) -> Chat Reaction."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for s in result.slices:
        # Create training triples whenever there is speech or a notable chat reaction
        if s.streamer_transcript or s.chat_message_count >= 2:
            alignment = "NEUTRAL"
            if s.agreement_score is not None:
                if s.agreement_score >= 0.33:
                    alignment = "AUDIENCE_ALIGNED (BASED)"
                elif s.agreement_score <= -0.33:
                    alignment = "AUDIENCE_REVOLT (CALLING CAP)"
                else:
                    alignment = "DIVIDED"

            triple = {
                "instruction": (
                    "Given the visual scene context and the livestreamer's spoken commentary, "
                    "predict the expected real-time audience reaction, dominant emotes, and alignment verdict."
                ),
                "input": {
                    "timestamp_sec": s.start_sec,
                    "scene_type": s.active_scene_type,
                    "visual_context": s.visual_description,
                    "screen_ocr": s.screen_ocr,
                    "streamer_commentary": s.streamer_transcript,
                    "external_video_audio": s.external_audio_transcript,
                },
                "output": {
                    "predicted_chat_velocity_per_sec": round(s.chat_velocity_per_sec, 2),
                    "predicted_dominant_emotes": list(s.dominant_emotes.keys()),
                    "audience_sentiment_polarity": round(s.chat_sentiment_polarity, 2),
                    "audience_alignment_verdict": alignment,
                    "agreement_index": s.agreement_score,
                },
            }
            lines.append(json.dumps(triple))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return output_path
