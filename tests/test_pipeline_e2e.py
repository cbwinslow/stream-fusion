"""End-to-end integration test for the full StreamPipeline."""

from pathlib import Path
from stream_fusion.config import StreamFusionConfig
from stream_fusion.pipeline import StreamPipeline


def test_stream_pipeline_end_to_end(tmp_path: Path):
    fixture_video = Path(__file__).parent / "fixtures" / "sample_test_vod.mp4"
    fixture_chat = Path(__file__).parent / "fixtures" / "sample_twitch_chat.json"

    assert fixture_video.exists()
    assert fixture_chat.exists()

    # Use lightweight test configuration
    config = StreamFusionConfig()
    config.audio.whisper_model = "tiny"
    config.audio.device = "cpu"
    config.audio.compute_type = "int8"
    config.vision.device = "cpu"
    config.vision.sample_interval_sec = 2.0
    config.chat.bucket_window_sec = 2.0

    pipeline = StreamPipeline(config=config)
    result = pipeline.run(
        media_input=fixture_video,
        chat_input=fixture_chat,
        output_dir=tmp_path / "e2e_output",
        duration_sec=10.0,
    )

    # Assertions on pipeline result
    assert result.stream_id == "sample_test_vod"
    assert result.duration_sec == 10.0
    assert len(result.slices) == 6  # 0 to 10s in 2s buckets
    # Messages at 19s-20s fall outside the 10.0s window; only the first 2 messages fall within [0, 10s]
    assert result.total_chat_messages == 2

    # Check generated files
    report_file = tmp_path / "e2e_output" / "sample_test_vod_grounding_report.html"
    assert report_file.exists()
    assert report_file.stat().st_size > 1000  # valid HTML content

    wav_file = tmp_path / "e2e_output" / "sample_test_vod_audio_16k.wav"
    assert wav_file.exists()
    assert wav_file.stat().st_size > 10000
