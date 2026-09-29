"""Unit and integration tests for Adaptive Frame Density Optimizer & Production Profiler (Spec 30)."""

from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

from stream_fusion.config import VisionConfig
from stream_fusion.ingest.demuxer import MediaDemuxer
from stream_fusion.models.schemas import AudioSegment, ChatMessage, ProductionBenchmarkReport
from stream_fusion.monitoring.profiler import ProductionRunProfiler
from stream_fusion.vision.optimizer import AdaptiveFrameOptimizer
from stream_fusion.workers.bounded_buffer import BoundedFrameBuffer


def test_optimizer_fixed_mode_fallback():
    cfg = VisionConfig(sampling_mode="fixed", sample_interval_sec=2.0)
    optimizer = AdaptiveFrameOptimizer(config=cfg)

    # 10 second stream with 2.0s interval -> [0.0, 2.0, 4.0, 6.0, 8.0]
    timestamps = optimizer.compute_optimal_timestamps(duration_sec=10.0)
    assert len(timestamps) == 5
    assert timestamps == [0.0, 2.0, 4.0, 6.0, 8.0]


def test_optimizer_chat_burst_densification():
    cfg = VisionConfig(
        sampling_mode="adaptive",
        sample_interval_sec=2.0,
        min_interval_sec=0.5,
        max_interval_sec=4.0,
        burst_window_sec=6.0,
        chat_burst_zscore_threshold=2.0,
    )
    optimizer = AdaptiveFrameOptimizer(config=cfg)

    # Create synthetic chat: baseline 1 msg every 2s, but huge flood at t=20s (15 messages)
    chat_msgs = []
    for t in range(0, 40, 2):
        chat_msgs.append(ChatMessage(message_id=f"m_{t}", user_id=f"u_{t}", author_name=f"user_{t}", content="hey", timestamp_offset=float(t)))

    # Burst at 20.0s
    for i in range(15):
        chat_msgs.append(ChatMessage(message_id=f"burst_{i}", user_id=f"spammer_{i}", author_name=f"spammer_{i}", content="CLIP THAT LMAO POG", timestamp_offset=20.0 + (i * 0.1)))

    burst_intervals = optimizer.identify_chat_bursts(chat_msgs, duration_sec=40.0)
    assert len(burst_intervals) >= 1
    # Check that burst window surrounds t=20s
    b_start, b_end = burst_intervals[0]
    assert b_start <= 20.0 <= b_end

    # Calculate optimal timestamps across 40s duration
    timestamps = optimizer.compute_optimal_timestamps(
        duration_sec=40.0,
        chat_messages=chat_msgs,
    )

    # In idle zones (e.g. 0 to 16s), step should be max_interval_sec (4.0s)
    idle_points = [t for t in timestamps if t < 16.0]
    assert len(idle_points) <= 5

    # In burst zone (18s to 26s), step should be min_interval_sec (0.5s)
    burst_points = [t for t in timestamps if 18.0 <= t <= 26.0]
    assert len(burst_points) >= 12  # Densified!


def test_optimizer_audio_burst_densification():
    cfg = VisionConfig(
        sampling_mode="adaptive",
        min_interval_sec=0.5,
        max_interval_sec=5.0,
    )
    optimizer = AdaptiveFrameOptimizer(config=cfg)

    # Audio segment with excitement exclamation
    audio_segs = [
        AudioSegment(segment_id=1, start_sec=5.0, end_sec=8.0, speaker_label="STREAMER", transcript="Just walking in the lobby."),
        AudioSegment(segment_id=2, start_sec=25.0, end_sec=28.0, speaker_label="STREAMER", transcript="HOLY WTF HE IS ONE SHOT LET'S GO!"),
    ]

    audio_bursts = optimizer.identify_audio_bursts(audio_segs, duration_sec=40.0)
    assert len(audio_bursts) >= 1
    b_start, b_end = audio_bursts[0]
    assert b_start <= 25.0 and b_end >= 28.0

    timestamps = optimizer.compute_optimal_timestamps(
        duration_sec=40.0,
        audio_segments=audio_segs,
    )

    # Burst zone around 25s has dense samples
    excited_points = [t for t in timestamps if 24.0 <= t <= 30.0]
    assert len(excited_points) >= 8


def test_optimizer_scene_cuts_inclusion():
    cfg = VisionConfig(sampling_mode="adaptive", max_interval_sec=5.0, min_interval_sec=1.0)
    optimizer = AdaptiveFrameOptimizer(config=cfg)

    scene_cuts = [12.4, 28.7]
    timestamps = optimizer.compute_optimal_timestamps(
        duration_sec=40.0,
        scene_cuts=scene_cuts,
    )

    # Scene cut timestamps (+0.1s) must be included
    assert any(abs(t - 12.5) < 0.1 for t in timestamps)
    assert any(abs(t - 28.8) < 0.1 for t in timestamps)


def test_demuxer_extract_frames_at_timestamps(tmp_path: Path):
    fixture_video = Path(__file__).parent / "fixtures" / "sample_test_vod.mp4"
    assert fixture_video.exists()

    demuxer = MediaDemuxer()
    out_dir = tmp_path / "custom_frames"

    # Extract non-uniform frames at specific timestamps
    target_ts = [0.5, 2.2, 5.0]
    results = demuxer.extract_frames_at_timestamps(fixture_video, out_dir, target_ts)

    assert len(results) == 3
    for ts, f_path in results:
        assert ts in target_ts
        assert f_path.exists()
        assert f_path.stat().st_size > 0


def test_bounded_frame_buffer_with_adaptive_timestamps(tmp_path: Path):
    fixture_video = Path(__file__).parent / "fixtures" / "sample_test_vod.mp4"
    assert fixture_video.exists()

    demuxer = MediaDemuxer()
    buffer = BoundedFrameBuffer(demuxer=demuxer, base_temp_dir=tmp_path / "buf")

    custom_ts = [0.0, 1.0, 1.5, 4.0, 7.5]
    processed_items = []

    def mock_processor(items, start, end):
        processed_items.extend(items)
        return []

    buffer.process_stream_windowed(
        fixture_video,
        total_duration_sec=10.0,
        window_size_sec=5.0,
        optimal_timestamps=custom_ts,
        frame_processor=mock_processor,
    )

    assert len(processed_items) == len(custom_ts)
    extracted_timestamps = [item["timestamp_sec"] for item in processed_items]
    assert extracted_timestamps == custom_ts


def test_production_run_profiler(tmp_path: Path):
    profiler = ProductionRunProfiler(stream_id="test_stream_001", duration_sec=3600.0)

    # 1 hour stream:
    # 60 FPS total potential = 216,000 frames
    # Fixed 2.0s = 1,800 frames
    # Suppose adaptive mode extracted 1,000 frames
    analyzed_ts = [float(i * 3.6) for i in range(1000)]
    bursts = [(500.0, 520.0), (1200.0, 1230.0)]

    profiler.record_sampling_run(
        analyzed_timestamps=analyzed_ts,
        sampling_mode="adaptive",
        sample_interval_sec=2.0,
        burst_intervals=bursts,
    )

    report = profiler.generate_report()
    assert isinstance(report, ProductionBenchmarkReport)
    assert report.stream_id == "test_stream_001"
    assert report.total_potential_frames == 216000
    assert report.fixed_sampling_frames == 1801
    assert report.actual_analyzed_frames == 1000
    assert report.frames_saved == 801
    assert report.reduction_pct > 40.0
    assert report.estimated_gpu_time_saved_sec > 60.0
    assert report.burst_zone_coverage_pct == 100.0

    # JSON export test
    json_path = tmp_path / "benchmark.json"
    profiler.export_json(json_path)
    assert json_path.exists()
    assert json_path.stat().st_size > 100

    # Markdown summary
    md = profiler.format_markdown_summary()
    assert "Production Run Benchmark" in md
    assert "Frames Saved" in md
