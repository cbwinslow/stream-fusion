"""Unit and Integration Tests for Spec 18: Subprocess Worker Isolation & Bounded Buffering."""

from pathlib import Path
import shutil
import tempfile
import pytest
from typer.testing import CliRunner
from PIL import Image

from stream_fusion.models.schemas import AudioSegment, VisualKeyframe
from stream_fusion.schema.envelope import StreamEventType, StreamFusionEnvelope
from stream_fusion.workers.protocol import WorkerTaskInput, WorkerTaskType
from stream_fusion.workers.isolation import WorkerIsolationManager
from stream_fusion.workers.bounded_buffer import BoundedFrameBuffer
from stream_fusion.pipeline import StreamPipeline
from stream_fusion.config import StreamFusionConfig
from stream_fusion.cli import app


@pytest.fixture
def runner():
    return CliRunner()


def test_worker_task_input_schema():
    """Verify WorkerTaskInput serialization and default fields."""
    task = WorkerTaskInput(
        task_type=WorkerTaskType.VISION_OCR,
        media_path="sample.mp4",
        output_path="out.json",
        config={"backend": "fallback"},
        timeout_sec=120.0,
    )
    assert task.task_type == WorkerTaskType.VISION_OCR
    assert task.timeout_sec == 120.0

    json_str = task.model_dump_json()
    deserialized = WorkerTaskInput.model_validate_json(json_str)
    assert deserialized.task_type == WorkerTaskType.VISION_OCR


def test_isolated_worker_vision_execution(tmp_path: Path):
    """Verify that an isolated subprocess executes vision processing and produces an envelope."""
    # Create a synthetic image
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    img_path = img_dir / "frame_001.jpg"
    img = Image.new("RGB", (320, 240), color="blue")
    img.save(img_path)

    worker_mgr = WorkerIsolationManager(temp_dir=tmp_path / "worker_tmp")

    frame_items = [
        {"path": str(img_path), "timestamp_sec": 2.0, "frame_index": 1}
    ]
    cfg = {"backend": "fallback", "device": "cpu", "detect_objects": False}

    keyframes = worker_mgr.process_keyframes_isolated(
        frame_items=frame_items, config=cfg, timeout_sec=60.0
    )

    assert len(keyframes) == 1
    assert isinstance(keyframes[0], VisualKeyframe)
    assert keyframes[0].frame_index == 1
    assert keyframes[0].timestamp_sec == 2.0


def test_isolated_worker_timeout_handling(tmp_path: Path):
    """Verify that a worker exceeding timeout fails cleanly with an ERROR envelope."""
    worker_mgr = WorkerIsolationManager(temp_dir=tmp_path / "worker_tmp")

    # Pass an impossible timeout to trigger timeout handling
    task = WorkerTaskInput(
        task_type=WorkerTaskType.VISION_OCR,
        media_path="nonexistent.mp4",
        output_path="",
        config={"backend": "fallback"},
        timeout_sec=0.0001,  # immediate timeout
        extra_payload={"frame_items": []},
    )

    envelope = worker_mgr.run_task(task)
    assert envelope.event_type == StreamEventType.ERROR
    assert "timed out" in envelope.payload.get("error", "").lower()


def test_isolated_worker_error_propagation(tmp_path: Path):
    """Verify that child process exceptions are cleanly caught and returned in error envelope."""
    worker_mgr = WorkerIsolationManager(temp_dir=tmp_path / "worker_tmp")

    # Pass nonexistent image file to trigger exception inside child process
    frame_items = [
        {"path": str(tmp_path / "does_not_exist.jpg"), "timestamp_sec": 0.0, "frame_index": 1}
    ]
    task = WorkerTaskInput(
        task_type=WorkerTaskType.VISION_OCR,
        media_path=str(tmp_path / "does_not_exist.jpg"),
        output_path="",
        config={"backend": "fallback"},
        timeout_sec=30.0,
        extra_payload={"frame_items": frame_items},
    )

    envelope = worker_mgr.run_task(task)
    assert envelope.event_type == StreamEventType.ERROR
    assert "not found" in envelope.payload.get("error", "").lower() or "error" in envelope.payload


def test_bounded_frame_buffer_purging(tmp_path: Path):
    """Verify that BoundedFrameBuffer unlinks frame images after window completion."""
    buffer = BoundedFrameBuffer(base_temp_dir=tmp_path / "buffer_temp")

    # Mock demuxer that creates fake image files
    class MockDemuxer:
        def extract_frames_at_interval(self, video, output_dir, interval_sec, start_time_sec, duration_sec):
            output_dir.mkdir(parents=True, exist_ok=True)
            created = []
            for i in range(3):
                p = output_dir / f"frame_{int(start_time_sec)}_{i}.jpg"
                p.write_bytes(b"dummy image data")
                created.append(p)
            return created

    buffer.demuxer = MockDemuxer()

    fake_video = tmp_path / "test_video.mp4"
    fake_video.write_bytes(b"dummy video data")

    observed_windows = []

    def mock_processor(frame_items, w_start, w_end):
        # Verify images exist during the processor call
        for item in frame_items:
            assert Path(item["path"]).exists()
        observed_windows.append((w_start, w_end, len(frame_items)))
        return [
            VisualKeyframe(
                frame_index=item["frame_index"],
                timestamp_sec=item["timestamp_sec"],
                screen_summary="Test Summary",
            )
            for item in frame_items
        ]

    keyframes = buffer.process_stream_windowed(
        video_path=fake_video,
        total_duration_sec=30.0,
        sample_interval_sec=2.0,
        window_size_sec=10.0,
        frame_processor=mock_processor,
        purge_on_complete=True,
    )

    # 3 windows of 10s: 0-10, 10-20, 20-30 -> 3 frames per window = 9 total keyframes
    assert len(keyframes) == 9
    assert len(observed_windows) == 3

    # Verify that all temporary window folders and frame images have been completely unlinked
    remaining_files = list((tmp_path / "buffer_temp").glob("**/*.jpg"))
    assert len(remaining_files) == 0, f"Expected 0 lingering frames, found: {remaining_files}"


def test_pipeline_with_worker_isolation_and_bounded_buffer(tmp_path: Path):
    """Test full pipeline run with isolate_gpu_workers=True and bounded_buffering=True."""
    video_path = Path("asmon_sample_60s.mp4")
    chat_path = Path("sample_asmon_chat.json")

    if not video_path.exists():
        pytest.skip("Sample media asmon_sample_60s.mp4 not found in project root")

    config = StreamFusionConfig()
    config.audio.whisper_model = "tiny"
    config.audio.device = "cpu"
    config.audio.compute_type = "int8"
    config.audio.diarization_enabled = False
    config.vision.device = "cpu"
    config.execution.isolate_gpu_workers = True
    config.execution.bounded_buffering = True
    config.execution.window_size_sec = 20.0

    pipeline = StreamPipeline(config=config)
    result = pipeline.run(
        media_input=video_path,
        chat_input=chat_path,
        output_dir=tmp_path / "pipeline_iso_out",
        duration_sec=10.0,  # Fast 10s slice for CI
    )

    assert result is not None
    assert len(result.slices) > 0
    assert result.duration_sec == 10.0


def test_cli_flags(runner):
    """Verify that CLI process command exposes --isolate-workers and --bounded-buffer flags."""
    res = runner.invoke(app, ["process", "--help"])
    assert res.exit_code == 0
    assert "--isolate-workers" in res.stdout
    assert "--bounded-buffer" in res.stdout
    assert "--window-size" in res.stdout
