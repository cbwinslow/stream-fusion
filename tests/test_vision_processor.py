"""Tests for VisionProcessor and screen understanding."""

from pathlib import Path
import pytest
from stream_fusion.vision.processor import VisionProcessor


def test_vision_processor_fallback():
    frame_path = Path(__file__).parent / "fixtures" / "sample_frame.jpg"
    assert frame_path.exists(), "Sample frame fixture missing"

    processor = VisionProcessor(backend="fallback")
    keyframe = processor.process_frame(frame_path, timestamp_sec=14.5, frame_index=1)

    assert keyframe.frame_index == 1
    assert keyframe.timestamp_sec == 14.5
    assert keyframe.scene_type in ["REACT_VIDEO", "BROWSER"]
    assert "640x360" in keyframe.screen_summary

    # Test unload
    processor.unload()
    assert processor._model is None


def test_missing_image_exception():
    processor = VisionProcessor(backend="fallback")
    with pytest.raises(FileNotFoundError):
        processor.process_frame(Path("non_existent_file.jpg"), 0.0, 0)
