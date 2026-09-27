"""Tests for VisionProcessor and screen understanding."""

from pathlib import Path
import pytest
from stream_fusion.models.schemas import BoundingBox
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


def test_locate_streamer_facecam():
    # 1. No detected objects -> None
    assert VisionProcessor.locate_streamer_facecam([]) is None

    # 2. Detected objects with no person -> None
    non_person = [
        BoundingBox(label="monitor", confidence=0.9, box=[0.1, 0.1, 0.8, 0.8]),
        BoundingBox(label="keyboard", confidence=0.8, box=[0.7, 0.2, 0.9, 0.7]),
    ]
    assert VisionProcessor.locate_streamer_facecam(non_person) is None

    # 3. Streamer detected in bottom-right corner
    objects = [
        BoundingBox(label="screen", confidence=0.95, box=[0.0, 0.0, 1.0, 1.0]),
        BoundingBox(label="person", confidence=0.98, box=[0.60, 0.70, 0.98, 0.98]),
    ]
    facecam = VisionProcessor.locate_streamer_facecam(objects)
    assert facecam is not None
    assert facecam["x"] == pytest.approx(0.70, abs=0.01)
    assert facecam["y"] == pytest.approx(0.60, abs=0.01)
    assert facecam["w"] == pytest.approx(0.28, abs=0.01)
    assert facecam["h"] == pytest.approx(0.38, abs=0.01)


def test_missing_image_exception():
    processor = VisionProcessor(backend="fallback")
    with pytest.raises(FileNotFoundError):
        processor.process_frame(Path("non_existent_file.jpg"), 0.0, 0)


def test_vision_processor_ollama(monkeypatch):
    frame_path = Path(__file__).parent / "fixtures" / "sample_frame.jpg"
    processor = VisionProcessor(backend="ollama")

    # 1. Success mock
    class MockResponse:
        status_code = 200
        def json(self):
            return {"response": "Streamer is watching a game trailer with live commentary."}

    import requests
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: MockResponse())
    kf = processor.process_frame(frame_path, timestamp_sec=10.0, frame_index=1)
    assert kf.scene_type == "REACT_VIDEO"
    assert "game trailer" in kf.screen_summary

    # 2. Connection failure fallback
    def mock_fail(*args, **kwargs):
        raise ConnectionError("Ollama endpoint unreachable")

    monkeypatch.setattr(requests, "post", mock_fail)
    kf_fallback = processor.process_frame(frame_path, timestamp_sec=10.0, frame_index=1)
    assert "640x360" in kf_fallback.screen_summary


def test_vision_processor_florence_fallback():
    frame_path = Path(__file__).parent / "fixtures" / "sample_frame.jpg"
    # Florence on invalid device or missing weights should fall back cleanly without crashing
    processor = VisionProcessor(backend="florence", model_id="invalid/model/id")
    kf = processor.process_frame(frame_path, timestamp_sec=5.0, frame_index=1)
    assert kf.frame_index == 1
    assert "640x360" in kf.screen_summary


def test_locate_streamer_facecam_fullscreen_filter():
    # Candidate 1 is fullscreen person (> 0.85 w and h)
    # Candidate 2 is a corner streamer facecam
    objects = [
        BoundingBox(label="person", confidence=0.9, box=[0.0, 0.0, 0.95, 0.95]),
        BoundingBox(label="streamer", confidence=0.95, box=[0.7, 0.7, 0.95, 0.95]),
    ]
    facecam = VisionProcessor.locate_streamer_facecam(objects)
    assert facecam is not None
    assert facecam["x"] == pytest.approx(0.7, abs=0.01)
    assert facecam["y"] == pytest.approx(0.7, abs=0.01)

