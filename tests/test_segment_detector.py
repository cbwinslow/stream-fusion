"""Tests for StreamSegmentDetector."""

from stream_fusion.models.schemas import FusionSlice
from stream_fusion.segmentation.segment_detector import StreamSegmentDetector


def test_segment_detector_and_youtube_chapters():
    # 0s to 80s: Just Chatting
    slices_p1 = [
        FusionSlice(
            bucket_index=i,
            start_sec=i * 2.0,
            end_sec=(i + 1) * 2.0,
            active_scene_type="FULLSCREEN_CAM",
            visual_description="Streamer talking to chat",
            chat_velocity_per_sec=3.0,
        )
        for i in range(40)  # 0 to 80s
    ]

    # 80s to 180s: Reaction Video
    slices_p2 = [
        FusionSlice(
            bucket_index=i,
            start_sec=i * 2.0,
            end_sec=(i + 1) * 2.0,
            active_scene_type="REACT_VIDEO",
            visual_description="YouTube Video",
            screen_ocr=["Developer Update 2026"],
            chat_velocity_per_sec=5.0,
        )
        for i in range(40, 90)  # 80 to 180s
    ]

    all_slices = slices_p1 + slices_p2
    detector = StreamSegmentDetector(min_segment_duration_sec=60.0)
    segments = detector.detect_segments(all_slices)

    assert len(segments) == 2
    # Segment 1: Just Chatting
    assert segments[0].category == "FULLSCREEN_CAM"
    assert segments[0].start_sec == 0.0

    # Segment 2: React Video
    assert segments[1].category == "REACT_VIDEO"
    assert segments[1].start_sec == 80.0
    assert "Developer Update 2026" in segments[1].title

    # Test YouTube chapters formatting
    chapters = detector.generate_youtube_chapters(segments)
    assert "00:00" in chapters
    assert "01:20" in chapters
