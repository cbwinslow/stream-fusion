"""Tests for VerticalHighlightClipper."""

from pathlib import Path
from stream_fusion.export.clipper import VerticalHighlightClipper
from stream_fusion.models.schemas import WordTiming


def test_build_crop_filter():
    clipper = VerticalHighlightClipper()
    filter_str = clipper.build_crop_filter()
    assert "vstack" in filter_str
    assert "scale=1080:768" in filter_str


def test_build_crop_filter_with_custom_boxes_and_subtitles():
    clipper = VerticalHighlightClipper()
    filter_str = clipper.build_crop_filter(
        facecam_box={"x": 0.60, "y": 0.50, "w": 0.40, "h": 0.50},
        content_box={"x": 0.0, "y": 0.0, "w": 0.80, "h": 0.90},
        ass_subtitle_path=Path("C:/test/sub.ass"),
    )
    assert "in_w*0.400:in_h*0.500:in_w*0.600:in_h*0.500" in filter_str
    assert "in_w*0.800:in_h*0.900:in_w*0.000:in_h*0.000" in filter_str
    assert "subtitles=" in filter_str


def test_build_clip_command():
    clipper = VerticalHighlightClipper()
    cmd = clipper.build_clip_command(
        video_path=Path("sample.mp4"),
        start_sec=10.0,
        end_sec=25.0,
        output_path=Path("out_short.mp4"),
    )
    assert "-filter_complex" in cmd
    assert "-ss" in cmd
    assert "10.0" in cmd
    assert "25.0" in cmd


def test_export_highlight_short_real_execution(tmp_path: Path):
    fixture_video = Path(__file__).parent / "fixtures" / "sample_test_vod.mp4"
    assert fixture_video.exists()

    clipper = VerticalHighlightClipper()
    output_short = tmp_path / "asmon_test_short.mp4"

    # Clip 2 seconds (from 1.0s to 3.0s) with simulated word timings
    words = [
        WordTiming(word="Hold", start=1.0, end=1.5),
        WordTiming(word="on", start=1.5, end=2.0),
        WordTiming(word="bro!", start=2.0, end=2.8),
    ]

    result_path = clipper.export_highlight_short(
        video_path=fixture_video,
        start_sec=1.0,
        end_sec=3.0,
        output_path=output_short,
        words=words,
        facecam_box={"x": 0.65, "y": 0.55, "w": 0.35, "h": 0.45},
        burn_subtitles=True,
        dry_run=False,
    )

    assert result_path.exists()
    assert result_path.stat().st_size > 1000  # Valid MP4 file produced
