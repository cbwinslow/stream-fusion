"""Tests for VerticalHighlightClipper."""

from pathlib import Path
from stream_fusion.export.clipper import VerticalHighlightClipper


def test_build_crop_filter():
    clipper = VerticalHighlightClipper()
    filter_str = clipper.build_crop_filter()
    assert "vstack" in filter_str
    assert "scale=1080:768" in filter_str


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

    # Clip 2 seconds (from 1.0s to 3.0s)
    result_path = clipper.export_highlight_short(
        video_path=fixture_video,
        start_sec=1.0,
        end_sec=3.0,
        output_path=output_short,
        dry_run=False,
    )

    assert result_path.exists()
    assert result_path.stat().st_size > 1000  # Valid MP4 file produced
