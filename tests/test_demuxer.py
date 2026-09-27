"""Tests for MediaDemuxer using real test video fixture."""

from pathlib import Path
import pytest
from stream_fusion.ingest.demuxer import MediaDemuxer, find_ffmpeg_binary


def test_find_ffmpeg_binary():
    ffmpeg_bin = find_ffmpeg_binary()
    assert Path(ffmpeg_bin).exists()
    assert "ffmpeg" in ffmpeg_bin.lower()


def test_extract_audio_16k_mono(tmp_path: Path):
    fixture_video = Path(__file__).parent / "fixtures" / "sample_test_vod.mp4"
    assert fixture_video.exists()

    demuxer = MediaDemuxer()
    output_wav = tmp_path / "test_extracted.wav"

    res_path = demuxer.extract_audio_16k_mono(fixture_video, output_wav)
    assert res_path.exists()
    assert res_path.stat().st_size > 10000  # 10s of 16kHz mono 16-bit PCM is ~320,000 bytes


def test_extract_frames_at_interval(tmp_path: Path):
    fixture_video = Path(__file__).parent / "fixtures" / "sample_test_vod.mp4"
    assert fixture_video.exists()

    demuxer = MediaDemuxer()
    frames_dir = tmp_path / "frames"

    # 10-second video at 2.0s interval -> should yield ~5 frames
    frames = demuxer.extract_frames_at_interval(fixture_video, frames_dir, interval_sec=2.0)
    assert len(frames) >= 4
    for f in frames:
        assert f.exists()
        assert f.stat().st_size > 0
