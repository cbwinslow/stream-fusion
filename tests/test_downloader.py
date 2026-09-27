"""Tests for StreamDownloader wrapping yt-dlp."""

from pathlib import Path
from stream_fusion.ingest.downloader import StreamDownloader, find_ytdlp_binary


def test_find_ytdlp_binary():
    ytdlp_bin = find_ytdlp_binary()
    assert Path(ytdlp_bin).exists()
    assert "yt-dlp" in ytdlp_bin.lower()


def test_build_download_command():
    downloader = StreamDownloader()
    out_file = Path("output/test.mp4")
    cmd = downloader.build_download_command(
        url="https://www.twitch.tv/videos/123456",
        output_path=out_file,
        start_time_sec=15.0,
        duration_sec=45.0,
        download_chat=True,
    )

    assert downloader.ytdlp_bin in cmd
    assert "--download-sections" in cmd
    assert "*15.0-60.0" in cmd
    assert "https://www.twitch.tv/videos/123456" in cmd
    assert "--write-subs" in cmd


def test_dry_run_simulation(tmp_path: Path):
    downloader = StreamDownloader()
    out_file = tmp_path / "sim.mp4"
    # Testing simulation mode with public domain or dummy url flag
    cmd = downloader.build_download_command("https://example.com/video", out_file)
    assert "--simulate" not in cmd  # not inserted yet
