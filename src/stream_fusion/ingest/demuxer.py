"""Media demuxer: audio extraction and frame sampling using FFmpeg."""

import os
from pathlib import Path
import shutil
import subprocess
from typing import List, Optional


def find_ffmpeg_binary() -> str:
    """Finds ffmpeg binary in PATH or standard Windows WinGet locations."""
    found = shutil.which("ffmpeg")
    if found:
        return found

    # Fallback for Windows WinGet Gyan.FFmpeg installation
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    if local_app_data:
        winget_path = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
        if winget_path.exists():
            for p in winget_path.rglob("ffmpeg.exe"):
                if p.is_file():
                    return str(p)

    raise FileNotFoundError(
        "FFmpeg binary not found in PATH or standard installation directories. "
        "Please install FFmpeg or ensure it is accessible."
    )


def find_ffprobe_binary() -> str:
    """Finds ffprobe binary in PATH or standard Windows WinGet locations."""
    found = shutil.which("ffprobe")
    if found:
        return found

    local_app_data = os.environ.get("LOCALAPPDATA", "")
    if local_app_data:
        winget_path = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
        if winget_path.exists():
            for p in winget_path.rglob("ffprobe.exe"):
                if p.is_file():
                    return str(p)

    raise FileNotFoundError("FFprobe binary not found in PATH or standard installation directories.")


class MediaDemuxer:
    """Demuxes video containers into audio tracks and keyframe sequences."""

    def __init__(self, ffmpeg_bin: Optional[str] = None):
        self.ffmpeg_bin = ffmpeg_bin or find_ffmpeg_binary()

    def extract_audio_16k_mono(self, video_path: Path, output_wav_path: Path) -> Path:
        """Extracts 16kHz mono WAV suitable for Faster-Whisper and PyAnnote."""
        output_wav_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(output_wav_path),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg audio extraction failed (code {res.returncode}): {res.stderr}")

        if not output_wav_path.exists() or output_wav_path.stat().st_size == 0:
            raise RuntimeError(f"Audio extraction output is empty or missing: {output_wav_path}")

        return output_wav_path

    def extract_frames_at_interval(
        self, video_path: Path, output_dir: Path, interval_sec: float = 2.0
    ) -> List[Path]:
        """Extracts frames spaced by interval_sec seconds."""
        output_dir.mkdir(parents=True, exist_ok=True)
        fps = 1.0 / interval_sec
        pattern = output_dir / "frame_%04d.jpg"

        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-i", str(video_path),
            "-vf", f"fps={fps}",
            "-q:v", "2",
            str(pattern),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg frame extraction failed (code {res.returncode}): {res.stderr}")

        frames = sorted(output_dir.glob("frame_*.jpg"))
        return frames
