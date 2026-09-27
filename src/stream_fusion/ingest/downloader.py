"""Stream VOD downloader wrapping yt-dlp with time-slice support."""

import os
from pathlib import Path
import shutil
import subprocess
from typing import List, Optional


def find_ytdlp_binary() -> str:
    """Finds yt-dlp binary in PATH or user Python Scripts directories."""
    found = shutil.which("yt-dlp")
    if found:
        return found

    # Check Windows user roaming scripts
    app_data = os.environ.get("APPDATA", "")
    if app_data:
        py_scripts = Path(app_data) / "Python"
        for candidate in py_scripts.rglob("yt-dlp.exe"):
            if candidate.is_file():
                return str(candidate)

    raise FileNotFoundError("yt-dlp binary not found in PATH or Python Scripts directory.")


class StreamDownloader:
    """Manages downloading VOD media and chat replay using yt-dlp."""

    def __init__(self, ytdlp_bin: Optional[str] = None):
        self.ytdlp_bin = ytdlp_bin or find_ytdlp_binary()

    def build_download_command(
        self,
        url: str,
        output_path: Path,
        start_time_sec: Optional[float] = None,
        duration_sec: Optional[float] = None,
        download_chat: bool = True,
    ) -> List[str]:
        """Builds the argument list for yt-dlp."""
        cmd = [
            self.ytdlp_bin,
            "--no-playlist",
            "-o", str(output_path),
        ]

        # Time range section slicing
        if start_time_sec is not None or duration_sec is not None:
            s_start = start_time_sec or 0.0
            s_end = s_start + (duration_sec or 60.0)
            cmd.extend(["--download-sections", f"*{s_start}-{s_end}"])

        # Chat subtitles if available on Twitch/YouTube
        if download_chat:
            cmd.extend(["--write-subs", "--sub-langs", "all,-live_chat"])

        cmd.append(url)
        return cmd

    def download_segment(
        self,
        url: str,
        output_path: Path,
        start_time_sec: Optional[float] = None,
        duration_sec: Optional[float] = None,
        dry_run: bool = False,
    ) -> Path:
        """Executes download for a targeted stream segment."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = self.build_download_command(
            url=url,
            output_path=output_path,
            start_time_sec=start_time_sec,
            duration_sec=duration_sec,
        )

        if dry_run:
            cmd.insert(1, "--simulate")

        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"yt-dlp failed (code {res.returncode}): {res.stderr}")

        return output_path
