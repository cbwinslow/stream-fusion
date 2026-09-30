"""Video Archival Compression & Ephemeral Purging Engine (Spec 32).

Transcodes multi-hour high-resolution broadcasts into ultra-compact, web-scrubbable
reference proxies (~95% space reduction) and purges ephemeral intermediate assets.
"""

import logging
import os
from pathlib import Path
import shutil
import subprocess
from typing import List, Optional

from stream_fusion.models.schemas import ArchivalTranscodeResult

logger = logging.getLogger(__name__)


class VideoArchiver:
    """Manages archival downsampling, proxy generation, and ephemeral file deletion."""

    def __init__(self, ffmpeg_bin: str = "ffmpeg"):
        self.ffmpeg_bin = ffmpeg_bin

    def transcode_to_reference_proxy(
        self,
        source_video: Path,
        output_proxy: Optional[Path] = None,
        target_height: int = 480,
        target_fps: int = 24,
        video_bitrate_kbps: int = 350,
        audio_bitrate_kbps: int = 64,
        replace_source: bool = True,
        dry_run: bool = False,
    ) -> ArchivalTranscodeResult:
        """Transcodes source video to a compact 480p/360p reference proxy."""
        source_path = Path(source_video)
        if not source_path.exists() and not dry_run:
            raise FileNotFoundError(f"Source video not found: {source_path}")

        original_size = source_path.stat().st_size if source_path.exists() else 0
        proxy_path = Path(output_proxy or (source_path.parent / f"{source_path.stem}_proxy.mp4"))
        proxy_path.parent.mkdir(parents=True, exist_ok=True)

        if dry_run:
            mock_proxy_size = max(1024, int(original_size * 0.05)) if original_size > 0 else 1024 * 1024
            reduction_pct = 95.0
            return ArchivalTranscodeResult(
                source_path=str(source_path),
                proxy_path=str(proxy_path),
                original_size_bytes=original_size,
                proxy_size_bytes=mock_proxy_size,
                reduction_pct=reduction_pct,
                duration_sec=0.0,
                resolution=f"-2x{target_height}",
                success=True,
            )

        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-i", str(source_path),
            "-vf", f"scale=-2:{target_height},fps={target_fps}",
            "-c:v", "libx264",
            "-crf", "28",
            "-preset", "fast",
            "-b:v", f"{video_bitrate_kbps}k",
            "-maxrate", f"{int(video_bitrate_kbps * 1.4)}k",
            "-bufsize", f"{int(video_bitrate_kbps * 2.5)}k",
            "-c:a", "aac",
            "-b:a", f"{audio_bitrate_kbps}k",
            "-movflags", "+faststart",
            str(proxy_path),
        ]

        logger.info("Executing archival transcoding: %s", " ".join(cmd))
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as ex:
            logger.error("FFmpeg archival transcoding failed: %s", ex)
            return ArchivalTranscodeResult(
                source_path=str(source_path),
                proxy_path=str(proxy_path),
                original_size_bytes=original_size,
                proxy_size_bytes=0,
                reduction_pct=0.0,
                duration_sec=0.0,
                resolution=f"-2x{target_height}",
                success=False,
                error_message=str(ex),
            )

        proxy_size = proxy_path.stat().st_size if proxy_path.exists() else 0
        reduction_pct = 0.0
        if original_size > 0 and proxy_size > 0:
            reduction_pct = round((1.0 - (proxy_size / original_size)) * 100.0, 2)

        # Replace source if requested and proxy succeeded
        if replace_source and proxy_size > 0 and source_path != proxy_path:
            try:
                source_path.unlink()
                logger.info("Replaced original source %s with compressed proxy %s", source_path.name, proxy_path.name)
            except OSError as ex:
                logger.warning("Failed to unlink original source video: %s", ex)

        return ArchivalTranscodeResult(
            source_path=str(source_path),
            proxy_path=str(proxy_path),
            original_size_bytes=original_size,
            proxy_size_bytes=proxy_size,
            reduction_pct=reduction_pct,
            duration_sec=0.0,
            resolution=f"-2x{target_height}",
            success=True,
        )

    def purge_ephemeral_assets(self, target_paths: List[Path]) -> int:
        """Purges intermediate WAVs and temporary JPEG frames, returning bytes freed."""
        bytes_freed = 0
        for p in target_paths:
            p = Path(p)
            if not p.exists():
                continue
            if p.is_file():
                try:
                    size = p.stat().st_size
                    p.unlink()
                    bytes_freed += size
                except OSError as ex:
                    logger.warning("Failed to purge ephemeral file %s: %s", p, ex)
            elif p.is_dir():
                for f in p.glob("*"):
                    if f.is_file():
                        try:
                            size = f.stat().st_size
                            f.unlink()
                            bytes_freed += size
                        except OSError:
                            pass
                try:
                    shutil.rmtree(p, ignore_errors=True)
                except Exception:
                    pass

        logger.info("Purged ephemeral intermediate assets (freed %d bytes)", bytes_freed)
        return bytes_freed
