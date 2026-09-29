"""Homelab Harvester Engine & Worker Pool (Spec 25).

Manages multi-threaded VOD downloads, per-streamer rate limiting,
disk space guards, and structured homelab storage staging with SHA-256 checksums.
"""

from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from stream_fusion.harvester.catalog import HarvestCatalog
from stream_fusion.models.schemas import (
    ChatMessage,
    HarvestedVodRecord,
    HarvesterStatusReport,
    HarvestStatus,
    StreamerTargetRecord,
)

logger = logging.getLogger(__name__)


def compute_file_sha256(path: Path) -> str:
    """Computes SHA-256 hex digest for a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class HomelabHarvester:
    """Multi-threaded harvester engine with disk safety guards and per-streamer rate limits."""

    def __init__(
        self,
        catalog: HarvestCatalog,
        homelab_root: Path = Path("./homelab_storage"),
        max_concurrent_workers: int = 3,
        max_per_streamer: int = 1,
        min_free_disk_gb: float = 50.0,
        ytdlp_bin: Optional[str] = None,
        simulate: bool = False,
        custom_downloader: Optional[Callable[[HarvestedVodRecord, Path, StreamerTargetRecord], None]] = None,
    ):
        self.catalog = catalog
        self.homelab_root = Path(homelab_root).resolve()
        self.max_concurrent_workers = max_concurrent_workers
        self.max_per_streamer = max_per_streamer
        self.min_free_disk_gb = min_free_disk_gb
        self.ytdlp_bin = ytdlp_bin or shutil.which("yt-dlp")
        self.simulate = simulate
        self.custom_downloader = custom_downloader

        self._lock = threading.Lock()
        self._active_streamers: Dict[str, int] = {}  # streamer_id -> active count
        self._active_tasks: Dict[str, Dict[str, Any]] = {}  # vod_id -> task info
        self._running = False
        self._executor: Optional[ThreadPoolExecutor] = None

        self.homelab_root.mkdir(parents=True, exist_ok=True)

    def check_free_disk_gb(self) -> float:
        """Returns free disk space in GB on the homelab volume."""
        try:
            usage = shutil.disk_usage(self.homelab_root)
            return usage.free / (1024.0 ** 3)
        except Exception:
            return 999.0

    def get_status(self) -> HarvesterStatusReport:
        """Returns comprehensive real-time status of the harvester."""
        counts = self.catalog.count_vods_by_status()
        free_gb = round(self.check_free_disk_gb(), 2)

        with self._lock:
            active_info = list(self._active_tasks.values())
            active_workers = len(self._active_tasks)

        return HarvesterStatusReport(
            active_workers=active_workers,
            max_workers=self.max_concurrent_workers,
            queue_depth=counts.get(HarvestStatus.QUEUED.value, 0),
            downloading_count=counts.get(HarvestStatus.DOWNLOADING.value, 0),
            harvested_count=counts.get(HarvestStatus.HARVESTED.value, 0)
            + counts.get(HarvestStatus.READY_FOR_ANALYSIS.value, 0)
            + counts.get(HarvestStatus.ANALYZED.value, 0),
            error_count=counts.get(HarvestStatus.ERROR.value, 0),
            free_disk_gb=free_gb,
            active_downloads=active_info,
        )

    def harvest_vod(self, vod_id: str) -> HarvestedVodRecord:
        """Synchronously harvests a single VOD into homelab storage."""
        vod = self.catalog.get_vod(vod_id)
        if not vod:
            raise ValueError(f"VOD '{vod_id}' not found in catalog.")

        target = self.catalog.get_target(vod.streamer_id)
        if not target:
            # Fallback target record if not found
            target = StreamerTargetRecord(
                streamer_id=vod.streamer_id,
                display_name=vod.streamer_id,
                primary_platform=vod.platform,
            )

        # 1. Disk safety check
        free_gb = self.check_free_disk_gb()
        if free_gb < self.min_free_disk_gb and not self.simulate:
            err_msg = f"Insufficient disk space: {free_gb:.1f} GB free < {self.min_free_disk_gb:.1f} GB required threshold."
            logger.error(err_msg)
            self.catalog.update_vod_status(vod_id, HarvestStatus.ERROR, error_message=err_msg)
            raise RuntimeError(err_msg)

        # 2. Prepare destination path
        vod_dir = self._resolve_vod_directory(vod, target)
        vod_dir.mkdir(parents=True, exist_ok=True)

        media_path = vod_dir / "media.mp4"
        chat_path = vod_dir / "chat.json"
        metadata_path = vod_dir / "metadata.json"
        thumbnail_path = vod_dir / "thumbnail.jpg"
        checksum_path = vod_dir / "checksums.sha256"
        manifest_path = vod_dir / "ingest_manifest.json"

        # 3. Update status to DOWNLOADING
        start_time = time.time()
        self.catalog.update_vod_status(vod_id, HarvestStatus.DOWNLOADING, error_message=None)

        with self._lock:
            self._active_tasks[vod_id] = {
                "vod_id": vod_id,
                "streamer_id": vod.streamer_id,
                "title": vod.title,
                "started_at": datetime.now(timezone.utc).isoformat(),
            }

        try:
            # 4. Perform download (custom, simulated, or yt-dlp)
            if self.custom_downloader:
                self.custom_downloader(vod, vod_dir, target)
            elif self.simulate:
                self._simulate_harvest(vod, target, vod_dir)
            else:
                self._execute_ytdlp_harvest(vod, target, vod_dir)

            duration_sec = max(0.1, time.time() - start_time)
            file_size = media_path.stat().st_size if media_path.exists() else 0
            download_speed = round((file_size * 8 / (1024 * 1024)) / duration_sec, 2) if duration_sec > 0 else 0.0

            # 5. Compute SHA-256 checksums
            checksum_lines: List[str] = []
            file_hashes: Dict[str, str] = {}
            for target_file in [media_path, chat_path, metadata_path]:
                if target_file.exists():
                    f_hash = compute_file_sha256(target_file)
                    file_hashes[target_file.name] = f_hash
                    checksum_lines.append(f"{f_hash}  {target_file.name}\n")

            with open(checksum_path, "w", encoding="utf-8") as f:
                f.writelines(checksum_lines)

            # 6. Write StreamFusion ingest_manifest.json
            manifest_data = {
                "vod_id": vod_id,
                "streamer_id": vod.streamer_id,
                "platform": vod.platform,
                "title": vod.title,
                "quality_preset": target.quality_preset,
                "harvested_at": datetime.now(timezone.utc).isoformat(),
                "media_path": str(media_path.resolve()),
                "chat_path": str(chat_path.resolve()) if chat_path.exists() else None,
                "metadata_path": str(metadata_path.resolve()) if metadata_path.exists() else None,
                "thumbnail_path": str(thumbnail_path.resolve()) if thumbnail_path.exists() else None,
                "file_size_bytes": file_size,
                "download_speed_mbps": download_speed,
                "checksums": file_hashes,
            }
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=2)

            # 7. Update catalog to HARVESTED / READY_FOR_ANALYSIS
            updated_vod = self.catalog.update_vod_status(
                vod_id,
                HarvestStatus.HARVESTED,
                video_path=str(media_path.resolve()),
                chat_path=str(chat_path.resolve()) if chat_path.exists() else None,
                metadata_path=str(metadata_path.resolve()) if metadata_path.exists() else None,
                thumbnail_path=str(thumbnail_path.resolve()) if thumbnail_path.exists() else None,
                file_size_bytes=file_size,
                download_speed_mbps=download_speed,
                harvested_at=datetime.now(timezone.utc),
            )
            return updated_vod or vod

        except Exception as e:
            logger.error(f"Download failed for {vod_id}: {e}", exc_info=True)
            self.catalog.update_vod_status(
                vod_id,
                HarvestStatus.ERROR,
                error_message=str(e),
                retry_count=vod.retry_count + 1,
            )
            raise

        finally:
            with self._lock:
                self._active_tasks.pop(vod_id, None)

    def process_queue(self, limit: Optional[int] = None) -> List[HarvestedVodRecord]:
        """Processes queued VODs respecting max_concurrent_workers and max_per_streamer."""
        queued_vods = self.catalog.get_queued_vods(limit=limit)
        if not queued_vods:
            return []

        # Filter candidates based on per-streamer concurrency limits
        eligible: List[HarvestedVodRecord] = []
        active_counts: Dict[str, int] = {}

        with self._lock:
            for s_id, cnt in self._active_streamers.items():
                active_counts[s_id] = cnt

        for v in queued_vods:
            curr = active_counts.get(v.streamer_id, 0)
            if curr < self.max_per_streamer:
                eligible.append(v)
                active_counts[v.streamer_id] = curr + 1

            if len(eligible) >= self.max_concurrent_workers:
                break

        if not eligible:
            return []

        # Execute eligible downloads concurrently
        results: List[HarvestedVodRecord] = []
        with ThreadPoolExecutor(max_workers=self.max_concurrent_workers) as pool:
            futures: Dict[Future, str] = {}
            for v in eligible:
                with self._lock:
                    self._active_streamers[v.streamer_id] = self._active_streamers.get(v.streamer_id, 0) + 1

                fut = pool.submit(self._run_with_streamer_tracking, v.vod_id, v.streamer_id)
                futures[fut] = v.vod_id

            for fut in futures:
                try:
                    res = fut.result()
                    results.append(res)
                except Exception as e:
                    logger.error(f"Worker execution failed for {futures[fut]}: {e}")

        return results

    def _run_with_streamer_tracking(self, vod_id: str, streamer_id: str) -> HarvestedVodRecord:
        """Wrapper ensuring streamer active counter is decremented upon completion."""
        try:
            return self.harvest_vod(vod_id)
        finally:
            with self._lock:
                cnt = self._active_streamers.get(streamer_id, 1) - 1
                if cnt <= 0:
                    self._active_streamers.pop(streamer_id, None)
                else:
                    self._active_streamers[streamer_id] = cnt

    def _resolve_vod_directory(self, vod: HarvestedVodRecord, target: StreamerTargetRecord) -> Path:
        """Determines target homelab folder: vods/<streamer_id>/<YYYY-MM-DD>_<vod_id>/."""
        base_dir = self.homelab_root / "vods"
        if target.destination_override:
            base_dir = Path(target.destination_override)

        date_prefix = "unknown_date"
        if vod.published_at:
            date_prefix = vod.published_at.strftime("%Y-%m-%d")

        folder_name = f"{date_prefix}_{vod.vod_id}"
        return base_dir / target.streamer_id / folder_name

    def _simulate_harvest(
        self,
        vod: HarvestedVodRecord,
        target: StreamerTargetRecord,
        vod_dir: Path,
    ) -> None:
        """Creates valid simulated assets for testing and verification without network calls."""
        # 1. media.mp4
        media_path = vod_dir / "media.mp4"
        with open(media_path, "wb") as f:
            # Write a minimal simulated payload
            f.write(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"SIMULATED_STREAM_VIDEO" * 100)

        # 2. chat.json
        chat_path = vod_dir / "chat.json"
        if target.include_chat:
            simulated_chat: List[Dict[str, Any]] = [
                {
                    "message_id": f"msg-{i}",
                    "timestamp_offset": float(i * 1.5),
                    "user_id": f"user_{i % 5}",
                    "author_name": f"Viewer_{i % 5}",
                    "content": f"Message {i} from {target.display_name} chat! POGGERS LUL",
                    "emotes": [{"id": "pog", "name": "POGGERS", "count": 1}] if i % 2 == 0 else [],
                    "badges": ["subscriber"] if i % 3 == 0 else [],
                    "metadata": {},
                }
                for i in range(20)
            ]
            with open(chat_path, "w", encoding="utf-8") as f:
                json.dump(simulated_chat, f, indent=2)

        # 3. metadata.json
        meta_path = vod_dir / "metadata.json"
        metadata = {
            "vod_id": vod.vod_id,
            "streamer_id": target.streamer_id,
            "display_name": target.display_name,
            "title": vod.title,
            "platform": vod.platform,
            "duration_sec": vod.duration_sec or 60.0,
            "quality": target.quality_preset,
            "tags": target.tags,
            "raw_metadata": vod.raw_metadata,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # 4. thumbnail.jpg
        thumb_path = vod_dir / "thumbnail.jpg"
        with open(thumb_path, "wb") as f:
            f.write(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb" + b"\x00" * 32)

    def _execute_ytdlp_harvest(
        self,
        vod: HarvestedVodRecord,
        target: StreamerTargetRecord,
        vod_dir: Path,
    ) -> None:
        """Downloads real media and chat using yt-dlp."""
        if not self.ytdlp_bin:
            raise RuntimeError("yt-dlp binary not found. Cannot execute download.")

        media_path = vod_dir / "media.mp4"
        meta_path = vod_dir / "metadata.json"

        # Save metadata JSON first
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(vod.raw_metadata or {"title": vod.title}, f, indent=2)

        # Build yt-dlp format string based on quality preset
        format_str = "bestvideo+bestaudio/best"
        if target.quality_preset == "1080p":
            format_str = "bestvideo[height<=1080]+bestaudio/best[height<=1080]"
        elif target.quality_preset == "720p":
            format_str = "bestvideo[height<=720]+bestaudio/best[height<=720]"
        elif target.quality_preset == "audio_only":
            format_str = "bestaudio/best"

        url = vod.raw_metadata.get("webpage_url") or vod.raw_metadata.get("url") or vod.vod_id

        cmd = [
            self.ytdlp_bin,
            "--no-playlist",
            "-f", format_str,
            "-o", str(media_path),
            "--write-thumbnail",
            "--convert-thumbnails", "jpg",
            url,
        ]

        if target.include_chat:
            cmd.extend(["--write-subs", "--sub-langs", "all,-live_chat"])

        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=600,
        )
        if res.returncode != 0:
            raise RuntimeError(f"yt-dlp failed with return code {res.returncode}: {res.stderr}")
