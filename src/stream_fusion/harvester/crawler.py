"""Channel VOD Discovery & Lookback Crawler (Spec 25).

Discovers recent VODs across Twitch, YouTube, and Kick via metadata extraction,
filters by lookback window, and deduplicates against the catalog database.
"""

from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Callable, Dict, List, Optional

from stream_fusion.harvester.catalog import HarvestCatalog
from stream_fusion.models.schemas import HarvestedVodRecord, HarvestStatus, StreamerTargetRecord

logger = logging.getLogger(__name__)


def find_ytdlp() -> Optional[str]:
    """Finds yt-dlp executable path."""
    return shutil.which("yt-dlp")


class ChannelVodCrawler:
    """Discovers and parses recent VODs from target streamer channels."""

    def __init__(
        self,
        ytdlp_bin: Optional[str] = None,
        extractor_override: Optional[Callable[[str, int], Dict[str, Any]]] = None,
    ):
        self.ytdlp_bin = ytdlp_bin or find_ytdlp()
        self._extractor_override = extractor_override

    def discover_target_vods(
        self,
        target: StreamerTargetRecord,
        catalog: Optional[HarvestCatalog] = None,
        auto_queue: bool = True,
    ) -> List[HarvestedVodRecord]:
        """Discovers recent VODs for a streamer target across all their channel URLs."""
        if not target.enabled:
            return []

        all_discovered: List[HarvestedVodRecord] = []
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=target.lookback_days)

        for channel_url in target.channel_urls:
            discovered = self.crawl_channel_url(
                channel_url=channel_url,
                streamer_id=target.streamer_id,
                primary_platform=target.primary_platform,
                max_vods=target.max_recent_vods,
                cutoff_date=cutoff_date,
            )

            for vod in discovered:
                # Deduplication check against catalog
                if catalog is not None:
                    existing = catalog.get_vod(vod.vod_id)
                    if existing is not None:
                        continue  # Already known

                    if auto_queue:
                        vod.status = HarvestStatus.QUEUED
                    catalog.add_vod(vod)

                all_discovered.append(vod)

        if catalog is not None:
            catalog.update_target_sync_time(target.streamer_id)

        return all_discovered

    def crawl_channel_url(
        self,
        channel_url: str,
        streamer_id: str,
        primary_platform: str = "TWITCH",
        max_vods: int = 5,
        cutoff_date: Optional[datetime] = None,
    ) -> List[HarvestedVodRecord]:
        """Extracts VOD metadata from a channel URL via yt-dlp."""
        raw_info = self._extract_metadata(channel_url, max_vods)
        if not raw_info:
            return []

        platform = self._detect_platform(channel_url, primary_platform)
        entries = raw_info.get("entries") or [raw_info]

        records: List[HarvestedVodRecord] = []
        for entry in entries:
            if not entry:
                continue

            raw_id = entry.get("id") or entry.get("url") or ""
            if not raw_id:
                continue

            # Deterministic prefixed VOD ID
            vod_id = self._normalize_vod_id(str(raw_id), platform)
            title = entry.get("title") or f"{streamer_id} VOD {raw_id}"
            duration = float(entry.get("duration") or 0.0)

            # Published at timestamp
            pub_date = self._parse_publish_date(entry)
            if cutoff_date and pub_date and pub_date < cutoff_date:
                # Outside lookback window
                continue

            # Extract thumbnail
            thumbnail_url = entry.get("thumbnail") or None
            if not thumbnail_url and entry.get("thumbnails"):
                thumbnail_url = entry["thumbnails"][-1].get("url")

            rec = HarvestedVodRecord(
                vod_id=vod_id,
                streamer_id=streamer_id,
                platform=platform,
                title=title,
                published_at=pub_date,
                duration_sec=duration,
                status=HarvestStatus.DISCOVERED,
                thumbnail_path=thumbnail_url,
                raw_metadata=entry,
            )
            records.append(rec)

        return records

    def sync_all(
        self,
        catalog: HarvestCatalog,
        auto_queue: bool = True,
    ) -> Dict[str, List[HarvestedVodRecord]]:
        """Crawls all enabled streamer targets in the catalog."""
        targets = catalog.list_targets(enabled_only=True)
        results: Dict[str, List[HarvestedVodRecord]] = {}

        for target in targets:
            try:
                vods = self.discover_target_vods(target, catalog=catalog, auto_queue=auto_queue)
                results[target.streamer_id] = vods
            except Exception as e:
                logger.error(f"Failed to crawl streamer {target.streamer_id}: {e}")
                results[target.streamer_id] = []

        return results

    def _extract_metadata(self, channel_url: str, max_vods: int) -> Dict[str, Any]:
        """Runs yt-dlp metadata extraction or custom extractor override."""
        if self._extractor_override:
            return self._extractor_override(channel_url, max_vods)

        if not self.ytdlp_bin:
            logger.warning("yt-dlp not found on system. Returning empty discovery.")
            return {}

        cmd = [
            self.ytdlp_bin,
            "--flat-playlist",
            "-J",
            "--playlist-end", str(max_vods),
            "--no-warnings",
            channel_url,
        ]

        try:
            res = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=45,
            )
            if res.returncode != 0:
                logger.warning(f"yt-dlp metadata extraction failed for {channel_url}: {res.stderr}")
                return {}
            return json.loads(res.stdout)
        except Exception as e:
            logger.error(f"Error running yt-dlp on {channel_url}: {e}")
            return {}

    @staticmethod
    def _detect_platform(channel_url: str, default_platform: str) -> str:
        """Infers platform from channel URL."""
        url_lower = channel_url.lower()
        if "twitch.tv" in url_lower:
            return "TWITCH"
        elif "youtube.com" in url_lower or "youtu.be" in url_lower:
            return "YOUTUBE"
        elif "kick.com" in url_lower:
            return "KICK"
        return default_platform.upper()

    @staticmethod
    def _normalize_vod_id(raw_id: str, platform: str) -> str:
        """Normalizes VOD ID with a platform prefix for global uniqueness."""
        prefix = platform.lower()
        raw_slug = re.sub(r"[^\w\-]", "", raw_id)
        if raw_slug.startswith(f"{prefix}_"):
            return raw_slug
        return f"{prefix}_{raw_slug}"

    @staticmethod
    def _parse_publish_date(entry: Dict[str, Any]) -> Optional[datetime]:
        """Parses publish timestamp or upload date from yt-dlp entry."""
        # 1. Unix timestamp
        ts = entry.get("timestamp") or entry.get("release_timestamp")
        if ts is not None:
            try:
                return datetime.fromtimestamp(float(ts), tz=timezone.utc)
            except Exception:
                pass

        # 2. ISO / YYYYMMDD string
        upload_date = entry.get("upload_date")
        if upload_date and len(upload_date) == 8:
            try:
                return datetime.strptime(upload_date, "%Y%m%d").replace(tzinfo=timezone.utc)
            except Exception:
                pass

        return None
