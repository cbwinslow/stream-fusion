"""Targeted Streamer Roster Ingestion & Homelab Harvester Pipeline (Spec 25)."""

from stream_fusion.harvester.bridge import ingest_to_pipeline, verify_harvest_checksums
from stream_fusion.harvester.catalog import HarvestCatalog
from stream_fusion.harvester.crawler import ChannelVodCrawler
from stream_fusion.harvester.engine import HomelabHarvester, compute_file_sha256
from stream_fusion.harvester.roster import RosterLoader
from stream_fusion.harvester.daemon import HomelabDaemon
from stream_fusion.harvester.scheduler import (
    HomelabScheduler,
    ScheduledTask,
    run_storage_retention_cleanup,
)
from stream_fusion.harvester.synergy import (
    audit_harvested_sponsors,
    feed_adaptive_slang,
    ingest_chatter_safety,
    seed_voiceprints,
)

__all__ = [
    "ChannelVodCrawler",
    "HarvestCatalog",
    "HomelabDaemon",
    "HomelabHarvester",
    "HomelabScheduler",
    "RosterLoader",
    "ScheduledTask",
    "audit_harvested_sponsors",
    "compute_file_sha256",
    "feed_adaptive_slang",
    "ingest_chatter_safety",
    "ingest_to_pipeline",
    "run_storage_retention_cleanup",
    "seed_voiceprints",
    "verify_harvest_checksums",
]
