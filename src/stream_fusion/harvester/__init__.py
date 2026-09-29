"""Targeted Streamer Roster Ingestion & Homelab Harvester Pipeline (Spec 25)."""

from stream_fusion.harvester.bridge import ingest_to_pipeline, verify_harvest_checksums
from stream_fusion.harvester.catalog import HarvestCatalog
from stream_fusion.harvester.crawler import ChannelVodCrawler
from stream_fusion.harvester.engine import HomelabHarvester, compute_file_sha256
from stream_fusion.harvester.roster import RosterLoader
from stream_fusion.harvester.synergy import (
    audit_harvested_sponsors,
    feed_adaptive_slang,
    ingest_chatter_safety,
    seed_voiceprints,
)

__all__ = [
    "ChannelVodCrawler",
    "HarvestCatalog",
    "HomelabHarvester",
    "RosterLoader",
    "audit_harvested_sponsors",
    "compute_file_sha256",
    "feed_adaptive_slang",
    "ingest_chatter_safety",
    "ingest_to_pipeline",
    "seed_voiceprints",
    "verify_harvest_checksums",
]
