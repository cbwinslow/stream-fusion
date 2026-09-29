"""Synergy Integrations between Harvester & StreamFusion Intelligence (Spec 25).

Connects ingested streamer rosters and harvested assets to:
- Spec 11: Speaker Voiceprint Enrollment
- Spec 12: Historical Chatter Safety Store
- Spec 17: Adaptive Slang Discovery
- Spec 09: Sponsor & Brand Performance Auditing
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from stream_fusion.analytics.sponsor_quantifier import SponsorImpactQuantifier
from stream_fusion.audio.voiceprint import VoiceprintLibrary
from stream_fusion.chat.profiler import ChatterProfileStore
from stream_fusion.models.schemas import (
    AudioSegment,
    BrandProfile,
    ChatMessage,
    HarvestedVodRecord,
    SponsorImpactReport,
    StreamerTargetRecord,
    VisualKeyframe,
)
from stream_fusion.nlp.adaptive_slang import AdaptiveSlangEngine

logger = logging.getLogger(__name__)


def seed_voiceprints(
    roster: List[StreamerTargetRecord],
    library: VoiceprintLibrary,
) -> int:
    """Enrolls any pre-seeded voiceprints defined in target streamer records."""
    count = 0
    for target in roster:
        if target.voiceprint_embedding and len(target.voiceprint_embedding) > 0:
            channel = target.channel_urls[0] if target.channel_urls else None
            library.enroll_creator(
                creator_id=target.streamer_id,
                display_name=target.display_name,
                embedding=target.voiceprint_embedding,
                primary_channel=channel,
            )
            count += 1
            logger.info(f"Enrolled voiceprint for streamer '{target.streamer_id}'")
    return count


def ingest_chatter_safety(
    chat_path: Path,
    store: ChatterProfileStore,
    streamer_id: str,
) -> int:
    """Ingests harvested chat messages into ChatterProfileStore for cross-stream safety."""
    p = Path(chat_path)
    if not p.exists():
        return 0

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        return 0

    count = 0
    for item in data:
        if isinstance(item, dict):
            try:
                msg = ChatMessage(**item)
                store.ingest_message(msg, vod_id=streamer_id)
                count += 1
            except Exception:
                pass
    return count



def feed_adaptive_slang(
    chat_path: Path,
    engine: AdaptiveSlangEngine,
    streamer_id: str,
) -> int:
    """Feeds harvested chat replay into AdaptiveSlangEngine to update community lexicon."""
    p = Path(chat_path)
    if not p.exists():
        return 0

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        return 0

    messages: List[ChatMessage] = []
    for item in data:
        if isinstance(item, dict):
            try:
                messages.append(ChatMessage(**item))
            except Exception:
                pass

    if not messages:
        return 0

    candidates, _ = engine.process_chat_stream(messages)
    return len(candidates)


def audit_harvested_sponsors(
    vod: HarvestedVodRecord,
    brands: List[BrandProfile],
    audio_segments: Optional[List[AudioSegment]] = None,
    keyframes: Optional[List[VisualKeyframe]] = None,
    chat_messages: Optional[List[ChatMessage]] = None,
) -> List[SponsorImpactReport]:
    """Audits stream sponsors against BrandProfile catalog."""
    quantifier = SponsorImpactQuantifier(brands=brands)
    result = quantifier.analyze_sponsors(
        stream_id=vod.vod_id,
        audio_segments=audio_segments,
        keyframes=keyframes,
        chat_messages=chat_messages,
        brand_profiles=brands,
    )
    return result.reports

