"""StreamFusion Co-Stream & Cross-Platform Alignment Subsystem (Spec 23)."""

from stream_fusion.costream.audience_comparator import (
    CrossAudienceComparator,
    POLARITY_LEXICON,
)
from stream_fusion.costream.coordinator import (
    MultiStreamCoordinator,
    MultiStreamSupervisor,
)
from stream_fusion.costream.debate_analyzer import CoStreamDebateAnalyzer
from stream_fusion.costream.exceptions import (
    ChannelConnectionError,
    CoStreamError,
    ResourceCeilingExceededError,
    StreamBufferOverflowError,
    StreamSynchronizationError,
)
from stream_fusion.costream.short_composer import MultiAngleShortComposer
from stream_fusion.costream.sync_engine import CrossStreamSyncEngine

__all__ = [
    "ChannelConnectionError",
    "CoStreamDebateAnalyzer",
    "CoStreamError",
    "CrossAudienceComparator",
    "CrossStreamSyncEngine",
    "MultiAngleShortComposer",
    "MultiStreamCoordinator",
    "MultiStreamSupervisor",
    "POLARITY_LEXICON",
    "ResourceCeilingExceededError",
    "StreamBufferOverflowError",
    "StreamSynchronizationError",
]
