"""Multi-Angle Co-Stream Highlight Climax Detection and Short Packaging (Spec 23)."""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple
import uuid

from stream_fusion.models.schemas import (
    AudioSegment,
    ChatMessage,
    CrossAudienceSentimentPoint,
    MultiAngleShortCandidate,
)

logger = logging.getLogger(__name__)


class MultiAngleShortComposer:
    """Detects multi-stream climax moments across co-streaming channels and composes
    multi-angle vertical short packages with spatial layout presets.
    """

    def __init__(
        self,
        min_duration_sec: float = 15.0,
        max_duration_sec: float = 60.0,
        consensus_threshold: float = 0.60,
    ):
        self.min_duration_sec = min_duration_sec
        self.max_duration_sec = max_duration_sec
        self.consensus_threshold = consensus_threshold

    def find_multi_angle_candidates(
        self,
        sentiment_timeline: Sequence[CrossAudienceSentimentPoint],
        aligned_messages: Sequence[Tuple[float, str, ChatMessage]],
        aligned_audio: Optional[Sequence[Tuple[float, float, str, AudioSegment]]] = None,
        top_k: int = 3,
    ) -> List[MultiAngleShortCandidate]:
        """Identifies peak climax windows where multiple channels experienced concurrent bursts.
        
        Args:
            sentiment_timeline: Bucketed cross-platform sentiment points.
            aligned_messages: Unified chronological chat messages.
            aligned_audio: Unified audio segments.
            top_k: Maximum number of short candidates to return.
            
        Returns:
            List of MultiAngleShortCandidate models sorted by virality score descending.
        """
        if not sentiment_timeline:
            return []

        # Find points with high multi-platform engagement or divergence
        candidates: List[MultiAngleShortCandidate] = []
        channels_seen = set()
        for _, ch_id, _ in aligned_messages:
            channels_seen.add(ch_id)
        channel_list = sorted(list(channels_seen))

        for point in sentiment_timeline:
            # Score this moment
            num_active_channels = len(point.channel_sentiments)
            if num_active_channels < 2 and len(channel_list) >= 2:
                continue

            # Calculate Virality Signal
            # 1. Platform sentiment magnitude
            plat_scores = list(point.platform_sentiments.values())
            intensity = max(abs(s) for s in plat_scores) if plat_scores else 0.0

            # 2. Consensus or dramatic polarization
            divergence_bonus = 20.0 if point.divergence_detected else 0.0
            agreement_bonus = 15.0 if point.cross_platform_agreement >= 0.85 else 0.0

            # 3. Message velocity in window
            center_ts = point.timestamp_sec
            w = point.window_sec
            window_msgs = [
                m for u_ts, ch, m in aligned_messages
                if abs(u_ts - center_ts) <= (self.min_duration_sec / 2.0)
            ]
            msg_count = len(window_msgs)

            # Combined virality score (0-100)
            base_score = (intensity * 40.0) + min(30.0, msg_count * 1.5) + divergence_bonus + agreement_bonus
            virality = float(max(10.0, min(99.0, base_score)))

            # Choose layout preset based on participating channel count
            participating = [ch for ch, s in point.channel_sentiments.items()] or channel_list
            if len(participating) == 2:
                layout = "STACKED_SPLIT"
            elif len(participating) > 2:
                layout = "QUAD_GRID"
            else:
                layout = "PICTURE_IN_PICTURE"

            # Boundaries snapped to min/max duration
            t_start = max(0.0, center_ts - (self.min_duration_sec / 2.0))
            t_end = t_start + self.min_duration_sec

            # Title and hook suggestions
            dominant_token = ""
            for em_list in point.dominant_emotes.values():
                if em_list:
                    dominant_token = em_list[0]
                    break

            if point.divergence_detected:
                title = f"Multi-Platform Clash: {dominant_token} Debate Moment"
                hooks = [
                    f"Twitch vs Kick vs YouTube completely split on this!",
                    f"One chat loved it, the other revolted!",
                ]
            else:
                title = f"Insane Sync: All Streams React to {dominant_token}"
                hooks = [
                    f"Every single stream reacted at the exact same second!",
                    f"When the entire co-stream loses their mind together...",
                ]

            cand = MultiAngleShortCandidate(
                candidate_id=f"mashort-{uuid.uuid4().hex[:8]}",
                start_sec=round(t_start, 2),
                end_sec=round(t_end, 2),
                duration_sec=round(t_end - t_start, 2),
                participating_channels=participating,
                layout_preset=layout,
                cross_platform_agreement=point.cross_platform_agreement,
                virality_score=round(virality, 1),
                title=title,
                hooks=hooks,
            )
            candidates.append(cand)

        # De-duplicate overlapping candidates
        candidates.sort(key=lambda c: c.virality_score, reverse=True)
        filtered: List[MultiAngleShortCandidate] = []
        for c in candidates:
            # Check overlap with already chosen candidates
            overlap = any(
                abs(c.start_sec - existing.start_sec) < (self.min_duration_sec * 0.75)
                for existing in filtered
            )
            if not overlap:
                filtered.append(c)
            if len(filtered) >= top_k:
                break

        return filtered
