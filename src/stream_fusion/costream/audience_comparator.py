"""Cross-Platform Audience Sentiment Comparator and Meme Cascade Tracker (Spec 23)."""

from collections import Counter, defaultdict
import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from stream_fusion.models.schemas import (
    ChatMessage,
    CrossAudienceSentimentPoint,
    CrossStreamBurstPropagation,
    LivePlatform,
)

logger = logging.getLogger(__name__)

# Standard sentiment polarity map for cross-platform emotes and vernacular
POLARITY_LEXICON: Dict[str, float] = {
    # Strong Positive / Hype / Amusement
    "pog": 0.8,
    "poggers": 0.85,
    "w": 0.9,
    "based": 0.8,
    "gigachad": 0.85,
    "true": 0.7,
    "nodders": 0.7,
    "clap": 0.6,
    "omegalul": 0.7,
    "kekw": 0.7,
    "icant": 0.75,
    "lul": 0.6,
    "lulw": 0.65,
    "fire": 0.8,
    "goat": 0.9,
    "letsgo": 0.8,
    # Negative / Skepticism / Disapproval
    "l": -0.85,
    "cringe": -0.8,
    "cap": -0.75,
    "nopers": -0.7,
    "weirdchamp": -0.85,
    "monkaw": -0.5,
    "aware": -0.6,
    "despair": -0.7,
    "cooked": -0.8,
    "ratio": -0.7,
    "scam": -0.9,
    "boo": -0.8,
    "trash": -0.9,
    "mid": -0.6,
}


class CrossAudienceComparator:
    """Analyzes and compares multi-platform audience sentiment and meme cascades.
    
    Operates on temporally aligned multi-stream chat feeds to quantify cross-platform consensus,
    detect audience divergence (e.g. YouTube cheering while Twitch roasts), and measure viral
    propagation speed across channel boundaries.
    """

    def __init__(
        self,
        bucket_window_sec: float = 2.0,
        divergence_threshold: float = 0.75,
        min_messages_per_bucket: int = 2,
    ):
        self.bucket_window_sec = bucket_window_sec
        self.divergence_threshold = divergence_threshold
        self.min_messages_per_bucket = min_messages_per_bucket

    def score_message_sentiment(self, message: ChatMessage) -> float:
        """Computes rule-based sentiment polarity in [-1.0, 1.0] for a chat message."""
        tokens = message.content.lower().split()
        if not tokens:
            return 0.0

        scores: List[float] = []
        for t in tokens:
            # Clean punctuation
            clean_t = t.strip("!?,.:;\"'()[]{}")
            if clean_t in POLARITY_LEXICON:
                scores.append(POLARITY_LEXICON[clean_t])

        # Also check emote names
        for em in message.emotes:
            em_name = em.name.lower()
            if em_name in POLARITY_LEXICON:
                scores.append(POLARITY_LEXICON[em_name])

        if not scores:
            return 0.0

        return float(sum(scores) / len(scores))

    def extract_emotes_and_tokens(self, message: ChatMessage) -> List[str]:
        """Extracts significant tokens and emotes for frequency analysis."""
        items: List[str] = [em.name for em in message.emotes]
        words = message.content.split()
        for w in words:
            clean = w.strip("!?,.:;\"'()[]{}")
            if len(clean) >= 2 and (clean.isupper() or clean.lower() in POLARITY_LEXICON):
                items.append(clean)
        return items

    def compute_aligned_sentiment_timeline(
        self,
        aligned_messages: Sequence[Tuple[float, str, ChatMessage]],
        channel_to_platform: Optional[Dict[str, str]] = None,
    ) -> List[CrossAudienceSentimentPoint]:
        """Computes synchronized time-bucketed sentiment and agreement index.
        
        Args:
            aligned_messages: Chronologically sorted list of (unified_ts, channel_id, message).
            channel_to_platform: Optional mapping from channel_id to platform string (TWITCH, KICK, YOUTUBE_LIVE).
            
        Returns:
            List of CrossAudienceSentimentPoint objects.
        """
        if not aligned_messages:
            return []

        ch_plat = channel_to_platform or {}

        # 1. Bucket messages by time window
        min_ts = aligned_messages[0][0]
        max_ts = aligned_messages[-1][0]
        w = self.bucket_window_sec

        num_buckets = max(1, int(math.ceil((max_ts - min_ts) / w)) + 1)
        bucket_channel_msgs: Dict[int, Dict[str, List[ChatMessage]]] = defaultdict(lambda: defaultdict(list))

        for u_ts, ch_id, msg in aligned_messages:
            b_idx = int((u_ts - min_ts) / w)
            bucket_channel_msgs[b_idx][ch_id].append(msg)

        timeline: List[CrossAudienceSentimentPoint] = []

        for b_idx in range(num_buckets):
            b_center = round(min_ts + (b_idx * w) + (w / 2.0), 3)
            ch_data = bucket_channel_msgs.get(b_idx, {})

            channel_sentiments: Dict[str, float] = {}
            platform_sentiments_accum: Dict[str, List[float]] = defaultdict(list)
            dominant_emotes: Dict[str, List[str]] = defaultdict(list)

            total_bucket_msgs = sum(len(msgs) for msgs in ch_data.values())
            if total_bucket_msgs < self.min_messages_per_bucket:
                continue

            for ch_id, msgs in ch_data.items():
                s_vals = [self.score_message_sentiment(m) for m in msgs]
                avg_s = float(sum(s_vals) / len(s_vals)) if s_vals else 0.0
                channel_sentiments[ch_id] = round(avg_s, 3)

                # Map to platform
                platform = ch_plat.get(ch_id, ch_id.upper())
                platform_sentiments_accum[platform].extend(s_vals)

                # Emotes for this channel/platform
                em_counter: Counter = Counter()
                for m in msgs:
                    for em in self.extract_emotes_and_tokens(m):
                        em_counter[em] += 1
                top_em = [item for item, _ in em_counter.most_common(3)]
                dominant_emotes[ch_id] = top_em

            platform_sentiments: Dict[str, float] = {}
            for plat, vals in platform_sentiments_accum.items():
                platform_sentiments[plat] = round(float(sum(vals) / len(vals)), 3)

            # Compute Cross-Platform Agreement Index
            plat_scores = list(platform_sentiments.values())
            if len(plat_scores) >= 2:
                spread = max(plat_scores) - min(plat_scores)
                agreement = max(0.0, min(1.0, 1.0 - (spread / 2.0)))
            else:
                spread = 0.0
                agreement = 1.0

            # Divergence Detection
            divergence = False
            div_desc = None
            if spread >= self.divergence_threshold and len(plat_scores) >= 2:
                divergence = True
                sorted_plat = sorted(platform_sentiments.items(), key=lambda kv: kv[1])
                lowest_plat, lowest_score = sorted_plat[0]
                highest_plat, highest_score = sorted_plat[-1]
                div_desc = (
                    f"Audience divergence detected: {highest_plat} (+{highest_score:.2f}) vs "
                    f"{lowest_plat} ({lowest_score:.2f}) [spread: {spread:.2f}]"
                )

            point = CrossAudienceSentimentPoint(
                timestamp_sec=b_center,
                window_sec=w,
                platform_sentiments=platform_sentiments,
                channel_sentiments=channel_sentiments,
                cross_platform_agreement=round(agreement, 3),
                dominant_emotes=dominant_emotes,
                divergence_detected=divergence,
                divergence_description=div_desc,
            )
            timeline.append(point)

        return timeline

    def track_meme_cascade(
        self,
        aligned_messages: Sequence[Tuple[float, str, ChatMessage]],
        target_token: str,
        channel_to_platform: Optional[Dict[str, str]] = None,
        burst_threshold: int = 3,
        window_sec: float = 5.0,
    ) -> Optional[CrossStreamBurstPropagation]:
        """Tracks the chronological propagation of a slang or meme burst across channels/platforms.
        
        Args:
            aligned_messages: Chronologically sorted list of (unified_ts, channel_id, message).
            target_token: Token or emote string to track (e.g. 'OMEGALUL', 'Poggers', 'COOKED').
            channel_to_platform: Channel to platform map.
            burst_threshold: Minimum occurrences within window_sec to qualify as burst arrival.
            window_sec: Sliding window size for burst qualification.
            
        Returns:
            CrossStreamBurstPropagation if cascade detected, else None.
        """
        token_lower = target_token.lower()
        ch_plat = channel_to_platform or {}

        # Collect occurrences per channel: {channel: [timestamps]}
        channel_timestamps: Dict[str, List[float]] = defaultdict(list)
        for u_ts, ch_id, msg in aligned_messages:
            content_tokens = [t.strip("!?,.:;\"'()[]{}").lower() for t in msg.content.split()]
            emote_tokens = [em.name.lower() for em in msg.emotes]
            if token_lower in content_tokens or token_lower in emote_tokens:
                channel_timestamps[ch_id].append(u_ts)

        if not channel_timestamps:
            return None

        # Determine burst arrival time for each channel
        arrival_times: Dict[str, float] = {}
        for ch_id, t_list in channel_timestamps.items():
            if len(t_list) < burst_threshold:
                # Single message doesn't constitute a burst, but if small dataset, use first timestamp
                if t_list:
                    arrival_times[ch_id] = t_list[0]
                continue

            # Sliding window burst detection
            for i in range(len(t_list) - burst_threshold + 1):
                span = t_list[i + burst_threshold - 1] - t_list[i]
                if span <= window_sec:
                    arrival_times[ch_id] = t_list[i]
                    break
            if ch_id not in arrival_times and t_list:
                arrival_times[ch_id] = t_list[0]

        if len(arrival_times) < 2:
            return None

        # Find earliest arrival
        sorted_arrivals = sorted(arrival_times.items(), key=lambda kv: kv[1])
        origin_channel, origin_ts = sorted_arrivals[0]
        origin_platform = ch_plat.get(origin_channel, origin_channel.upper())

        cascade_timeline: Dict[str, float] = {}
        propagation_deltas: List[float] = []

        for ch_id, t_arr in sorted_arrivals:
            plat = ch_plat.get(ch_id, ch_id.upper())
            delta = max(0.0, t_arr - origin_ts)
            cascade_timeline[f"{plat}:{ch_id}"] = round(delta, 3)
            if ch_id != origin_channel:
                propagation_deltas.append(delta)

        avg_velocity = (
            float(sum(propagation_deltas) / len(propagation_deltas))
            if propagation_deltas
            else 0.0
        )

        return CrossStreamBurstPropagation(
            token=target_token,
            origin_platform=origin_platform,
            origin_channel=origin_channel,
            origin_timestamp=round(origin_ts, 3),
            cascade_timeline=cascade_timeline,
            cascade_velocity_sec=round(avg_velocity, 3),
        )

    def summarize_session_agreement(
        self,
        timeline: Sequence[CrossAudienceSentimentPoint],
    ) -> Dict[str, Any]:
        """Provides high-level session summary metrics of audience consensus."""
        if not timeline:
            return {
                "overall_agreement_index": 1.0,
                "total_points": 0,
                "divergence_count": 0,
                "most_divergent_moments": [],
            }

        agreements = [p.cross_platform_agreement for p in timeline]
        overall = float(sum(agreements) / len(agreements))

        divergent = [p for p in timeline if p.divergence_detected]
        sorted_divergent = sorted(
            divergent,
            key=lambda p: p.cross_platform_agreement,
        )

        top_divergent = [
            {
                "timestamp_sec": p.timestamp_sec,
                "cross_platform_agreement": p.cross_platform_agreement,
                "platform_sentiments": p.platform_sentiments,
                "description": p.divergence_description,
            }
            for p in sorted_divergent[:5]
        ]

        return {
            "overall_agreement_index": round(overall, 3),
            "total_points": len(timeline),
            "divergence_count": len(divergent),
            "most_divergent_moments": top_divergent,
        }
