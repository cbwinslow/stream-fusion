"""Chat replay parser, latency calibrator, and sentiment analyzer."""

from collections import Counter
from typing import Dict, List
import json
from pathlib import Path

from stream_fusion.models.schemas import ChatMessage, ChatEmote


# Well-known Twitch/BTTV/7TV emote polarities for stream reaction analysis
EMOTE_POLARITY_LEXICON = {
    # Positive / Hype / Amusement
    "OMEGALUL": 0.8,
    "LUL": 0.7,
    "LULW": 0.8,
    "KEKW": 0.8,
    "ICANT": 0.8,
    "Pog": 0.9,
    "PogChamp": 0.9,
    "POGGERS": 0.9,
    "W": 0.7,
    "based": 0.7,
    "TRUE": 0.6,
    "NODDERS": 0.6,
    "EZ": 0.6,
    "Clap": 0.5,

    # Negative / Skeptical / Disgust
    "L": -0.7,
    "cap": -0.6,
    "NOPERS": -0.6,
    "monkaW": -0.4,
    "monkaS": -0.4,
    "Aware": -0.5,
    "cringe": -0.8,
    "weirdchamp": -0.8,
    "DESPAIR": -0.7,
    "???": -0.3,
    "HUH": -0.3,
}


class ChatAnalyzer:
    """Parses and analyzes chat streams with broadcast latency compensation."""

    def __init__(self, latency_offset_sec: float = 4.5):
        self.latency_offset_sec = latency_offset_sec

    def parse_twitch_downloader_json(self, json_path: Path) -> List[ChatMessage]:
        """Parses standard TwitchDownloader / chat-downloader JSON format."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_comments = data if isinstance(data, list) else data.get("comments", [])
        messages: List[ChatMessage] = []

        for idx, item in enumerate(raw_comments):
            offset = item.get("content_offset_seconds", 0.0)
            commenter = item.get("commenter", {})
            msg_body = item.get("message", {}).get("body", "")

            # Extract emotes if present
            emotes = []
            for em in item.get("message", {}).get("fragments", []):
                em_id = em.get("emoticon_id")
                if em_id:
                    emotes.append(ChatEmote(id=str(em_id), name=em.get("text", "")))

            messages.append(
                ChatMessage(
                    message_id=str(item.get("_id", f"msg_{idx}")),
                    timestamp_offset=float(offset),
                    user_id=str(commenter.get("_id", "anon")),
                    author_name=commenter.get("display_name", "Anonymous"),
                    content=msg_body,
                    emotes=emotes,
                )
            )

        return messages

    def aggregate_into_buckets(
        self,
        messages: List[ChatMessage],
        stream_duration_sec: float,
        bucket_size_sec: float = 2.0,
    ) -> List[Dict[str, object]]:
        """Groups messages into calibrated time buckets offset by broadcast latency."""
        num_buckets = int(stream_duration_sec // bucket_size_sec) + 1
        buckets: List[List[ChatMessage]] = [[] for _ in range(num_buckets)]

        for msg in messages:
            # Calibrate: Stream event occurred BEFORE chat message was posted
            calibrated_time = max(0.0, msg.timestamp_offset - self.latency_offset_sec)
            b_idx = int(calibrated_time // bucket_size_sec)
            if b_idx < num_buckets:
                buckets[b_idx].append(msg)

        results = []
        for idx, b_msgs in enumerate(buckets):
            start_sec = idx * bucket_size_sec
            end_sec = start_sec + bucket_size_sec
            count = len(b_msgs)
            velocity = count / bucket_size_sec

            # Count words and emotes
            word_counts = Counter()
            sentiment_total = 0.0
            sentiment_weight = 0.0

            for m in b_msgs:
                for token in m.content.split():
                    word_counts[token] += 1
                    if token in EMOTE_POLARITY_LEXICON:
                        pol = EMOTE_POLARITY_LEXICON[token]
                        sentiment_total += pol
                        sentiment_weight += 1.0

            avg_sentiment = (
                (sentiment_total / sentiment_weight) if sentiment_weight > 0 else 0.0
            )

            # Top 5 emotes/tokens
            top_emotes = dict(word_counts.most_common(5))

            results.append(
                {
                    "bucket_index": idx,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "chat_message_count": count,
                    "chat_velocity_per_sec": velocity,
                    "dominant_emotes": top_emotes,
                    "chat_sentiment_polarity": round(avg_sentiment, 3),
                }
            )

        return results
