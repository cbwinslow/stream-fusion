"""Advanced Chat NLP, fine-grained emotional intent, copy-pasta meme tracking, and chatter influence ranking."""

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

from stream_fusion.models.schemas import (
    AudioSegment,
    ChatMessage,
    ChatIntentDistribution,
    MemeBurstEvent,
    ChatterProfile,
)


# Normalized emotion and culture intents
INTENT_WEIGHTS = {
    "AMUSEMENT": 0.8,
    "HYPE": 0.9,
    "DISBELIEF": -0.3,
    "DISGUST_CRINGE": -0.7,
    "AGREEMENT": 0.6,
    "HEART_WARM": 0.9,
    "QUESTION": 0.0,
    "NEUTRAL": 0.0,
}

INTENT_LEXICON: Dict[str, Set[str]] = {
    "AMUSEMENT": {
        "lul", "lulw", "kekw", "omegalul", "icant", "lmao", "lmfao", "dying",
        "haha", "hahaha", "lol", "rofl", "hehe", "pepega",
    },
    "HYPE": {
        "pog", "pogchamp", "poggers", "pogbones", "letsgo", "w", "clip", "hypers",
        "insane", "clutch", "holy", "goat", "ez", "clap",
    },
    "DISBELIEF": {
        "huh", "cap", "fake", "noshot", "surely", "aintnoway", "what", "whaaat",
        "fr", "sus", "monkaw", "monkas",
    },
    "DISGUST_CRINGE": {
        "cringe", "weirdchamp", "aware", "despair", "wutface", "dansgame", "ew",
        "gross", "yikes", "nopers", "l",
    },
    "AGREEMENT": {
        "true", "based", "facts", "real", "gigachad", "nodders", "agree", "+1",
        "yep", "frfr",
    },
    "HEART_WARM": {
        "biblethump", "feelsstrongman", "widepeepohappy", "wholesome", "peepohappy",
        "feelsgoodman", "love", "heart", "<3",
    },
    "QUESTION": {
        "why", "how", "who", "where", "when", "explain", "lore",
    },
}


def normalize_repeated_chars(token: str) -> str:
    """Collapses repeated characters (e.g. 'loooool' -> 'lol', 'poggg' -> 'pog', '????' -> '?')."""
    token = token.lower()
    # Replace sequences of 3+ repeated characters with 1 or 2
    collapsed = re.sub(r"(.)\1{2,}", r"\1", token)
    return collapsed


def tokenize_for_similarity(text: str) -> Set[str]:
    """Tokenizes text into a set of normalized alphanumeric words."""
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = [t.strip() for t in cleaned.split() if len(t.strip()) > 1]
    return set(tokens)


def jaccard_similarity(tokens_a: Set[str], tokens_b: Set[str]) -> float:
    """Computes Jaccard similarity between two sets of tokens."""
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return float(intersection) / float(union) if union > 0 else 0.0


class ChatNLPAnalyzer:
    """Analyzes Twitch / live stream chat using semiotic and streamer-specific heuristics."""

    def __init__(self, custom_lexicon: Optional[Dict[str, Set[str]]] = None):
        self.lexicon = INTENT_LEXICON.copy()
        if custom_lexicon:
            for intent, words in custom_lexicon.items():
                self.lexicon[intent] = self.lexicon.get(intent, set()).union(words)

    def classify_message(self, message: ChatMessage) -> ChatIntentDistribution:
        """Classifies a single chat message into an intent distribution."""
        raw_text = message.content.strip()
        tokens = [normalize_repeated_chars(t) for t in raw_text.split()]

        # Check question punctuation
        has_question_mark = "?" in raw_text

        scores: Dict[str, float] = defaultdict(float)

        if has_question_mark:
            scores["QUESTION"] += 1.0

        for token in tokens:
            cleaned = re.sub(r"[^\w<3]", "", token)
            for intent, keywords in self.lexicon.items():
                if cleaned in keywords or token in keywords:
                    scores[intent] += 1.0

        total_matches = sum(scores.values())
        if total_matches == 0:
            scores["NEUTRAL"] = 1.0
            total_matches = 1.0

        normalized_scores = {k: round(v / total_matches, 3) for k, v in scores.items()}
        primary_intent = max(normalized_scores.items(), key=lambda x: x[1])[0]

        # Calculate overall valence (-1.0 to 1.0)
        valence = 0.0
        for intent, score in normalized_scores.items():
            valence += score * INTENT_WEIGHTS.get(intent, 0.0)

        return ChatIntentDistribution(
            primary_intent=primary_intent,
            intent_scores=normalized_scores,
            valence=round(valence, 3),
        )

    def analyze_message_stream(
        self, messages: List[ChatMessage]
    ) -> List[Tuple[ChatMessage, ChatIntentDistribution]]:
        """Processes a sequence of chat messages with their intent distributions."""
        return [(msg, self.classify_message(msg)) for msg in messages]


class MemeBurstTracker:
    """Tracks copy-pasta propagation and viral meme spikes in chat using sliding Jaccard similarity."""

    def __init__(
        self,
        window_sec: float = 10.0,
        similarity_threshold: float = 0.65,
        min_distinct_authors: int = 5,
    ):
        self.window_sec = window_sec
        self.similarity_threshold = similarity_threshold
        self.min_distinct_authors = min_distinct_authors

    def detect_meme_bursts(self, messages: List[ChatMessage]) -> List[MemeBurstEvent]:
        """Identifies copy-pastas and viral chat bursts across the message stream."""
        if not messages:
            return []

        # Sort messages by timestamp
        sorted_msgs = sorted(messages, key=lambda m: m.timestamp_offset)
        n = len(sorted_msgs)
        token_sets = [tokenize_for_similarity(m.content) for m in sorted_msgs]

        visited: Set[int] = set()
        bursts: List[MemeBurstEvent] = []
        burst_idx = 0

        for i in range(n):
            if i in visited:
                continue
            t_i = token_sets[i]
            if len(t_i) == 0:
                continue

            current_cluster: List[int] = [i]
            start_time = sorted_msgs[i].timestamp_offset

            # Search forward within window
            for j in range(i + 1, n):
                if j in visited:
                    continue
                time_diff = sorted_msgs[j].timestamp_offset - start_time
                if time_diff > self.window_sec:
                    break

                t_j = token_sets[j]
                sim = jaccard_similarity(t_i, t_j)
                if sim >= self.similarity_threshold:
                    current_cluster.append(j)

            # Check if cluster meets author threshold
            authors = {sorted_msgs[idx].author_name for idx in current_cluster}
            if len(authors) >= self.min_distinct_authors:
                # Mark as visited
                visited.update(current_cluster)

                cluster_msgs = [sorted_msgs[idx] for idx in current_cluster]
                burst_start = cluster_msgs[0].timestamp_offset
                burst_end = cluster_msgs[-1].timestamp_offset
                duration = max(1.0, burst_end - burst_start)
                velocity = round(len(cluster_msgs) / duration, 2)

                # Representative text: most frequent content
                content_counter = Counter(m.content.strip() for m in cluster_msgs)
                rep_text = content_counter.most_common(1)[0][0]

                # Peak timestamp: center or bin with highest concentration
                burst_peak = (burst_start + burst_end) / 2.0

                origin_msg = cluster_msgs[0]
                burst_event = MemeBurstEvent(
                    meme_id=f"meme_{burst_idx}",
                    representative_text=rep_text,
                    origin_message_id=origin_msg.message_id,
                    origin_user_id=origin_msg.user_id,
                    origin_author_name=origin_msg.author_name,
                    burst_start_sec=round(burst_start, 2),
                    burst_peak_sec=round(burst_peak, 2),
                    burst_end_sec=round(burst_end, 2),
                    propagation_velocity=velocity,
                    unique_spreaders=len(authors),
                    total_occurrences=len(cluster_msgs),
                )
                bursts.append(burst_event)
                burst_idx += 1

        return bursts


class ChatterInfluenceScorer:
    """Calculates Chatter Influence ('Opinion Leader') ranking based on meme bursts and streamer reactions."""

    def __init__(self, alpha: float = 0.5, beta: float = 0.4, gamma: float = 0.1):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def score_chatters(
        self,
        messages: List[ChatMessage],
        meme_bursts: List[MemeBurstEvent],
        streamer_audio: Optional[List[AudioSegment]] = None,
        streamer_response_window: Tuple[float, float] = (3.0, 8.0),
    ) -> List[ChatterProfile]:
        """Calculates chatter profiles and ranks them by influence score."""
        author_totals = Counter(m.author_name for m in messages)
        author_ids = {m.author_name: m.user_id for m in messages}

        # 1. First Meme Origin Count
        first_origins: Counter = Counter()
        for burst in meme_bursts:
            first_origins[burst.origin_author_name] += 1

        # 2. Streamer Verbal Response Count
        streamer_responses: Counter = Counter()
        if streamer_audio:
            # Filter to STREAMER segments
            streamer_segments = [
                seg for seg in streamer_audio
                if seg.speaker_label in ("STREAMER", "SPEAKER_00") and seg.transcript.strip()
            ]

            for msg in messages:
                msg_tokens = tokenize_for_similarity(msg.content)
                if len(msg_tokens) < 2:
                    continue  # Skip single character or emote spam

                # Check audio segments in response window [t + 3.0, t + 8.0]
                min_t = msg.timestamp_offset + streamer_response_window[0]
                max_t = msg.timestamp_offset + streamer_response_window[1]

                for seg in streamer_segments:
                    if min_t <= seg.start_sec <= max_t or (seg.start_sec <= min_t and seg.end_sec >= min_t):
                        seg_tokens = tokenize_for_similarity(seg.transcript)
                        # Check if chatter words are echoed by streamer
                        common = msg_tokens & seg_tokens
                        if len(common) >= 2 or (len(msg_tokens) == 2 and len(common) >= 1):
                            streamer_responses[msg.author_name] += 1
                            break

        profiles: List[ChatterProfile] = []
        for author, total_msgs in author_totals.items():
            f_count = first_origins[author]
            s_count = streamer_responses[author]
            # Influence formula:
            # score = alpha * origins + beta * streamer_responses + gamma * log(1 + total_messages)
            score = (
                self.alpha * f_count
                + self.beta * s_count
                + self.gamma * math.log(1 + total_msgs)
            )

            profiles.append(
                ChatterProfile(
                    user_id=author_ids.get(author, "unknown"),
                    author_name=author,
                    total_messages=total_msgs,
                    first_meme_origin_count=f_count,
                    streamer_response_count=s_count,
                    influence_score=round(score, 3),
                )
            )

        # Sort profiles by influence score descending
        profiles.sort(key=lambda p: (p.influence_score, p.total_messages), reverse=True)
        return profiles
