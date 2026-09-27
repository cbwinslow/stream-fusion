"""Sponsor & Brand Performance Quantifier (Spec 09).

Detects multi-modal brand mentions in audio transcripts and OCR visual keyframes,
isolates sponsor segments, and quantifies audience attention, sentiment delta, and backlash.
"""

import re
from typing import Dict, List, Optional, Set, Tuple

from stream_fusion.chat.analyzer import EMOTE_POLARITY_LEXICON
from stream_fusion.models.schemas import (
    AudioSegment,
    BrandProfile,
    ChatMessage,
    SponsorImpactReport,
    SponsorSegment,
    VisualKeyframe,
)


BACKLASH_TOKENS = {
    "sellout",
    "ad",
    "ads",
    "adw",
    "skip",
    "skipad",
    "cringe",
    "l",
    "sponsor",
    "adbreak",
}


def fuzzy_match_token(pattern: str, text: str) -> bool:
    """Matches pattern within text case-insensitively with punctuation stripping."""
    pattern_norm = re.sub(r"[^\w]", "", pattern.lower())
    text_norm = re.sub(r"[^\w\s]", " ", text.lower())
    words = text_norm.split()

    if not pattern_norm:
        return False

    # Check direct substring
    if pattern_norm in "".join(words):
        return True

    # Check word or bigram match
    for w in words:
        if pattern_norm == w:
            return True
        # Allow edit distance 1 for slight transcription typos
        if len(pattern_norm) >= 4 and abs(len(w) - len(pattern_norm)) <= 1:
            diff = sum(1 for a, b in zip(pattern_norm, w) if a != b) + abs(
                len(pattern_norm) - len(w)
            )
            if diff <= 1:
                return True

    return False


class SponsorDetector:
    """Detects brand appearances in audio and visual streams and extracts sponsor segments."""

    def __init__(self, merge_threshold_sec: float = 30.0):
        self.merge_threshold_sec = merge_threshold_sec

    def _matches_brand(self, text: str, brand: BrandProfile) -> bool:
        """Determines if text mentions brand name, aliases, or promo codes."""
        targets = [brand.brand_name] + brand.aliases + brand.promo_codes
        for target in targets:
            if fuzzy_match_token(target, text):
                return True
        return False

    def detect_segments(
        self,
        brand: BrandProfile,
        audio_segments: Optional[List[AudioSegment]] = None,
        keyframes: Optional[List[VisualKeyframe]] = None,
    ) -> List[SponsorSegment]:
        """Detects sponsor mentions across audio and video keyframes and groups them into segments."""
        raw_mentions: List[Tuple[float, float, str, str]] = []  # (start, end, type, text)

        # 1. Scan audio segments
        if audio_segments:
            for audio in audio_segments:
                if self._matches_brand(audio.transcript, brand):
                    raw_mentions.append(
                        (audio.start_sec, audio.end_sec, "AUDIO", audio.transcript)
                    )

        # 2. Scan OCR blocks in keyframes
        if keyframes:
            for kf in keyframes:
                for ocr_block in kf.ocr_text_blocks:
                    if self._matches_brand(ocr_block, brand):
                        raw_mentions.append(
                            (kf.timestamp_sec, kf.timestamp_sec + 2.0, "OCR", ocr_block)
                        )

        if not raw_mentions:
            return []

        # Sort mentions chronologically
        raw_mentions.sort(key=lambda m: m[0])

        # Cluster mentions within merge_threshold_sec
        clusters: List[List[Tuple[float, float, str, str]]] = []
        current_cluster = [raw_mentions[0]]

        for mention in raw_mentions[1:]:
            last_end = current_cluster[-1][1]
            if mention[0] - last_end <= self.merge_threshold_sec:
                current_cluster.append(mention)
            else:
                clusters.append(current_cluster)
                current_cluster = [mention]

        if current_cluster:
            clusters.append(current_cluster)

        # Build SponsorSegments
        segments: List[SponsorSegment] = []
        for idx, cluster in enumerate(clusters):
            start_sec = min(m[0] for m in cluster)
            end_sec = max(m[1] for m in cluster)
            matched_audio = [m[3] for m in cluster if m[2] == "AUDIO"]
            matched_ocr = [m[3] for m in cluster if m[2] == "OCR"]

            segments.append(
                SponsorSegment(
                    segment_id=idx + 1,
                    brand_id=brand.brand_id,
                    start_sec=round(start_sec, 2),
                    end_sec=round(end_sec, 2),
                    matched_audio_transcripts=matched_audio,
                    matched_ocr_texts=matched_ocr,
                )
            )

        return segments


class SponsorReportGenerator:
    """Analyzes chat reaction metrics around sponsor segments and generates impact scorecards."""

    def __init__(self, window_buffer_sec: float = 60.0):
        self.window_buffer_sec = window_buffer_sec

    def _calculate_sentiment(self, messages: List[ChatMessage]) -> float:
        """Computes average polarity across messages based on lexicon."""
        total_score = 0.0
        total_weight = 0.0

        for msg in messages:
            for token in msg.content.split():
                if token in EMOTE_POLARITY_LEXICON:
                    total_score += EMOTE_POLARITY_LEXICON[token]
                    total_weight += 1.0

        return round(total_score / total_weight, 3) if total_weight > 0 else 0.0

    def generate_report(
        self,
        brand: BrandProfile,
        segment: SponsorSegment,
        all_chat_messages: List[ChatMessage],
    ) -> SponsorImpactReport:
        """Evaluates chat around sponsor segment and produces an impact report."""
        window_start = max(0.0, segment.start_sec - self.window_buffer_sec)
        window_end = segment.end_sec + self.window_buffer_sec
        window_duration = max(1.0, window_end - window_start)

        # Baseline sentiment across all chat messages
        baseline_sentiment = self._calculate_sentiment(all_chat_messages)

        # Messages within sponsor impact window
        window_msgs = [
            m
            for m in all_chat_messages
            if window_start <= m.timestamp_offset <= window_end
        ]
        window_sentiment = self._calculate_sentiment(window_msgs)
        sentiment_delta = round(window_sentiment - baseline_sentiment, 3)

        # Check brand keywords & promo codes in window
        brand_terms: Set[str] = set()
        for term in (
            [brand.brand_name]
            + brand.aliases
            + brand.promo_codes
            + brand.product_keywords
        ):
            brand_terms.add(term.lower())

        mention_count = 0
        backlash_count = 0

        for m in window_msgs:
            msg_lower = m.content.lower()
            tokens = set(re.findall(r"\w+", msg_lower))

            if any(term in msg_lower or term in tokens for term in brand_terms):
                mention_count += 1

            if any(b_token in tokens for b_token in BACKLASH_TOKENS):
                backlash_count += 1

        mention_velocity = round(mention_count / window_duration, 3)
        total_window_msgs = len(window_msgs)
        backlash_index = (
            round(backlash_count / total_window_msgs, 3)
            if total_window_msgs > 0
            else 0.0
        )

        # Brand Attention Score (0 - 100):
        # A_brand = 100 * (0.4 * min(1.0, mentions / 50) + 0.3 * ((S_sponsor + 1) / 2) + 0.3 * (1 - B_sponsor))
        term_mentions = 0.4 * min(1.0, mention_count / 50.0)
        term_sentiment = 0.3 * ((window_sentiment + 1.0) / 2.0)
        term_backlash = 0.3 * max(0.0, 1.0 - backlash_index)

        attention_score = round(
            max(0.0, min(100.0, 100.0 * (term_mentions + term_sentiment + term_backlash))),
            2,
        )

        summary = (
            f"Brand '{brand.brand_name}' segment [{segment.start_sec}s - {segment.end_sec}s] "
            f"generated {mention_count} mentions ({mention_velocity}/s). "
            f"Sentiment delta: {sentiment_delta:+.2f} relative to baseline. "
            f"Backlash index: {backlash_index:.1%}. Overall Attention Score: {attention_score}/100."
        )

        return SponsorImpactReport(
            brand_id=brand.brand_id,
            brand_name=brand.brand_name,
            sponsor_segment=segment,
            chat_mention_count=mention_count,
            mention_velocity=mention_velocity,
            window_start_sec=round(window_start, 2),
            window_end_sec=round(window_end, 2),
            sentiment_during_sponsor=window_sentiment,
            stream_baseline_sentiment=baseline_sentiment,
            sentiment_delta=sentiment_delta,
            backlash_index=backlash_index,
            brand_attention_score=attention_score,
            summary=summary,
        )
