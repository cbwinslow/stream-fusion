"""Co-Stream Conversational Dynamics and Debate Analyzer (Spec 23)."""

from collections import defaultdict
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple
import uuid

from stream_fusion.models.schemas import (
    AudioSegment,
    CoStreamDebateTurn,
)

logger = logging.getLogger(__name__)


class CoStreamDebateAnalyzer:
    """Analyzes conversational turn-taking, speech dominance, interruptions, and stance consensus
    between co-streaming creators across synchronized audio feeds.
    """

    def __init__(
        self,
        min_turn_duration_sec: float = 0.5,
        interrupt_overlap_sec: float = 0.3,
    ):
        self.min_turn_duration_sec = min_turn_duration_sec
        self.interrupt_overlap_sec = interrupt_overlap_sec

    def extract_turns(
        self,
        aligned_segments: Sequence[Tuple[float, float, str, AudioSegment]],
        speaker_name_map: Optional[Dict[str, str]] = None,
    ) -> List[CoStreamDebateTurn]:
        """Extracts conversational turns from chronologically aligned audio segments.
        
        Args:
            aligned_segments: List of (unified_start, unified_end, channel_id, audio_segment).
            speaker_name_map: Optional mapping from channel_id or segment.speaker_label to display name.
            
        Returns:
            List of CoStreamDebateTurn models.
        """
        if not aligned_segments:
            return []

        name_map = speaker_name_map or {}
        turns: List[CoStreamDebateTurn] = []
        prev_turn: Optional[CoStreamDebateTurn] = None

        for u_start, u_end, ch_id, seg in aligned_segments:
            duration = max(0.0, u_end - u_start)
            if duration < self.min_turn_duration_sec:
                continue

            # Determine speaker display name
            spk_label = seg.speaker_label
            if spk_label in name_map:
                speaker = name_map[spk_label]
            elif ch_id in name_map:
                speaker = name_map[ch_id]
            elif spk_label and not spk_label.upper().startswith("SPEAKER_"):
                speaker = spk_label
            else:
                speaker = f"{ch_id}:{spk_label}"

            # Simple sentiment proxy from transcript tokens
            s_score = self._estimate_speech_sentiment(seg.transcript)

            # Interrupt detection: did this turn start before the previous turn ended?
            interrupts = False
            if prev_turn and prev_turn.speaker_name != speaker:
                if u_start < (prev_turn.end_sec - self.interrupt_overlap_sec):
                    interrupts = True

            turn = CoStreamDebateTurn(
                turn_id=f"turn-{uuid.uuid4().hex[:6]}",
                speaker_name=speaker,
                channel_id=ch_id,
                start_sec=round(u_start, 3),
                end_sec=round(u_end, 3),
                duration_sec=round(duration, 3),
                transcript=seg.transcript.strip(),
                sentiment_score=round(s_score, 3),
                interrupts_previous=interrupts,
            )
            turns.append(turn)
            prev_turn = turn

        return turns

    def analyze_debate(
        self,
        aligned_segments: Sequence[Tuple[float, float, str, AudioSegment]],
        speaker_name_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Performs complete conversational turn, talk-time, and debate analysis.
        
        Returns:
            Dict containing:
                - turns: List of CoStreamDebateTurn
                - talk_time_sec: Dict[speaker, float]
                - talk_time_percent: Dict[speaker, float]
                - interrupt_counts: Dict[speaker, int]
                - debate_moments: List[Dict] (points where speakers expressed opposite polarity)
        """
        turns = self.extract_turns(aligned_segments, speaker_name_map=speaker_name_map)
        if not turns:
            return {
                "turns": [],
                "talk_time_sec": {},
                "talk_time_percent": {},
                "interrupt_counts": {},
                "debate_moments": [],
            }

        talk_time: Dict[str, float] = defaultdict(float)
        interrupt_counts: Dict[str, int] = defaultdict(int)

        for t in turns:
            talk_time[t.speaker_name] += t.duration_sec
            if t.interrupts_previous:
                interrupt_counts[t.speaker_name] += 1

        total_talk_sec = sum(talk_time.values())
        talk_time_pct: Dict[str, float] = {}
        for spk, secs in talk_time.items():
            talk_time_pct[spk] = round((secs / total_talk_sec) * 100.0, 1) if total_talk_sec > 0 else 0.0

        # Detect debate moments: adjacent turns between different speakers with opposing sentiment
        debate_moments: List[Dict[str, Any]] = []
        for i in range(1, len(turns)):
            t_curr = turns[i]
            t_prev = turns[i - 1]

            if t_curr.speaker_name != t_prev.speaker_name:
                # Sign flip with substantial polarity
                if (t_curr.sentiment_score * t_prev.sentiment_score < -0.1) and (
                    abs(t_curr.sentiment_score - t_prev.sentiment_score) > 0.6
                ):
                    debate_moments.append({
                        "timestamp_sec": t_curr.start_sec,
                        "speaker_1": t_prev.speaker_name,
                        "quote_1": t_prev.transcript,
                        "sentiment_1": t_prev.sentiment_score,
                        "speaker_2": t_curr.speaker_name,
                        "quote_2": t_curr.transcript,
                        "sentiment_2": t_curr.sentiment_score,
                        "delta": round(abs(t_curr.sentiment_score - t_prev.sentiment_score), 2),
                    })

        return {
            "turns": [turn.model_dump() for turn in turns],
            "talk_time_sec": {spk: round(sec, 2) for spk, sec in talk_time.items()},
            "talk_time_percent": talk_time_pct,
            "interrupt_counts": dict(interrupt_counts),
            "debate_moments": debate_moments,
        }

    def _estimate_speech_sentiment(self, text: str) -> float:
        """Lightweight sentiment scorer for speech transcripts."""
        words = text.lower().split()
        if not words:
            return 0.0

        pos_lexicon = {
            "good", "great", "awesome", "amazing", "based", "true", "agree",
            "love", "incredible", "correct", "perfect", "w", "win", "fire",
        }
        neg_lexicon = {
            "bad", "terrible", "awful", "trash", "cringe", "fake", "cap",
            "disagree", "horrible", "wrong", "hate", "l", "loss", "cooked",
        }

        pos_count = sum(1 for w in words if w.strip("!?,.") in pos_lexicon)
        neg_count = sum(1 for w in words if w.strip("!?,.") in neg_lexicon)

        total = pos_count + neg_count
        if total == 0:
            return 0.0

        return float((pos_count - neg_count) / total)
