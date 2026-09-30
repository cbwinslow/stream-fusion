"""Director Agent for Autonomous Short Production (Spec 21).

Curates viral moments, defines narrative arcs, and sets hook strategies.
"""

from typing import Dict, List, Optional
import uuid

from stream_fusion.models.schemas import (
    AudioSegment,
    ChatMessage,
    DynamicMomentThresholds,
    FusionSlice,
    NarrativeArc,
    ShortCandidate,
    StreamAnalysisResult,
    WordTiming,
)


class DirectorAgent:
    """Multi-modal director that evaluates stream highlights and curates short candidates."""

    def __init__(self, min_duration_sec: float = 15.0, max_duration_sec: float = 60.0):
        self.min_duration_sec = min_duration_sec
        self.max_duration_sec = max_duration_sec

    def select_candidates(
        self,
        analysis: StreamAnalysisResult,
        fusion_slices: Optional[List[FusionSlice]] = None,
        audio_segments: Optional[List[AudioSegment]] = None,
        chat_messages: Optional[List[ChatMessage]] = None,
        top_k: Optional[int] = 3,
        min_highlight_score: float = 0.4,
        dynamic: bool = False,
        thresholds: Optional[DynamicMomentThresholds] = None,
    ) -> List[ShortCandidate]:
        """Scans highlight moments and fusion slices to curate vertical short candidates."""
        candidates: List[ShortCandidate] = []
        fusion_slices = fusion_slices or getattr(analysis, "slices", []) or getattr(analysis, "fusion_slices", [])
        audio_segments = audio_segments or getattr(analysis, "audio_segments", [])
        chat_messages = chat_messages or getattr(analysis, "chat_messages", [])

        effective_min_score = (thresholds.min_highlight_score if (dynamic and thresholds) else min_highlight_score)
        min_separation = (thresholds.min_separation_sec if (dynamic and thresholds) else (60.0 if dynamic else 20.0))
        safety_max = (thresholds.safety_max_shorts if (dynamic and thresholds) else (50 if dynamic else (top_k or 3)))

        # Identify candidate anchor timestamps from highlights or fusion slices
        anchor_points: List[Dict[str, float]] = []

        if analysis.highlights:
            for hl in analysis.highlights:
                if isinstance(hl, dict):
                    ts = hl.get("timestamp_sec", 0.0)
                    score = hl.get("score", 0.5)
                else:
                    ts = getattr(hl, "timestamp_sec", 0.0)
                    score = getattr(hl, "score", 0.5)
                if score >= effective_min_score:
                    anchor_points.append({"timestamp_sec": float(ts), "score": float(score)})

        # Scan fusion slices for high highlight scores or significant activity
        if fusion_slices:
            for s in fusion_slices:
                s_ts = getattr(s, "timestamp_sec", None)
                if s_ts is None and hasattr(s, "start_sec"):
                    s_ts = (s.start_sec + s.end_sec) / 2.0
                s_ts = float(s_ts or 0.0)
                hl_score = getattr(s, "highlight_score", None) or getattr(s, "agreement_score", 0.0) or 0.0
                if hl_score >= effective_min_score:
                    anchor_points.append({
                        "timestamp_sec": s_ts,
                        "score": float(hl_score),
                    })

        # If still none and not dynamic, fall back to peaks of chat density
        if not anchor_points and fusion_slices:
            def _get_slice_density(x):
                return getattr(x, "chat_density", None) or getattr(x, "chat_velocity_per_sec", None) or float(getattr(x, "chat_message_count", 0))

            sorted_slices = sorted(fusion_slices, key=_get_slice_density, reverse=True)
            candidate_limit = top_k if (not dynamic and top_k) else 5
            for s in sorted_slices[:candidate_limit]:
                s_ts = getattr(s, "timestamp_sec", None)
                if s_ts is None and hasattr(s, "start_sec"):
                    s_ts = (s.start_sec + s.end_sec) / 2.0
                s_ts = float(s_ts or 0.0)
                score = getattr(s, "highlight_score", None) or getattr(s, "agreement_score", 0.5) or 0.5
                anchor_points.append({
                    "timestamp_sec": s_ts,
                    "score": float(score),
                })

        # Deduplicate anchor points within minimum separation window
        deduped_anchors: List[Dict[str, float]] = []
        for ap in sorted(anchor_points, key=lambda x: x["score"], reverse=True):
            if not any(abs(ap["timestamp_sec"] - d["timestamp_sec"]) < min_separation for d in deduped_anchors):
                deduped_anchors.append(ap)
            if not dynamic and top_k and len(deduped_anchors) >= top_k * 2:
                break
            if dynamic and safety_max and len(deduped_anchors) >= safety_max:
                break

        # Process each anchor into a ShortCandidate
        selected_anchors = deduped_anchors if dynamic else (deduped_anchors[:top_k] if top_k else deduped_anchors)
        if dynamic and safety_max:
            selected_anchors = selected_anchors[:safety_max]

        for anchor in selected_anchors:
            peak_ts = anchor["timestamp_sec"]
            score = anchor["score"]

            # Compute window: lead-in of 6s before peak, total duration ~30s clamped to [min, max]
            start_sec = max(0.0, peak_ts - 6.0)
            end_sec = start_sec + 30.0

            # Try to snap to natural audio speech boundaries if available
            snapped_start, snapped_end = self._snap_to_speech(start_sec, end_sec, audio_segments)
            duration = snapped_end - snapped_start

            # Extract chat messages and calculate burst Z-score in window
            burst_zscore, dominant_slangs = self._analyze_chat_burst(snapped_start, snapped_end, chat_messages)

            # Determine dominant emotion and narrative arc
            narrative_arc, emotion = self._determine_arc(score, burst_zscore, snapped_start, snapped_end, fusion_slices)

            # Extract hook and summary text
            hook_text, summary = self._extract_hook_and_summary(snapped_start, snapped_end, audio_segments)

            candidate = ShortCandidate(
                candidate_id=str(uuid.uuid4())[:8],
                start_sec=round(snapped_start, 2),
                end_sec=round(snapped_end, 2),
                duration_sec=round(duration, 2),
                peak_timestamp_sec=round(peak_ts, 2),
                highlight_score=round(score, 3),
                chat_burst_zscore=round(burst_zscore, 2),
                primary_emotion=emotion,
                narrative_arc=narrative_arc,
                hook_text=hook_text or "Wait until you see what happens...",
                summary=summary or f"Streamer reaction climax at {int(peak_ts)}s",
                dominant_slang=dominant_slangs,
            )
            candidates.append(candidate)

        return candidates

    def _snap_to_speech(
        self, start_sec: float, end_sec: float, audio_segments: Optional[List[AudioSegment]]
    ) -> tuple[float, float]:
        """Snaps start and end boundaries to sentence or segment starts/ends."""
        if not audio_segments:
            return start_sec, min(start_sec + self.max_duration_sec, end_sec)

        best_start = start_sec
        best_end = end_sec

        for seg in audio_segments:
            # If segment starts within 3 seconds of proposed start, snap to it
            if abs(seg.start_sec - start_sec) <= 3.0:
                best_start = seg.start_sec
                break

        for seg in reversed(audio_segments):
            # If segment ends within 4 seconds of proposed end, snap to it
            if abs(seg.end_sec - end_sec) <= 4.0 and seg.end_sec > best_start + self.min_duration_sec:
                best_end = seg.end_sec
                break

        duration = best_end - best_start
        if duration < self.min_duration_sec:
            best_end = best_start + self.min_duration_sec
        elif duration > self.max_duration_sec:
            best_end = best_start + self.max_duration_sec

        return best_start, best_end

    def _analyze_chat_burst(
        self, start_sec: float, end_sec: float, chat_messages: Optional[List[ChatMessage]]
    ) -> tuple[float, List[str]]:
        """Calculates chatter velocity Z-score and surfaces frequent slang / emotes in window."""
        if not chat_messages:
            return 1.5, []

        window_msgs = [m for m in chat_messages if start_sec <= m.timestamp_offset <= end_sec]
        msg_count = len(window_msgs)

        # Baseline density estimate across all chat
        total_span = max(1.0, max(m.timestamp_offset for m in chat_messages) - min(m.timestamp_offset for m in chat_messages))
        avg_rate = len(chat_messages) / total_span
        window_duration = max(1.0, end_sec - start_sec)
        window_rate = msg_count / window_duration

        std = max(0.5, avg_rate * 0.5)
        zscore = (window_rate - avg_rate) / std
        clamped_z = max(0.0, min(10.0, zscore))

        # Slang & emote discovery
        token_freq: Dict[str, int] = {}
        for m in window_msgs:
            for w in m.content.split():
                clean_w = w.strip("!?,.:;\"'()").lower()
                if len(clean_w) >= 3 and not clean_w.startswith("http"):
                    token_freq[clean_w] = token_freq.get(clean_w, 0) + 1

        common_slang = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)
        dominant = [t[0] for t in common_slang[:5] if t[1] >= 2]
        return clamped_z, dominant

    def _determine_arc(
        self,
        score: float,
        burst_z: float,
        start_sec: float,
        end_sec: float,
        fusion_slices: Optional[List[FusionSlice]],
    ) -> tuple[NarrativeArc, str]:
        """Infers the narrative storytelling structure and primary emotional tone."""
        if burst_z >= 3.5:
            return NarrativeArc.INSTANT_CLIMAX_REACTION, "HYSTERICAL_LAUGHTER"

        if score >= 0.8:
            return NarrativeArc.HOOK_BUILDUP_PAYOFF, "HYPE"

        # Check fusion slices for disagreement / debate
        if fusion_slices:
            slice_window = []
            for s in fusion_slices:
                s_ts = getattr(s, "timestamp_sec", None)
                if s_ts is None and hasattr(s, "start_sec"):
                    s_ts = (s.start_sec + s.end_sec) / 2.0
                if s_ts is not None and start_sec <= s_ts <= end_sec:
                    slice_window.append(s)

            if slice_window:
                avg_agreement = sum(getattr(s, "agreement_score", 0.5) or 0.5 for s in slice_window) / len(slice_window)
                if avg_agreement < 0.4:
                    return NarrativeArc.HOT_TAKE_AND_DEBATE, "CONTROVERSY"

        return NarrativeArc.HOOK_BUILDUP_PAYOFF, "EXCITEMENT"

    def _extract_hook_and_summary(
        self, start_sec: float, end_sec: float, audio_segments: Optional[List[AudioSegment]]
    ) -> tuple[str, str]:
        """Extracts the first spoken sentence as the visual hook and builds a summary."""
        if not audio_segments:
            return "", ""

        relevant = [s for s in audio_segments if s.end_sec >= start_sec and s.start_sec <= end_sec]
        if not relevant:
            return "", ""

        # First segment provides the opening hook line
        hook_candidate = relevant[0].transcript.strip()
        words = hook_candidate.split()
        short_hook = " ".join(words[:12]) + ("..." if len(words) > 12 else "")

        # Whole transcript provides the summary
        full_text = " ".join(s.transcript.strip() for s in relevant)
        summary = full_text[:160] + ("..." if len(full_text) > 160 else "")

        return short_hook, summary
