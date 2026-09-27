"""Fusion matrix engine for aligning audio, vision, and chat streams."""

from typing import Dict, List, Optional
from stream_fusion.models.schemas import (
    AudioSegment,
    VisualKeyframe,
    FusionSlice,
    StreamAnalysisResult,
)


class FusionEngine:
    """Combines extracted modalities into a unified temporal matrix."""

    def __init__(self, bucket_size_sec: float = 2.0):
        self.bucket_size_sec = bucket_size_sec

    def build_matrix(
        self,
        stream_id: str,
        duration_sec: float,
        audio_segments: List[AudioSegment],
        visual_keyframes: List[VisualKeyframe],
        chat_buckets: List[Dict[str, object]],
    ) -> StreamAnalysisResult:
        """Merges all modalities onto synchronized time buckets."""
        num_buckets = int(duration_sec // self.bucket_size_sec) + 1
        slices: List[FusionSlice] = []

        # Index visual keyframes by nearest bucket
        vis_by_bucket: Dict[int, VisualKeyframe] = {}
        for vk in visual_keyframes:
            b_idx = int(vk.timestamp_sec // self.bucket_size_sec)
            vis_by_bucket[b_idx] = vk

        # Build index of chat buckets
        chat_by_bucket: Dict[int, Dict[str, object]] = {
            cb["bucket_index"]: cb for cb in chat_buckets
        }

        # Calculate average chat velocity for spike thresholding
        all_velocities = [
            cb.get("chat_velocity_per_sec", 0.0) for cb in chat_buckets
        ]
        mean_vel = (
            sum(all_velocities) / len(all_velocities) if all_velocities else 1.0
        )
        spike_threshold = max(2.0, mean_vel * 2.5)

        current_visual_state: Optional[VisualKeyframe] = None

        for b_idx in range(num_buckets):
            start_sec = b_idx * self.bucket_size_sec
            end_sec = start_sec + self.bucket_size_sec

            # 1. Audio overlap check
            streamer_texts = []
            external_texts = []
            active_speakers = set()

            for seg in audio_segments:
                # Check for temporal overlap: max(start1, start2) < min(end1, end2)
                if max(start_sec, seg.start_sec) < min(end_sec, seg.end_sec):
                    active_speakers.add(seg.speaker_label)
                    if seg.speaker_label == "STREAMER":
                        streamer_texts.append(seg.transcript)
                    else:
                        external_texts.append(seg.transcript)

            # 2. Visual state carry-over (state persistence)
            if b_idx in vis_by_bucket:
                current_visual_state = vis_by_bucket[b_idx]

            vis_desc = (
                current_visual_state.screen_summary
                if current_visual_state
                else "Unchanged scene"
            )
            vis_type = (
                current_visual_state.scene_type
                if current_visual_state
                else "UNKNOWN"
            )
            vis_ocr = (
                current_visual_state.ocr_text_blocks
                if current_visual_state
                else []
            )

            # 3. Chat data
            c_data = chat_by_bucket.get(b_idx, {})
            msg_count = c_data.get("chat_message_count", 0)
            vel = c_data.get("chat_velocity_per_sec", 0.0)
            emotes = c_data.get("dominant_emotes", {})
            sentiment = c_data.get("chat_sentiment_polarity", 0.0)

            # 4. Spike detection
            is_spike = vel >= spike_threshold

            # 5. Agreement score: If streamer spoke and chat has high positive/negative alignment
            agreement: Optional[float] = None
            if streamer_texts and msg_count > 3:
                agreement = sentiment

            slices.append(
                FusionSlice(
                    bucket_index=b_idx,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    active_speakers=list(active_speakers),
                    streamer_transcript=" ".join(streamer_texts) if streamer_texts else None,
                    external_audio_transcript=" ".join(external_texts) if external_texts else None,
                    active_scene_type=vis_type,
                    visual_description=vis_desc,
                    screen_ocr=vis_ocr,
                    chat_message_count=msg_count,
                    chat_velocity_per_sec=vel,
                    dominant_emotes=emotes,
                    chat_sentiment_polarity=sentiment,
                    is_spike_moment=is_spike,
                    agreement_score=agreement,
                )
            )

        # Detect candidate highlight moments
        highlights = self._extract_highlights(slices)

        total_msgs = sum(s.chat_message_count for s in slices)

        return StreamAnalysisResult(
            stream_id=stream_id,
            duration_sec=duration_sec,
            total_chat_messages=total_msgs,
            slices=slices,
            highlights=highlights,
        )

    def _extract_highlights(self, slices: List[FusionSlice]) -> List[Dict[str, object]]:
        """Identifies peak moments based on chat spikes and speech triggers."""
        highlights = []
        for s in slices:
            if s.is_spike_moment:
                highlights.append(
                    {
                        "timestamp_sec": s.start_sec,
                        "reason": "Audience Reaction Spike",
                        "velocity": s.chat_velocity_per_sec,
                        "dominant_emotes": s.dominant_emotes,
                        "streamer_said": s.streamer_transcript,
                        "screen_context": s.visual_description,
                    }
                )
        return highlights
