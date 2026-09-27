"""Automated semantic segment and chapter boundary detector."""

from typing import List, Optional
from pydantic import BaseModel, Field

from stream_fusion.models.schemas import FusionSlice


class StreamSegment(BaseModel):
    segment_id: int
    title: str
    start_sec: float
    end_sec: float
    duration_sec: float
    category: str  # JUST_CHATTING, REACT_VIDEO, GAMEPLAY, AFK
    dominant_topic: Optional[str] = None
    average_chat_velocity: float = 0.0
    youtube_chapter_timestamp: str = ""


class StreamSegmentDetector:
    """Partitions continuous stream timeline slices into discrete semantic chapters."""

    def __init__(self, min_segment_duration_sec: float = 60.0):
        self.min_segment_duration_sec = min_segment_duration_sec

    def _format_timestamp(self, seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"

    def detect_segments(self, slices: List[FusionSlice]) -> List[StreamSegment]:
        """Detects segment transition boundaries based on visual scenes and topic changes."""
        if not slices:
            return []

        boundaries = [0]
        current_scene = slices[0].active_scene_type

        for i in range(1, len(slices)):
            s = slices[i]
            # Transition triggers:
            # 1. Macro visual scene type shift
            scene_changed = s.active_scene_type != current_scene and s.active_scene_type != "UNKNOWN"
            # 2. Minimum duration threshold
            elapsed_since_last = s.start_sec - slices[boundaries[-1]].start_sec

            if scene_changed and elapsed_since_last >= self.min_segment_duration_sec:
                boundaries.append(i)
                current_scene = s.active_scene_type

        # Build segments from boundary intervals
        segments: List[StreamSegment] = []
        for b_idx in range(len(boundaries)):
            start_i = boundaries[b_idx]
            end_i = boundaries[b_idx + 1] if b_idx + 1 < len(boundaries) else len(slices)

            sub_slices = slices[start_i:end_i]
            if not sub_slices:
                continue

            start_t = sub_slices[0].start_sec
            end_t = sub_slices[-1].end_sec
            dur = end_t - start_t

            # Dominant scene category
            scene_counts = {}
            for ss in sub_slices:
                scene_counts[ss.active_scene_type] = scene_counts.get(ss.active_scene_type, 0) + 1
            dom_scene = max(scene_counts, key=scene_counts.get) if scene_counts else "JUST_CHATTING"

            # Derive title from OCR or scene
            title = "Livestream Segment"
            ocr_candidates = [ss.screen_ocr[0] for ss in sub_slices if ss.screen_ocr]
            if ocr_candidates:
                title = f"Reacting: {ocr_candidates[0]}"
            elif dom_scene == "GAMEPLAY":
                title = "Gameplay Session"
            elif dom_scene == "REACT_VIDEO":
                title = "Video Reaction & Commentary"
            elif dom_scene == "FULLSCREEN_CAM":
                title = "Just Chatting & Community Discussion"

            avg_vel = sum(ss.chat_velocity_per_sec for ss in sub_slices) / len(sub_slices)
            ts_str = self._format_timestamp(start_t)

            segments.append(
                StreamSegment(
                    segment_id=b_idx + 1,
                    title=title,
                    start_sec=start_t,
                    end_sec=end_t,
                    duration_sec=round(dur, 2),
                    category=dom_scene,
                    average_chat_velocity=round(avg_vel, 2),
                    youtube_chapter_timestamp=f"{ts_str} {title}",
                )
            )

        return segments

    def generate_youtube_chapters(self, segments: List[StreamSegment]) -> str:
        """Formats segments into copy-paste YouTube chapter markers."""
        return "\n".join(seg.youtube_chapter_timestamp for seg in segments)
