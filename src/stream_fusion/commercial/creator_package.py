"""Creator Studio & YouTube Long-Form Metadata Packager (Spec 33).

Generates YouTube chapters, high-CTR titles, SEO descriptions, thumbnail timestamps,
and executive daily recaps.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from stream_fusion.models.schemas import (
    AudioSegment,
    StreamAnalysisResult,
    YouTubeMetadataPackage,
)

logger = logging.getLogger(__name__)


def _format_timestamp(seconds: float) -> str:
    """Formats seconds to HH:MM:SS or MM:SS."""
    secs = int(seconds)
    hours = secs // 3600
    minutes = (secs % 3600) // 60
    remaining_secs = secs % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{remaining_secs:02d}"
    return f"{minutes:02d}:{remaining_secs:02d}"


class CreatorStudioPackager:
    """Produces YouTube packages, chapter timelines, and daily briefings."""

    def generate_youtube_package(
        self,
        analysis: StreamAnalysisResult,
        audio_segments: Optional[List[AudioSegment]] = None,
        claims: Optional[List[Any]] = None,
        creator_name: str = "Streamer",
    ) -> YouTubeMetadataPackage:
        """Assembles YouTube long-form publication assets."""
        duration = float(getattr(analysis, "duration_sec", 3600.0))
        highlights = getattr(analysis, "highlights", []) or []

        # 1. Build Chapter Markers
        chapters: List[Dict[str, str]] = [
            {"timestamp": "00:00", "label": "Stream Start & Warmup"}
        ]

        # Use sorted highlights to define macro chapters
        valid_highlights = []
        for hl in highlights:
            ts = hl.get("timestamp_sec", 0.0) if isinstance(hl, dict) else getattr(hl, "timestamp_sec", 0.0)
            score = hl.get("score", 0.0) if isinstance(hl, dict) else getattr(hl, "score", 0.0)
            if score >= 0.60:
                valid_highlights.append((ts, score))

        valid_highlights.sort(key=lambda x: x[0])

        last_ts = 0.0
        chapter_idx = 1
        for ts, score in valid_highlights:
            if ts - last_ts >= 900.0:  # Space chapters at least 15 minutes apart
                formatted_ts = _format_timestamp(ts)
                label = f"Major Climax & Reaction #{chapter_idx}"
                if score >= 0.85:
                    label = f"Unbelievable Climax #{chapter_idx} (Chat Pop-Off)"
                chapters.append({"timestamp": formatted_ts, "label": label})
                last_ts = ts
                chapter_idx += 1

        if duration > 1800.0 and (duration - last_ts) >= 600.0:
            chapters.append({"timestamp": _format_timestamp(duration - 300.0), "label": "Final Thoughts & Sign-off"})

        # 2. Generate 3 High-CTR Titles
        suggested_titles = [
            f"The Moment {creator_name} Realized Everything Changed...",
            f"{creator_name} Reacts to The Most Insane Stream Event of the Year",
            f"I Can't Believe This Actually Happened Live on Stream ({int(duration // 3600)}h Broadcast)",
        ]

        # 3. Daily Recap Bullet Points
        daily_recap_bullets = [
            f"Broadcasted live for {_format_timestamp(duration)} with {len(valid_highlights)} major highlight moments.",
            f"Chat reached peak velocity with community reactions across multiple game chapters.",
            f"Reviewed community debates and stream milestones with full multimodal verification.",
        ]

        # 4. Thumbnail Candidate Timestamps (top 4 highlight timestamps)
        thumbnail_timestamps = [
            round(ts, 1) for ts, _ in sorted(valid_highlights, key=lambda x: x[1], reverse=True)[:4]
        ]
        if not thumbnail_timestamps:
            thumbnail_timestamps = [round(duration * 0.25, 1), round(duration * 0.5, 1)]

        # 5. SEO Tags & Description
        seo_tags = [
            creator_name.lower(),
            "stream highlights",
            "gaming",
            "twitch vod",
            "reaction",
            "best moments",
            "full broadcast",
        ]

        chapter_lines = "\n".join(f"{c['timestamp']} - {c['label']}" for c in chapters)
        bullet_text = "\n• ".join(daily_recap_bullets)
        hashtag_name = creator_name.replace(' ', '')
        seo_description = f"""Full broadcast recording of {creator_name}'s live stream.

Timestamps & Chapters:
{chapter_lines}

📌 Highlights & Daily Recap:
• {bullet_text}

Subscribe for daily highlights, full VODs, and live moments!
#{hashtag_name} #Gaming #TwitchHighlights
"""

        return YouTubeMetadataPackage(
            suggested_titles=suggested_titles,
            chapters=chapters,
            seo_description=seo_description,
            seo_tags=seo_tags,
            thumbnail_candidate_timestamps=thumbnail_timestamps,
            daily_recap_bullets=daily_recap_bullets,
        )
