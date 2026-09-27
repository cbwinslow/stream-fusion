"""Advanced SubStation Alpha (.ass) animated karaoke subtitle generator."""

import math
from pathlib import Path
from typing import List, Optional

from stream_fusion.models.schemas import WordTiming


def format_ass_timestamp(seconds: float) -> str:
    """Formats seconds as H:MM:SS.cs (e.g. 0:01:23.45)."""
    clamped = max(0.0, seconds)
    hours = int(clamped // 3600)
    minutes = int((clamped % 3600) // 60)
    secs = int(clamped % 60)
    centis = int(round((clamped - math.floor(clamped)) * 100))
    if centis >= 100:
        secs += 1
        centis = 0
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def generate_karaoke_ass(
    words: List[WordTiming],
    output_path: Path,
    base_offset_sec: float = 0.0,
    max_words_per_line: int = 5,
    font_size: int = 68,
    margin_v: int = 280,
) -> Path:
    """Generates an animated word-by-word karaoke ASS subtitle file for vertical video.
    
    Args:
        words: List of WordTiming models with start, end, and word text.
        output_path: Target .ass file path.
        base_offset_sec: Time in seconds to subtract from word timestamps (for clipped highlights).
        max_words_per_line: Maximum number of words displayed on screen at once.
        font_size: Subtitle font size (scaled for 1080x1920).
        margin_v: Vertical margin from bottom edge (in pixels).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = f"""[Script Info]
Title: StreamFusion Auto-Karaoke Highlights
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Karaoke,Arial Black,{font_size},&H00FFFFFF,&H0000FFFF,&H00000000,&H90000000,-1,0,0,0,100,100,1,0,1,5,3,2,40,40,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    if not words:
        output_path.write_text(header, encoding="utf-8")
        return output_path

    # Filter and offset words
    rel_words: List[WordTiming] = []
    for w in words:
        w_start = max(0.0, w.start - base_offset_sec)
        w_end = max(w_start + 0.05, w.end - base_offset_sec)
        rel_words.append(WordTiming(word=w.word.strip(), start=w_start, end=w_end))

    # Chunk into lines of max_words_per_line or sentence breaks
    lines = []
    current_line = []
    for w in rel_words:
        if not w.word:
            continue
        if current_line and (len(current_line) >= max_words_per_line or (w.start - current_line[-1].end > 0.8)):
            lines.append(current_line)
            current_line = []
        current_line.append(w)
    if current_line:
        lines.append(current_line)

    dialogue_entries = []
    for group in lines:
        line_start = group[0].start
        line_end = max(line_start + 0.5, group[-1].end + 0.1)

        karaoke_parts = []
        for w in group:
            dur_cs = max(1, int(round((w.end - w.start) * 100)))
            karaoke_parts.append(f"{{\\k{dur_cs}}}{w.word}")

        formatted_text = " ".join(karaoke_parts)
        start_str = format_ass_timestamp(line_start)
        end_str = format_ass_timestamp(line_end)
        dialogue_entries.append(
            f"Dialogue: 0,{start_str},{end_str},Karaoke,,0,0,0,,{formatted_text}"
        )

    content = header + "\n".join(dialogue_entries) + "\n"
    output_path.write_text(content, encoding="utf-8")
    return output_path
