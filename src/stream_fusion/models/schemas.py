"""Pydantic data models for StreamFusion data contracts."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ChatEmote(BaseModel):
    id: str
    name: str
    count: int = 1


class ChatMessage(BaseModel):
    message_id: str
    timestamp_offset: float = Field(..., description="Seconds offset from stream start")
    user_id: str
    author_name: str
    content: str
    emotes: List[ChatEmote] = Field(default_factory=list)
    badges: List[str] = Field(default_factory=list)


class WordTiming(BaseModel):
    word: str
    start: float
    end: float
    probability: float = 1.0


class AudioSegment(BaseModel):
    segment_id: int
    start_sec: float
    end_sec: float
    speaker_label: str = Field(
        ...,
        description="Identified speaker, e.g. 'STREAMER', 'EXTERNAL_VIDEO', 'UNKNOWN'"
    )
    transcript: str
    confidence: float = 1.0
    words: Optional[List[WordTiming]] = None


class BoundingBox(BaseModel):
    label: str
    confidence: float
    box: List[float] = Field(..., description="[ymin, xmin, ymax, xmax] normalized")


class VisualKeyframe(BaseModel):
    frame_index: int
    timestamp_sec: float
    scene_type: str = Field(
        default="UNKNOWN",
        description="'GAMEPLAY', 'REACT_VIDEO', 'FULLSCREEN_CAM', 'BROWSER', 'UNKNOWN'"
    )
    screen_summary: str = Field(default="", description="Dense caption of the on-screen event")
    ocr_text_blocks: List[str] = Field(default_factory=list)
    streamer_facial_expression: Optional[str] = None
    detected_objects: List[BoundingBox] = Field(default_factory=list)


class FusionSlice(BaseModel):
    bucket_index: int
    start_sec: float
    end_sec: float

    # Audio State
    active_speakers: List[str] = Field(default_factory=list)
    streamer_transcript: Optional[str] = None
    external_audio_transcript: Optional[str] = None

    # Visual State
    active_scene_type: str = "UNKNOWN"
    visual_description: str = ""
    screen_ocr: List[str] = Field(default_factory=list)

    # Chat State (Calibrated for stream latency)
    chat_message_count: int = 0
    chat_velocity_per_sec: float = 0.0
    dominant_emotes: Dict[str, int] = Field(default_factory=dict)
    chat_sentiment_polarity: float = Field(
        default=0.0,
        description="-1.0 (very negative) to +1.0 (very positive)"
    )

    # Signals
    is_spike_moment: bool = False
    agreement_score: Optional[float] = None


class StreamAnalysisResult(BaseModel):
    stream_id: str
    title: Optional[str] = None
    duration_sec: float
    total_chat_messages: int
    slices: List[FusionSlice] = Field(default_factory=list)
    highlights: List[Dict[str, object]] = Field(default_factory=list)
