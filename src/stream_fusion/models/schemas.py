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


# --- Spec 08: Chat NLP & Community Intelligence ---

class ChatIntentDistribution(BaseModel):
    primary_intent: str
    intent_scores: Dict[str, float] = Field(default_factory=dict)
    valence: float = 0.0


class MemeBurstEvent(BaseModel):
    meme_id: str
    representative_text: str
    origin_message_id: str
    origin_user_id: str
    origin_author_name: str
    burst_start_sec: float
    burst_peak_sec: float
    burst_end_sec: float
    propagation_velocity: float = Field(..., description="Messages per second during burst")
    unique_spreaders: int
    total_occurrences: int


class ChatterProfile(BaseModel):
    user_id: str
    author_name: str
    total_messages: int = 0
    first_meme_origin_count: int = 0
    streamer_response_count: int = 0
    influence_score: float = 0.0


# --- Spec 09: Sponsor & Brand Quantifier ---

class BrandProfile(BaseModel):
    brand_id: str
    brand_name: str
    aliases: List[str] = Field(default_factory=list)
    promo_codes: List[str] = Field(default_factory=list)
    product_keywords: List[str] = Field(default_factory=list)


class SponsorSegment(BaseModel):
    segment_id: int
    brand_id: str
    start_sec: float
    end_sec: float
    matched_audio_transcripts: List[str] = Field(default_factory=list)
    matched_ocr_texts: List[str] = Field(default_factory=list)


class SponsorImpactReport(BaseModel):
    brand_id: str
    brand_name: str
    sponsor_segment: SponsorSegment
    chat_mention_count: int
    mention_velocity: float
    window_start_sec: float
    window_end_sec: float
    sentiment_during_sponsor: float
    stream_baseline_sentiment: float
    sentiment_delta: float
    backlash_index: float
    brand_attention_score: float
    summary: str


# --- Spec 10: Stateful Resumption & Checkpoint ---

class ChunkManifest(BaseModel):
    stream_id: str
    chunk_duration_sec: float = 300.0
    total_chunks_expected: int = 0
    completed_chunks: List[int] = Field(default_factory=list)
    status: str = "IN_PROGRESS"
    last_updated: str = ""
    chunk_files: Dict[str, List[str]] = Field(default_factory=dict)

