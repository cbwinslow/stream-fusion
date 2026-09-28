"""Pydantic data models for StreamFusion data contracts."""

from typing import Any, Dict, List, Optional
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

    # Extended Signals (Specs 11 - 14)
    speaker_identities: List[str] = Field(default_factory=list, description="Resolved creator IDs, e.g. ['STREAMER:theburntpeanut', 'CO_STREAMER:hutchmf']")
    chat_intents: Dict[str, float] = Field(default_factory=dict, description="Fine-grained emotion intents from Spec 08")
    active_sponsor_brand: Optional[str] = Field(None, description="Active sponsor detected in this window (Spec 09)")
    griefer_messages_flagged: int = Field(default=0, description="Count of bad-faith contrarian/griefer messages (Spec 12)")
    active_domain: Optional[str] = Field(None, description="Active web domain being browsed, e.g. 'x.com' (Spec 14)")
    read_along_text: Optional[str] = Field(None, description="On-screen text being read aloud by streamer (Spec 14)")


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


# --- Spec 11: Speaker Voiceprint & Co-Stream Network ---

class VoiceprintProfile(BaseModel):
    creator_id: str
    display_name: str
    primary_channel: Optional[str] = None
    centroid_embedding: List[float] = Field(..., description="Normalized speaker d-vector")
    sample_count: int = 1
    confidence_threshold: float = 0.76
    last_updated: str = ""


class SpeakerMatchResult(BaseModel):
    assigned_label: str  # e.g. "STREAMER:theburntpeanut" or "CO_STREAMER:hutchmf"
    creator_id: Optional[str] = None
    confidence: float = 0.0
    is_known_creator: bool = False


class CoStreamInteraction(BaseModel):
    host_creator: str
    guest_creator: str
    vod_id: str
    interaction_start_sec: float
    interaction_end_sec: float
    total_spoken_duration_sec: float
    game_or_activity: str = "UNKNOWN"
    timestamp: str = ""


# --- Spec 12: Chatter Profiling & Safety ---

class ChatterDetailedProfile(BaseModel):
    user_id: str
    username: str
    first_seen_at: str
    last_seen_at: str
    total_messages: int = 0
    contrarian_index: float = 0.0
    hostility_index: float = 0.0
    banter_reciprocity: float = 0.0
    griefer_score: float = 0.0
    flagged_status: str = "CLEAN"  # "CLEAN", "WATCHLIST", "GRIEFER", "BRIGADE"
    is_automated_bot: bool = False


class ChatterSafetyVerdict(BaseModel):
    message_id: str
    user_id: str
    verdict: str  # "GOOD_NATURED_BANTER", "COMMUNITY_ROAST", "BAD_FAITH_GRIEFING", "MALICIOUS_HARASSMENT"
    confidence: float
    reason: str


class BrigadeCluster(BaseModel):
    cluster_id: str
    window_start_sec: float
    window_end_sec: float
    participant_user_ids: List[str]
    similarity_score: float
    flagged_phrase: str


# --- Spec 13: Streamer Knowledge Graph & Claims ---

class StreamerClaim(BaseModel):
    claim_id: str
    creator_id: str
    vod_id: str
    timestamp_sec: float
    end_sec: float
    topic_category: str = "GENERAL"
    subject_entity: str
    predicate: str
    object_value: str
    stance: str  # "APPROVAL", "DISAPPROVAL", "NEUTRAL"
    polarity: float = 0.0  # -1.0 to +1.0
    confidence: float = 1.0
    raw_quote: str
    visual_context_summary: str = ""
    supersedes_claim_id: Optional[str] = None
    is_stance_reversal: bool = False


class EntityStanceRecord(BaseModel):
    creator_id: str
    subject_entity: str
    aggregate_polarity: float
    overall_stance: str
    claims_count: int
    sub_attributes: Dict[str, float] = Field(default_factory=dict)
    last_updated: str = ""


# --- Spec 14: Dense Frame Extraction & Web Grounding ---

class SocialPostCard(BaseModel):
    platform: str = "X_TWITTER"
    author_handle: str
    author_name: str
    post_text: str
    bounding_box: List[float] = Field(default_factory=list)  # [ymin, xmin, ymax, xmax]
    has_embedded_media: bool = False


class ScreenWebContext(BaseModel):
    timestamp_sec: float
    browser_detected: bool = False
    detected_url: Optional[str] = None
    domain: Optional[str] = None
    active_tab_title: Optional[str] = None
    social_post_cards: List[SocialPostCard] = Field(default_factory=list)
    article_headlines: List[str] = Field(default_factory=list)


class ScreenGameContext(BaseModel):
    timestamp_sec: float
    game_title_hint: Optional[str] = None
    minimap_box: Optional[List[float]] = None
    health_percentage: Optional[float] = None
    mana_percentage: Optional[float] = None
    killfeed_entries: List[str] = Field(default_factory=list)
    inventory_open: bool = False


class ReadAlongSegment(BaseModel):
    start_sec: float
    end_sec: float
    spoken_text: str
    matched_screen_text: str
    alignment_score: float
    reading_wpm: float
    source_handle: Optional[str] = None


# --- Spec 17: Self-Expanding Adaptive Slang & Meme Engine ---

class SlangCandidate(BaseModel):
    term: str
    burst_velocity: float
    z_score: float
    total_occurrences: int
    unique_authors: int
    window_start_sec: float
    window_end_sec: float
    co_occurring_intents: Dict[str, float] = Field(default_factory=dict)
    inferred_intent: str = "NEUTRAL"
    inferred_valence: float = 0.0
    acoustic_energy_boost: float = 0.0
    confidence: float = 0.5


class AdaptiveTermEntry(BaseModel):
    term: str
    inferred_intent: str
    valence: float = 0.0
    confidence: float = 0.5
    first_seen_timestamp: str
    last_seen_timestamp: str
    occurrence_count: int = 1
    unique_authors_count: int = 1
    peak_z_score: float = 3.0
    decay_half_life_days: float = 14.0
    status: str = "ACTIVE"


# --- Spec 18: Subprocess Worker Isolation & Bounded Buffering ---

class WorkerTaskInput(BaseModel):
    task_type: str
    media_path: str
    output_path: str
    config: Dict[str, Any] = Field(default_factory=dict)
    timeout_sec: float = 300.0
    trace_id: Optional[str] = None
    extra_payload: Optional[Dict[str, Any]] = None




