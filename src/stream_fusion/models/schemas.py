"""Pydantic data models for StreamFusion data contracts."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid
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
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Platform-specific metadata (superchats, bits, colors)")


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
    creator_id: str = "creator"
    vod_id: str = "vod"
    timestamp_sec: float = 0.0
    end_sec: float = 0.0
    topic_category: str = "GENERAL"
    subject_entity: str = ""
    predicate: str = ""
    object_value: str = ""
    stance: str = "NEUTRAL"  # "APPROVAL", "DISAPPROVAL", "NEUTRAL"
    polarity: float = 0.0  # -1.0 to +1.0
    confidence: float = 1.0
    raw_quote: str = ""
    visual_context_summary: str = ""
    supersedes_claim_id: Optional[str] = None
    is_stance_reversal: bool = False

    # Aliases for convenience across Spec 13, Spec 14, and Spec 20
    subject: Optional[str] = None
    statement: Optional[str] = None
    object: Optional[str] = None

    def model_post_init(self, __context: Any) -> None:
        if self.subject and not self.subject_entity:
            self.subject_entity = self.subject
        elif self.subject_entity and not self.subject:
            self.subject = self.subject_entity

        if self.statement and not self.raw_quote:
            self.raw_quote = self.statement
        elif self.raw_quote and not self.statement:
            self.statement = self.raw_quote

        if self.object and not self.object_value:
            self.object_value = self.object
        elif self.object_value and not self.object:
            self.object = self.object_value


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


# --- Spec 21: Autonomous Multi-Agent Short Production & Auto-Publisher ---

class NarrativeArc(str, Enum):
    HOOK_BUILDUP_PAYOFF = "HOOK_BUILDUP_PAYOFF"
    INSTANT_CLIMAX_REACTION = "INSTANT_CLIMAX_REACTION"
    HOT_TAKE_AND_DEBATE = "HOT_TAKE_AND_DEBATE"
    SPONSOR_SHOWCASE = "SPONSOR_SHOWCASE"


class CropLayout(str, Enum):
    STACKED_CAM_CONTENT = "STACKED_CAM_CONTENT"
    FULL_CONTENT_PAN_SCAN = "FULL_CONTENT_PAN_SCAN"
    CAM_PIP = "CAM_PIP"


class SubtitleStylePreset(str, Enum):
    KARAOKE_POP = "KARAOKE_POP"
    CLEAN_MINIMAL = "CLEAN_MINIMAL"
    MEME_EMOTE = "MEME_EMOTE"


class AuditStatus(str, Enum):
    PASSED = "PASSED"
    FLAGGED = "FLAGGED"
    REJECTED = "REJECTED"


class ShortCandidate(BaseModel):
    candidate_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    start_sec: float
    end_sec: float
    duration_sec: float
    peak_timestamp_sec: float
    highlight_score: float
    chat_burst_zscore: float
    primary_emotion: str = "EXCITEMENT"
    narrative_arc: NarrativeArc = NarrativeArc.HOOK_BUILDUP_PAYOFF
    hook_text: str = ""
    summary: str = ""
    dominant_slang: List[str] = Field(default_factory=list)


class EditorialCutPlan(BaseModel):
    crop_layout: CropLayout = CropLayout.STACKED_CAM_CONTENT
    facecam_box: Optional[Dict[str, float]] = None
    content_box: Optional[Dict[str, float]] = None
    subtitle_preset: SubtitleStylePreset = SubtitleStylePreset.KARAOKE_POP
    highlight_word_color: str = "&H0000FFFF"
    burn_subtitles: bool = True
    audio_duck_music_db: float = -12.0
    hook_duration_sec: float = 3.0


class ContentAuditReport(BaseModel):
    audit_status: AuditStatus = AuditStatus.PASSED
    toxicity_score: float = 0.0
    flagged_terms: List[str] = Field(default_factory=list)
    has_sponsored_content: bool = False
    sponsor_brand_names: List[str] = Field(default_factory=list)
    ftc_disclosure_required: bool = False
    disclosure_tag: Optional[str] = None
    claim_verified: bool = True
    claim_notes: Optional[str] = None


class PlatformCopyBundle(BaseModel):
    youtube_title: str
    youtube_description: str
    youtube_tags: List[str] = Field(default_factory=list)
    tiktok_caption: str
    tiktok_hashtags: List[str] = Field(default_factory=list)
    twitter_thread: List[str] = Field(default_factory=list)
    hook_headline: str


class ViralityScoreCard(BaseModel):
    overall_virality_score: float
    hook_strength: float
    pacing_score: float
    chat_resonance: float
    meme_potential: float
    predicted_completion_rate: float


class ShortProductionPackage(BaseModel):
    package_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    candidate: ShortCandidate
    editorial_plan: EditorialCutPlan
    audit_report: ContentAuditReport
    copy_bundle: PlatformCopyBundle
    virality: ViralityScoreCard
    video_path: Optional[str] = None
    thumbnail_path: Optional[str] = None
    envelope_id: Optional[str] = None


class PublishResult(BaseModel):
    package_id: str
    platform: str
    status: str = "PUBLISHED"
    post_id: Optional[str] = None
    post_url: Optional[str] = None
    published_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    payload_snapshot: Dict[str, Any] = Field(default_factory=dict)


# --- Spec 19: Real-Time Live Ingestion & WebSocket Stream Tailing ---

class LivePlatform(str, Enum):
    TWITCH = "TWITCH"
    KICK = "KICK"
    YOUTUBE_LIVE = "YOUTUBE_LIVE"
    CUSTOM_HLS = "CUSTOM_HLS"


class LiveState(str, Enum):
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    ERROR = "ERROR"


class LiveStreamConfig(BaseModel):
    channel_name: str
    platform: LivePlatform = LivePlatform.TWITCH
    stream_url: Optional[str] = None
    buffer_duration_sec: float = Field(default=120.0, description="Depth of rolling video buffer")
    microbatch_duration_sec: float = Field(default=10.0, description="Duration per micro-batch sliding window")
    ws_port: int = 8765
    sse_port: int = 8766
    anonymous_chat: bool = True
    irc_nick: Optional[str] = None
    irc_oauth: Optional[str] = None
    max_replay_buffer_size: int = 250
    temp_dir: Optional[str] = None
    chatroom_id: Optional[int] = Field(default=None, description="Direct Kick chatroom ID if known")
    youtube_api_key: Optional[str] = Field(default=None, description="Optional YouTube Data API v3 key")
    custom_headers: Dict[str, str] = Field(default_factory=dict)


class LiveTailHealthMetrics(BaseModel):
    fps: float = 0.0
    chat_messages_per_sec: float = 0.0
    buffer_latency_sec: float = 0.0
    dropped_frames: int = 0
    buffered_seconds: float = 0.0
    memory_mb: float = 0.0


class LiveStreamStatus(BaseModel):
    stream_id: str
    state: LiveState = LiveState.STOPPED
    channel_name: str
    platform: LivePlatform
    uptime_sec: float = 0.0
    total_bytes_ingested: int = 0
    total_chat_messages: int = 0
    active_subscribers: int = 0
    health: LiveTailHealthMetrics = Field(default_factory=LiveTailHealthMetrics)
    last_error: Optional[str] = None
    started_at: Optional[datetime] = None


class LiveClientSubscription(BaseModel):
    client_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    transport: str = "WEBSOCKET"  # or "SSE"
    event_types: List[str] = Field(default_factory=lambda: ["*"])
    connected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    replay_count: int = 0


# --- Spec 20: Web Grounding & Live Knowledge Graph Expansion ---

class StancePolarity(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    MIXED = "MIXED"


class FactCheckVerdict(str, Enum):
    VERIFIED_TRUE = "VERIFIED_TRUE"
    CONTRADICTED = "CONTRADICTED"
    UNSUBSTANTIATED = "UNSUBSTANTIATED"
    OUTDATED = "OUTDATED"
    UNVERIFIABLE = "UNVERIFIABLE"


class WebGroundingCitation(BaseModel):
    url: str
    domain: str
    title: str
    snippet: str
    published_date: Optional[str] = None
    confidence_score: float = Field(default=0.8, ge=0.0, le=1.0)


class GroundedClaimResult(BaseModel):
    claim_id: str
    verdict: FactCheckVerdict
    search_query: str
    citations: List[WebGroundingCitation] = Field(default_factory=list)
    explanation: str
    verified_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    original_claim_statement: Optional[str] = None


class StanceShiftRecord(BaseModel):
    shift_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    entity_name: str
    previous_stance: str  # e.g. POSITIVE, NEGATIVE, NEUTRAL
    new_stance: str
    shift_delta: float = 0.0  # -2.0 to +2.0
    previous_stream_id: str
    new_stream_id: str
    evidence_quote_before: str
    evidence_quote_after: str
    is_reversal: bool = False
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EntityOpinionSynthesis(BaseModel):
    entity_name: str
    overall_consensus_stance: str
    total_claims_count: int = 0
    contradiction_count: int = 0
    volatility_score: float = Field(default=0.0, ge=0.0, le=1.0)
    stance_timeline: List[Dict[str, Any]] = Field(default_factory=list)
    contradictions: List[Dict[str, Any]] = Field(default_factory=list)
    summary: str
    synthesized_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# --- Spec 22: Multi-Platform Live Stream & Chat Connectors ---

class ConnectorState(str, Enum):
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    RECONNECTING = "RECONNECTING"
    ERROR = "ERROR"


class PlatformCapabilities(BaseModel):
    platform: LivePlatform
    supports_emotes: bool = True
    supports_badges: bool = True
    supports_superchats: bool = False
    supports_bits: bool = False
    supports_subscriptions: bool = True
    requires_api_key: bool = False
    supports_anonymous: bool = True


class MonetizationEvent(BaseModel):
    event_type: str = Field(..., description="'SUPER_CHAT', 'BITS', 'GIFT_SUB', 'MEMBERSHIP'")
    amount: float = 0.0
    currency: str = "USD"
    raw_text: Optional[str] = None
    tier: Optional[str] = None
    sender_name: Optional[str] = None








