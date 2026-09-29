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
    total_chat_messages: int = 0
    slices: List[FusionSlice] = Field(default_factory=list)
    highlights: List[Dict[str, object]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def fusion_slices(self) -> List[FusionSlice]:
        return self.slices


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


# --- Spec 23: Multi-Stream Co-Stream & Cross-Platform Alignment ---

class CoStreamChannelConfig(BaseModel):
    channel_id: str
    stream_config: LiveStreamConfig
    creator_name: Optional[str] = None
    is_reference_stream: bool = False
    manual_latency_offset: Optional[float] = None
    max_buffer_size_mb: int = 250
    voiceprint_profile_id: Optional[str] = None


class CoStreamSessionConfig(BaseModel):
    session_id: str = Field(default_factory=lambda: f"costream-{uuid.uuid4().hex[:8]}")
    session_title: str = "Live Co-Stream Event"
    channels: List[CoStreamChannelConfig] = Field(default_factory=list)
    reference_channel_id: Optional[str] = None
    auto_sync: bool = True
    sync_interval_sec: float = 30.0
    max_session_memory_mb: int = 1024
    bucket_window_sec: float = 2.0


class CoStreamChannelTelemetry(BaseModel):
    channel_id: str
    platform: LivePlatform
    state: ConnectorState = ConnectorState.STOPPED
    total_messages_received: int = 0
    current_message_velocity: float = 0.0
    calibrated_latency_offset: float = 0.0
    sync_confidence: float = 1.0
    reconnect_attempts: int = 0
    last_error: Optional[str] = None


class CoStreamSessionStatus(BaseModel):
    session_id: str
    is_active: bool = False
    uptime_sec: float = 0.0
    channels: Dict[str, CoStreamChannelTelemetry] = Field(default_factory=dict)
    reference_channel_id: Optional[str] = None
    cross_platform_agreement_index: float = 1.0
    total_messages: int = 0


class CrossStreamSyncResult(BaseModel):
    reference_channel_id: str
    channel_offsets: Dict[str, float] = Field(default_factory=dict)
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
    sync_method: str = "CROSS_CORRELATION"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CrossAudienceSentimentPoint(BaseModel):
    timestamp_sec: float
    window_sec: float = 2.0
    platform_sentiments: Dict[str, float] = Field(default_factory=dict)
    channel_sentiments: Dict[str, float] = Field(default_factory=dict)
    cross_platform_agreement: float = 1.0
    dominant_emotes: Dict[str, List[str]] = Field(default_factory=dict)
    divergence_detected: bool = False
    divergence_description: Optional[str] = None


class CrossStreamBurstPropagation(BaseModel):
    token: str
    origin_platform: str
    origin_channel: str
    origin_timestamp: float
    cascade_timeline: Dict[str, float] = Field(default_factory=dict)
    cascade_velocity_sec: float = 0.0


class CoStreamDebateTurn(BaseModel):
    turn_id: str = Field(default_factory=lambda: f"turn-{uuid.uuid4().hex[:6]}")
    speaker_name: str
    channel_id: str
    start_sec: float
    end_sec: float
    duration_sec: float
    transcript: str
    sentiment_score: float = 0.0
    interrupts_previous: bool = False


class MultiAngleShortCandidate(BaseModel):
    candidate_id: str = Field(default_factory=lambda: f"mashort-{uuid.uuid4().hex[:8]}")
    start_sec: float
    end_sec: float
    duration_sec: float
    participating_channels: List[str] = Field(default_factory=list)
    layout_preset: str = Field(default="STACKED_SPLIT", description="'STACKED_SPLIT', 'SIDE_BY_SIDE', 'PICTURE_IN_PICTURE', 'QUAD_GRID'")
    cross_platform_agreement: float = 1.0
    virality_score: float = 0.0
    title: str = ""
    hooks: List[str] = Field(default_factory=list)


# --- Spec 24: Full-Spectrum Pipeline Unification & Master Synergy Orchestrator ---

class StageExecutionStatus(str, Enum):
    SUCCESS = "SUCCESS"
    SKIPPED = "SKIPPED"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"


class FullSpectrumStageSummary(BaseModel):
    stage_name: str
    status: StageExecutionStatus = StageExecutionStatus.SUCCESS
    duration_sec: float = 0.0
    output_summary: str = ""
    error_message: Optional[str] = None


class FullSpectrumConfig(BaseModel):
    isolate_gpu_workers: bool = False
    bounded_buffer: bool = True
    window_size_sec: float = 30.0
    sample_interval_sec: float = 2.0
    enable_adaptive_slang: bool = True
    enable_web_grounding: bool = True
    enable_stance_tracking: bool = True
    enable_sponsor_quantifier: bool = True
    enable_short_production: bool = True
    short_candidate_count: int = 3
    dry_run_shorts: bool = False
    auto_latency: bool = True
    manual_latency_offset: Optional[float] = None
    stance_history_db: str = "./stance_history.json"
    adaptive_lexicon_db: str = "./adaptive_lexicon.json"


class FullSpectrumManifest(BaseModel):
    manifest_id: str = Field(default_factory=lambda: f"fsm-{uuid.uuid4().hex[:8]}")
    stream_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_pipeline_duration_sec: float = 0.0
    effective_media_duration_sec: float = 0.0
    stages: Dict[str, FullSpectrumStageSummary] = Field(default_factory=dict)
    total_audio_segments: int = 0
    total_keyframes: int = 0
    total_chat_messages: int = 0
    total_fusion_slices: int = 0
    calibrated_broadcast_delay_sec: float = 0.0
    take_agreement_mean: float = 0.0
    slang_terms_updated: int = 0
    claims_extracted_count: int = 0
    grounded_claims_count: int = 0
    stance_shifts_count: int = 0
    sponsor_mentions_count: int = 0
    shorts_produced_count: int = 0
    output_directory: str = ""
    html_report_path: Optional[str] = None


# --- Spec 25: Targeted Streamer Roster Ingestion & Homelab Harvester Pipeline ---

class HarvestStatus(str, Enum):
    DISCOVERED = "DISCOVERED"
    QUEUED = "QUEUED"
    DOWNLOADING = "DOWNLOADING"
    HARVESTED = "HARVESTED"
    READY_FOR_ANALYSIS = "READY_FOR_ANALYSIS"
    ANALYZING = "ANALYZING"
    ANALYZED = "ANALYZED"
    ERROR = "ERROR"
    FAILED = "FAILED"


class StreamerTargetRecord(BaseModel):
    streamer_id: str = Field(..., description="Normalized slug (e.g. 'asmongold')")
    display_name: str = Field(..., description="Human-readable creator name (e.g. 'Asmongold')")
    channel_urls: List[str] = Field(default_factory=list, description="Platform channel URLs")
    primary_platform: str = Field(default="TWITCH", description="'TWITCH', 'YOUTUBE', 'KICK'")
    quality_preset: str = Field(default="best", description="'best', '1080p', '720p', 'audio_only'")
    include_chat: bool = Field(default=True, description="Whether to fetch chat replay")
    max_recent_vods: int = Field(default=5, description="Max recent VODs to crawl per sync")
    lookback_days: int = Field(default=14, description="How far back to search in days")
    download_priority: int = Field(default=5, description="1 (lowest) to 10 (highest)")
    destination_override: Optional[str] = Field(None, description="Custom storage subpath")
    tags: List[str] = Field(default_factory=list, description="Categorization tags")
    enabled: bool = Field(default=True, description="Whether actively crawled and downloaded")
    created_at: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_synced_at: Optional[datetime] = None
    voiceprint_embedding: Optional[List[float]] = Field(None, description="Pre-seeded voiceprint acoustic embedding")


class HarvestedVodRecord(BaseModel):
    vod_id: str = Field(..., description="Unique VOD identifier, e.g. 'twitch_v123456789'")
    streamer_id: str = Field(..., description="Foreign key to streamer_targets.streamer_id")
    platform: str = Field(..., description="'TWITCH', 'YOUTUBE', 'KICK'")
    title: str = Field(default="", description="Stream or VOD title")
    published_at: Optional[datetime] = None
    duration_sec: float = Field(default=0.0, description="Duration in seconds")
    status: HarvestStatus = Field(default=HarvestStatus.DISCOVERED)
    video_path: Optional[str] = Field(None, description="Local path to downloaded video/audio")
    chat_path: Optional[str] = Field(None, description="Local path to downloaded chat JSON")
    metadata_path: Optional[str] = Field(None, description="Local path to metadata JSON")
    thumbnail_path: Optional[str] = Field(None, description="Local path to preview graphic")
    file_size_bytes: int = Field(default=0, description="Total size in bytes on disk")
    download_speed_mbps: float = Field(default=0.0, description="Recorded download throughput")
    retry_count: int = Field(default=0, description="Number of retry attempts")
    error_message: Optional[str] = Field(None, description="Last recorded error if failed")
    harvested_at: Optional[datetime] = None
    analyzed_at: Optional[datetime] = None
    raw_metadata: Dict[str, Any] = Field(default_factory=dict, description="Original platform metadata")


class HarvesterStatusReport(BaseModel):
    active_workers: int = 0
    max_workers: int = 3
    queue_depth: int = 0
    downloading_count: int = 0
    harvested_count: int = 0
    error_count: int = 0
    free_disk_gb: float = 0.0
    active_downloads: List[Dict[str, Any]] = Field(default_factory=list)


# --- Spec 26: Homelab 24/7 Scheduler Daemon & Service Orchestration ---

class DaemonState(str, Enum):
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    DRAINING = "DRAINING"
    ERROR = "ERROR"


class DaemonTaskType(str, Enum):
    CRAWL = "CRAWL"
    DOWNLOAD = "DOWNLOAD"
    PIPELINE = "PIPELINE"
    HOUSEKEEPING = "HOUSEKEEPING"


class DaemonConfig(BaseModel):
    crawl_interval_minutes: int = Field(default=30, description="Interval between roster channel crawls in minutes")
    download_poll_interval_seconds: int = Field(default=15, description="Interval to check and dispatch queued downloads in seconds")
    auto_analyze: bool = Field(default=True, description="Automatically trigger FullSpectrumPipeline upon harvest")
    max_concurrent_downloads: int = Field(default=2, description="Max concurrent download workers")
    max_concurrent_pipelines: int = Field(default=1, description="Max concurrent pipeline analysis processes (1 to prevent GPU/CPU thrashing)")
    min_free_disk_gb: float = Field(default=50.0, description="Minimum free disk space threshold in GB")
    retention_days: Optional[int] = Field(None, description="Optional raw video retention in days before pruning media.mp4")
    pid_file: str = Field(default="daemon.pid", description="Lockfile path tracking daemon process ID")
    log_file: str = Field(default="logs/daemon.log", description="Path to daemon log file")
    homelab_root: str = Field(default="./homelab_storage", description="Root homelab storage directory")
    catalog_db_url: str = Field(default="sqlite:///homelab_storage/catalog.db", description="Database connection URL")
    enable_shorts: bool = Field(default=True, description="Produce 9:16 vertical shorts in pipeline")
    enable_web_grounding: bool = Field(default=True, description="Ground claims in pipeline")
    enable_adaptive_slang: bool = Field(default=True, description="Update slang lexicon in pipeline")
    enable_sponsor_quantifier: bool = Field(default=True, description="Audit sponsors in pipeline")
    dry_run_shorts: bool = Field(default=False, description="Stage shorts without full video rendering")


class DaemonJobRecord(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_type: DaemonTaskType
    target_id: Optional[str] = None  # e.g. vod_id or streamer_id
    status: str = "PENDING"  # PENDING, RUNNING, COMPLETED, FAILED
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    duration_sec: float = 0.0
    error_message: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class DaemonStatusReport(BaseModel):
    state: DaemonState = DaemonState.STOPPED
    pid: Optional[int] = None
    uptime_seconds: float = 0.0
    started_at: Optional[datetime] = None
    last_crawl_at: Optional[datetime] = None
    next_crawl_at: Optional[datetime] = None
    active_downloads: List[Dict[str, Any]] = Field(default_factory=list)
    active_pipeline_vod: Optional[str] = None
    queued_vods_count: int = 0
    downloading_vods_count: int = 0
    harvested_vods_count: int = 0
    analyzed_vods_count: int = 0
    error_vods_count: int = 0
    free_disk_gb: float = 0.0
    recent_errors: List[str] = Field(default_factory=list)
    recent_jobs: List[DaemonJobRecord] = Field(default_factory=list)


# --- Spec 27: Unified Web Dashboard & Real-Time Studio Schemas ---

class DashboardConfig(BaseModel):
    host: str = Field(default="0.0.0.0", description="Host address to bind dashboard server")
    port: int = Field(default=8000, description="HTTP port to bind dashboard server")
    reload: bool = Field(default=False, description="Enable auto-reload for development")
    homelab_root: str = Field(default="./homelab_storage", description="Root storage directory for VODs and artifacts")
    catalog_db_url: str = Field(default="sqlite:///homelab_storage/catalog.db", description="Catalog database URL")
    enable_cors: bool = Field(default=True, description="Enable Cross-Origin Resource Sharing")
    cors_origins: List[str] = Field(default_factory=lambda: ["*"], description="Allowed CORS origin list")
    static_dir: Optional[str] = Field(default=None, description="Custom path to static frontend assets")
    enable_auth: bool = Field(default=False, description="Enable basic API key header authentication")
    api_key: Optional[str] = Field(default=None, description="API key when authentication is enabled")


class DashboardOverviewStats(BaseModel):
    streamer_count: int = 0
    total_vods_count: int = 0
    harvested_vods_count: int = 0
    analyzed_vods_count: int = 0
    queued_vods_count: int = 0
    downloading_vods_count: int = 0
    total_storage_bytes: int = 0
    free_storage_gb: float = 0.0
    daemon_state: str = "STOPPED"
    daemon_pid: Optional[int] = None
    active_live_streams: int = 0
    total_shorts_count: int = 0
    total_claims_count: int = 0
    last_updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ScrubberTimelinePayload(BaseModel):
    vod_id: str
    title: str = ""
    streamer_id: str = ""
    duration_sec: float = 0.0
    video_url: Optional[str] = None
    slices_count: int = 0
    scenes: List[Dict[str, Any]] = Field(default_factory=list)
    speech_segments: List[Dict[str, Any]] = Field(default_factory=list)
    chat_bursts: List[Dict[str, Any]] = Field(default_factory=list)
    sponsors: List[Dict[str, Any]] = Field(default_factory=list)
    claims: List[Dict[str, Any]] = Field(default_factory=list)
    waveform: List[float] = Field(default_factory=list)
    keyframes: List[Dict[str, Any]] = Field(default_factory=list)


class ShortStudioExportRequest(BaseModel):
    candidate_id: str
    vertical_width: int = 1080
    vertical_height: int = 1920
    subtitle_style: str = "DYNAMIC_WORD_HIGHLIGHT"
    reaction_layout: str = "SPLIT_CAM_GAME"
    render_full_video: bool = True
    auto_publish: bool = False
    target_platforms: List[str] = Field(default_factory=lambda: ["youtube_shorts", "tiktok"])
    export_dir: Optional[str] = None


class LiveTailSubscriptionRequest(BaseModel):
    stream_id: str
    channel_name: str
    platform: str = "twitch"
    include_chat: bool = True
    include_bursts: bool = True
    include_telemetry: bool = True
    replay_count: int = 50


class LiveStreamActionResponse(BaseModel):
    status: str = "OK"
    action: str
    target_id: Optional[str] = None
    message: str = ""
    details: Dict[str, Any] = Field(default_factory=dict)













