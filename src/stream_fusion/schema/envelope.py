"""Universal Message Bus Envelope & Stream Event Types (Spec 16)."""

from datetime import datetime, timezone
from enum import Enum
import json
from typing import Any, Dict, Generic, List, Optional, TypeVar
import uuid
from pydantic import BaseModel, Field


class StreamEventType(str, Enum):
    """Standardized event types across the StreamFusion pipeline and agent bus."""
    STAGE_START = "STAGE_START"
    STAGE_COMPLETE = "STAGE_COMPLETE"
    AUDIO_SEGMENT = "AUDIO_SEGMENT"
    KEYFRAME_ANALYSIS = "KEYFRAME_ANALYSIS"
    FUSION_SLICE = "FUSION_SLICE"
    CHAT_BURST = "CHAT_BURST"
    HIGHLIGHT_MOMENT = "HIGHLIGHT_MOMENT"
    SPONSOR_DETECTED = "SPONSOR_DETECTED"
    CLAIM_EXTRACTED = "CLAIM_EXTRACTED"
    SAFETY_ALERT = "SAFETY_ALERT"
    WEB_READ_ALONG = "WEB_READ_ALONG"
    SLANG_CANDIDATE_DETECTED = "SLANG_CANDIDATE_DETECTED"
    LEXICON_UPDATED = "LEXICON_UPDATED"
    AUDIT_BENCHMARK = "AUDIT_BENCHMARK"
    SHORT_PRODUCED = "SHORT_PRODUCED"
    SHORT_PUBLISHED = "SHORT_PUBLISHED"
    LIVE_STREAM_STARTED = "LIVE_STREAM_STARTED"
    LIVE_STREAM_STOPPED = "LIVE_STREAM_STOPPED"
    CHAT_MESSAGE = "CHAT_MESSAGE"
    MEDIA_SEGMENT_BUFFERED = "MEDIA_SEGMENT_BUFFERED"
    PIPELINE_COMPLETE = "PIPELINE_COMPLETE"
    COSTREAM_SESSION_STARTED = "COSTREAM_SESSION_STARTED"
    COSTREAM_SESSION_STOPPED = "COSTREAM_SESSION_STOPPED"
    COSTREAM_MESSAGE = "COSTREAM_MESSAGE"
    COSTREAM_BURST = "COSTREAM_BURST"
    COSTREAM_DIVERGENCE_ALERT = "COSTREAM_DIVERGENCE_ALERT"
    ERROR = "ERROR"


class EnvelopeTelemetry(BaseModel):
    """Lightweight stage and resource telemetry embedded in envelopes."""
    duration_ms: Optional[float] = None
    stage: Optional[str] = None
    ram_mb: Optional[float] = None
    vram_mb: Optional[float] = None


T = TypeVar("T")


class StreamFusionEnvelope(BaseModel, Generic[T]):
    """Standardized JSON envelope for all inter-module and inter-agent communication."""
    version: str = "1.0"
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    stream_id: str
    event_type: StreamEventType
    trace_id: Optional[str] = None
    parent_id: Optional[str] = None
    producer: Optional[str] = None
    payload: Any = None
    telemetry: Optional[EnvelopeTelemetry] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serializes the envelope to a valid JSON string."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "StreamFusionEnvelope":
        """Deserializes a JSON string into a StreamFusionEnvelope."""
        return cls.model_validate_json(json_str)

    @classmethod
    def create(
        cls,
        stream_id: str,
        event_type: StreamEventType,
        payload: Any,
        producer: Optional[str] = None,
        trace_id: Optional[str] = None,
        telemetry: Optional[EnvelopeTelemetry] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "StreamFusionEnvelope":
        """Convenience factory method to instantiate a standardized envelope."""
        # If payload is a Pydantic model, convert to dict for universal serializability
        p_data = payload.model_dump() if hasattr(payload, "model_dump") else payload
        return cls(
            stream_id=stream_id,
            event_type=event_type,
            payload=p_data,
            producer=producer,
            trace_id=trace_id,
            telemetry=telemetry,
            metadata=metadata or {},
        )
