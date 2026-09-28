"""Worker IPC protocol and task data contracts (Spec 18)."""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class WorkerTaskType(str, Enum):
    """Supported isolated worker task types."""
    AUDIO_TRANSCRIPTION = "AUDIO_TRANSCRIPTION"
    VOICE_DIARIZATION = "VOICE_DIARIZATION"
    VISION_OCR = "VISION_OCR"
    DENSE_VISION = "DENSE_VISION"


class WorkerTaskInput(BaseModel):
    """Input payload dispatched to an isolated subprocess worker."""
    task_type: WorkerTaskType
    media_path: str
    output_path: str
    config: Dict[str, Any] = Field(default_factory=dict)
    timeout_sec: float = 300.0
    trace_id: Optional[str] = None
    extra_payload: Optional[Dict[str, Any]] = None
