"""Configuration management for StreamFusion."""

from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field


class AudioConfig(BaseModel):
    whisper_model: str = "large-v3-turbo"
    compute_type: str = "float16"  # or int8
    device: str = "cuda"
    diarization_enabled: bool = True
    hf_token: Optional[str] = None


class VisionConfig(BaseModel):
    vlm_model: str = "microsoft/Florence-2-base"
    device: str = "cuda"
    sample_interval_sec: float = 2.0
    ocr_enabled: bool = True
    object_detection_enabled: bool = True


class ChatConfig(BaseModel):
    latency_offset_sec: float = Field(
        default=4.5,
        description="Broadcast delay offset: T_event = T_chat - latency_offset_sec"
    )
    auto_calibrate_latency: bool = Field(
        default=True,
        description="Whether to automatically calibrate latency offset via cross-correlation"
    )
    bucket_window_sec: float = Field(
        default=2.0,
        description="Time bucket aggregation slice duration"
    )


class StorageConfig(BaseModel):
    backend: str = "local"  # local, smb, or s3
    cache_dir: Path = Path("./cache")
    output_dir: Path = Path("./output")


class ExecutionConfig(BaseModel):
    isolate_gpu_workers: bool = Field(
        default=False,
        description="Whether to run Whisper and Florence in isolated child processes to reclaim VRAM"
    )
    bounded_buffering: bool = Field(
        default=False,
        description="Whether to use windowed frame extraction with immediate image purging"
    )
    window_size_sec: float = Field(
        default=30.0,
        description="Window size in seconds for bounded keyframe buffering"
    )
    worker_timeout_sec: float = Field(
        default=600.0,
        description="Timeout in seconds for isolated worker subprocesses"
    )


class StreamFusionConfig(BaseModel):
    audio: AudioConfig = Field(default_factory=AudioConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    chat: ChatConfig = Field(default_factory=ChatConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    chunk_duration_sec: Optional[float] = Field(
        default=None,
        description="Optional duration in seconds to process stream in chunks (e.g. 1800 for 30m chunks)"
    )

