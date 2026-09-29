import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
import yaml


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

    # Adaptive Density Optimizer (Spec 30)
    sampling_mode: str = "fixed"  # "fixed" | "adaptive"
    min_interval_sec: float = 0.5  # High-density burst interval (2 FPS)
    max_interval_sec: float = 5.0  # Low-density idle interval (0.2 FPS)
    burst_window_sec: float = 12.0  # Duration to maintain dense sampling after trigger
    chat_burst_zscore_threshold: float = 2.5
    scene_change_threshold: float = 0.35


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


class HomelabConfig(BaseModel):
    host: str = Field(
        default="cbwdellr720",
        description="Homelab hostname (e.g. 'cbwdellr720'), ZeroTier IP, or local LAN IP",
    )
    port: int = Field(default=5432, description="PostgreSQL or service port on homelab")
    database_url: Optional[str] = Field(
        default=None,
        description="Full database connection string (e.g. 'postgresql://user:pass@cbwdellr720:5432/streamfusion'). If None, falls back to SQLite.",
    )
    storage_root: str = Field(
        default="\\\\cbwdellr720\\streams",
        description="Root network share or local mount for VOD storage (e.g. '\\\\cbwdellr720\\streams' or '/mnt/homelab/streams')",
    )
    vods_subdir: str = Field(default="vods", description="Subdirectory for VOD assets")
    min_free_disk_gb: float = Field(
        default=50.0,
        description="Minimum free space required on storage destination before pausing downloads",
    )


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
    homelab: HomelabConfig = Field(default_factory=HomelabConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    chunk_duration_sec: Optional[float] = Field(
        default=None,
        description="Optional duration in seconds to process stream in chunks (e.g. 1800 for 30m chunks)"
    )


def load_config(config_path: Optional[Path] = None) -> StreamFusionConfig:
    """Loads configuration from a YAML file, environment variable, or default."""
    target_path = config_path
    if not target_path:
        env_path = os.getenv("STREAMFUSION_CONFIG")
        if env_path:
            target_path = Path(env_path)
        elif Path("streamfusion.yaml").exists():
            target_path = Path("streamfusion.yaml")
        elif Path("config.yaml").exists():
            target_path = Path("config.yaml")

    if target_path and target_path.exists():
        with open(target_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            cfg = StreamFusionConfig.model_validate(data)
    else:
        cfg = StreamFusionConfig()

    # Environment variable overrides
    if "STREAMFUSION_HOMELAB_HOST" in os.environ:
        cfg.homelab.host = os.environ["STREAMFUSION_HOMELAB_HOST"]
    if "STREAMFUSION_DATABASE_URL" in os.environ:
        cfg.homelab.database_url = os.environ["STREAMFUSION_DATABASE_URL"]
    if "STREAMFUSION_STORAGE_ROOT" in os.environ:
        cfg.homelab.storage_root = os.environ["STREAMFUSION_STORAGE_ROOT"]

    return cfg


def save_config(config: StreamFusionConfig, config_path: Path) -> None:
    """Saves a StreamFusionConfig instance to a YAML file."""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config.model_dump(mode="json"), f, default_flow_style=False, sort_keys=False)


