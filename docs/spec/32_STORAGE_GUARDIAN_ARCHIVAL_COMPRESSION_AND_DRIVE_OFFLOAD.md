# SPEC 32: Storage Guardian, Archival Video Compression & Cloud Drive Offloader

## 1. Overview & Objectives

High-resolution live stream ingestion presents severe storage bottlenecks. A single 9.5-hour broadcast at 1080p60 AVC1 consumes over **32 GB** of disk space. Ingesting multiple broadcasts without intelligent lifecycle management will rapidly exhaust local and homelab drives, triggering kernel I/O panics and pipeline aborts.

**Spec 32** implements a comprehensive, three-tiered **Storage Lifecycle & Protection Architecture**:
1. **Storage Guardian & Budgeting Engine (`StorageGuardian`)**:
   - Enforces a configurable global storage budget (`STORAGE_BUDGET_GB`) and an absolute filesystem free space safety floor (`MIN_FREE_DISK_GB`).
   - Executes pre-flight space reservation checks prior to any download or transcode operations.
   - Monitors active storage tiers and issues multi-stage telemetry alerts (Warning @ 80%, Critical @ 90%, Hard Halt @ 95%).
   - Configurable overflow handling policies: `HALT`, `OFFLOAD_GDRIVE`, or `PRUNE_OLDEST_PROXIES`.
2. **Ephemeral Intermediate Asset Purging**:
   - Intermediate 16 kHz WAV audio tracks and extracted JPEG frame batches are strictly temporary and are automatically deleted immediately after speech transcription, vision, and OCR inference stages finish.
3. **Ultra-Compressed Full VOD Archival Proxy (`VideoArchiver`)**:
   - Transcodes the completed high-resolution broadcast (e.g. 32 GB 1080p60) into an ultra-compact reference proxy:
     - Resolution: 480p (854×480) or 360p (640×360) @ 24/30 FPS.
     - Codec: H.264 (universal HTML5 browser scrubbing in StreamFusion Studio UI) with AAC audio.
     - Target bitrate: ~350–400 kbps total, achieving an **over 95% space reduction (~1.5–1.8 GB per 9.5h broadcast)**.
     - Web-optimized faststart MP4 container (`-movflags +faststart`).
   - Safely replaces the 32 GB uncompressed source with the archival proxy once verification succeeds.
4. **Google Drive Cloud Offload Driver (`GoogleDriveOffloader`)**:
   - Provides automated chunked/resumable upload of finalized proxies and shorts packages to designated Google Drive folders.
   - Supports checksum verification and post-upload local proxy purging to maintain near-zero disk usage on homelab nodes.

---

## 2. Technical Architecture & Components

### 2.1 Configuration Schema Additions (`src/stream_fusion/models/schemas.py` & `src/stream_fusion/config.py`)

```python
class StorageOverflowPolicy(str, Enum):
    HALT = "halt"
    OFFLOAD_GDRIVE = "offload_gdrive"
    PRUNE_OLDEST = "prune_oldest"

class StorageGuardianConfig(BaseModel):
    storage_budget_gb: float = 1500.0
    min_free_disk_gb: float = 50.0
    warning_threshold: float = 0.80
    critical_threshold: float = 0.90
    overflow_policy: StorageOverflowPolicy = StorageOverflowPolicy.HALT
    enable_gdrive_offload: bool = False
    gdrive_root_folder_id: Optional[str] = None
    gdrive_credentials_path: Optional[str] = None
    auto_purge_local_after_cloud_sync: bool = False

class ArchivalCompressionConfig(BaseModel):
    enabled: bool = True
    target_height: int = 480
    target_fps: int = 24
    video_bitrate_kbps: int = 350
    audio_bitrate_kbps: int = 64
    codec: str = "libx264"
    preset: str = "fast"
    crf: int = 28
    replace_source_after_transcode: bool = True
```

### 2.2 Storage Guardian Engine (`src/stream_fusion/storage/guardian.py`)

Implements `StorageGuardian`:
- `get_storage_report(path: Path) -> StorageReport`:
  Calculates total storage used across StreamFusion directories, free disk space, percent of budget consumed, and threshold alert levels.
- `assert_can_allocate(required_gb: float, path: Path) -> bool`:
  Checks if adding `required_gb` violates `storage_budget_gb` or reduces system disk free space below `min_free_disk_gb`. Raises `InsufficientStorageError` or executes `overflow_policy`.
- `prune_oldest_proxies(target_bytes_to_free: int) -> List[Path]`:
  Safely removes oldest archival proxies while preserving 100% of database records (PostgreSQL, ClickHouse, Qdrant).

### 2.3 Video Archiver Transcoder (`src/stream_fusion/storage/archiver.py`)

Implements `VideoArchiver`:
- `transcode_to_reference_proxy(source_video: Path, output_proxy: Path) -> ArchivalTranscodeResult`:
  Executes optimized FFmpeg command:
  ```bash
  ffmpeg -y -i <source> -vf "scale=-2:480,fps=24" -c:v libx264 -crf 28 -preset fast -b:v 350k -maxrate 500k -bufsize 1000k -c:a aac -b:a 64k -movflags +faststart <output_proxy>
  ```
- Validates the resulting proxy with `ffprobe` (ensures non-zero duration, valid video and audio streams).
- Safely unlinks the source file when `replace_source_after_transcode=True` and updates database asset paths.

### 2.4 Google Drive Offloader (`src/stream_fusion/storage/gdrive.py`)

Implements `GoogleDriveOffloader`:
- Resumable chunked upload protocol using Google Drive API or authenticated service account.
- Computes MD5 checksum verification before signaling safe deletion of local proxy.
- Graceful degradation if Google Drive credentials are not yet configured.

---

## 3. Verification Criteria

1. **Unit Tests (`tests/test_storage_guardian_and_archiver.py`)**:
   - Verify `StorageGuardian` accurately calculates free space and budget usage.
   - Verify `assert_can_allocate` blocks allocations exceeding the budget floor.
   - Verify `VideoArchiver` constructs correct FFmpeg parameters with faststart indexing and bitrate constraints.
   - Verify ephemeral cleanup hooks delete temporary WAV and JPEG directories.
2. **Integration Verification**:
   - Verify mock transcode replaces source video and updates manifest records.
   - All tests pass with zero regressions on existing 233+ test suite.
