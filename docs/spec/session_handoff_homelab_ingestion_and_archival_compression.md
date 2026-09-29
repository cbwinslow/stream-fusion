# StreamFusion Session Handoff: VOD Finalization & Archival Compression Architecture

**Date**: 2026-09-29  
**GitHub Repository**: [`cbwinslow/stream-fusion`](https://github.com/cbwinslow/stream-fusion) (branch `main`)  
**Status**: Zackrawrr VOD `2886498935` 100% Downloaded & Muxed. Ready for Pipeline & Archival Compression.  
**Test Suite**: **233 / 233 tests passing (100% green)**  

---

## 1. Current State & Completed Milestones

### Homelab Storage & Files (`cbwdellr720` @ `192.168.10.1`)
Directory: `/home/cbwinslow/workspace/streamfusion/vods/zackrawrr/2886498935/`
- **`media.mp4`**: **100% COMPLETE & FINALIZED** (32 GB, 9h 39m 52s, 1080p60 AVC1 with faststart moov indexing).
- **`chat.json`**: **100% COMPLETE & FINALIZED** (106 MB full chat history, downloaded via `TwitchDownloaderCLI`).
- **`media.info.json`**: Stream metadata & Twitch chapters.
- **Active Processes**: None (all downloads and FFmpeg remuxing jobs finished cleanly).

### Bare-Metal Database Stack
- **PostgreSQL 17.11**: Running bare-metal on port `5434` (cluster `17-main`), database `streamfusion`. VOD `2886498935` registered in catalog state machine.
- **ClickHouse 25.8**: Running bare-metal on ports `9000` / `8123`, database `streamfusion.chat_events`.
- **Qdrant**: Dedicated vector database on port `6333`, collection `streamfusion_moments`.
- **Storage Subsystem**: Dedicated partition with 1.8 TB free space on server, 0 MB consumed on local workstation `C:\`.

---

## 2. Storage Lifecycle & Video Compression Policy

### The Challenge
Twitch deletes VODs after 14–60 days. If the video file is deleted, the source footage is permanently lost. However, retaining uncompressed ~33 GB files per stream is unsustainable across multiple broadcasts.

### The Decided Solution
1. **High-Quality Inference**:
   - Process the full 1080p video and audio during pipeline execution (Whisper speech transcription, Florence-2 visual scene analysis, OCR, chat calibration).
2. **Ephemeral Asset Purging**:
   - Intermediate 16 kHz WAV audio tracks and extracted JPEG keyframes are strictly temporary and are deleted immediately after their respective inference models finish.
3. **Full Video Archival Compression**:
   - Once pipeline analysis is complete, an automated workflow compresses the full 9.5-hour video into an archival proxy (e.g. 720p or 480p using AV1 / H.265 at ~700–900 kbps with original AAC audio copied untouched).
   - This shrinks the 32 GB file down to **~3.5 GB (90% space reduction)** while keeping 100% of the broadcast completely intact, watchable, and scrubbable in the StreamFusion Web Studio for permanent homelab archiving.
   - The original 32 GB source file is safely replaced by the compressed archival copy.

---

## 3. Plan for Next Session (Spec 31 & Full Pipeline Run)

1. **Spec 31: Automated Archival Video Compression & Storage Lifecycle**:
   - Implement `stream_fusion.storage.archiver.VideoArchiver` (FFmpeg two-pass/CRF AV1/HEVC downsampler with audio passthrough).
   - Integrate post-analysis compression hook into `FullSpectrumPipeline`.
2. **Execute Pipeline on Zackrawrr VOD (`2886498935`)**:
   - Ingest `chat.json` (106 MB) into ClickHouse `streamfusion.chat_events`.
   - Run `AdaptiveFrameOptimizer` (Spec 30) for frame reduction during idle screen periods.
   - Run audio transcription and visual inference.
   - Index moments into Qdrant vector collection `streamfusion_moments`.
   - Trigger the archival compression hook to replace the 32 GB source file with the ~3.5 GB archival copy.
   - Update VOD status in PostgreSQL catalog to `ANALYZED`.
