# Spec 25: Targeted Streamer Roster Ingestion & Homelab Harvester Pipeline

## 1. Executive Summary & Objective

StreamFusion's analytical engine (Specs 01–24) provides industry-grade multi-modal intelligence: speech transcription, voiceprint matching, Florence-2 visual comprehension, adaptive slang tracking, knowledge graph extraction, claim grounding, sponsor auditing, and autonomous 9:16 vertical short production.

To feed this analytical engine continuously at scale, we require a **Targeted Streamer Roster Ingestion & Homelab Harvester Subsystem**. This system:
1. Accepts a configurable **Roster** of target streamers (names, URLs, platforms, quality preferences, download priorities).
2. Spawns managed **Worker Pools** that crawl channel endpoints, detect new/unprocessed VODs, and download both high-resolution video and timestamped chat replays.
3. Stages raw assets cleanly onto the **Homelab Server (NAS/Storage Pool)** with structured directory layouts, integrity checksums, and ingest descriptors.
4. Manages an internal **VOD Catalog (`catalog.db`)** tracking lifecycle states (`DISCOVERED`, `QUEUED`, `DOWNLOADING`, `HARVESTED`, `READY_FOR_ANALYSIS`, `ANALYZED`, `ERROR`).
5. Provides a frictionless bridge to trigger `FullSpectrumPipeline` on any harvested VOD on-demand or automatically.

```
+-----------------------------------------------------------------------------------------+
|                                Streamer Target Roster                                   |
| (YAML / JSON / CSV / SQLite): Streamer Names, Channel URLs, Priority, Quality Presets    |
+-----------------------------------------------------------------------------------------+
                                             │
                                             ▼
+-----------------------------------------------------------------------------------------+
|                      StreamFusion Homelab Harvester & Scheduler                         |
|  • Channel VOD Discovery & Lookback Crawler                                             |
|  • Concurrency Controller (Global & Per-Streamer limits)                                 |
|  • Bandwidth Throttling & Minimum Free Disk Space Guard                                 |
|  • yt-dlp Video Ingestion + chat-downloader Chat Replay                                 |
+-----------------------------------------------------------------------------------------+
                                             │
                                             ▼
+-----------------------------------------------------------------------------------------+
|                         Homelab Server Storage Architecture                             |
| \\homelab\storage\vods\<streamer_id>\<vod_date>_<vod_id>\                                |
|  ├── media.mp4                # Video stream (or audio-only preset)                     |
|  ├── chat.json                # Replay chat messages with offsets                       |
|  ├── metadata.json            # Streamer, title, views, game category                   |
|  ├── thumbnail.jpg            # Original stream thumbnail                               |
|  ├── checksums.sha256         # Integrity verification                                  |
|  └── ingest_manifest.json     # StreamFusion Ingest Descriptor                          |
+-----------------------------------------------------------------------------------------+
                                             │
                                             ▼
+-----------------------------------------------------------------------------------------+
|                     Handoff to StreamFusion FullSpectrumPipeline                        |
| `streamfusion harvest ingest-to-pipeline --vod-id <id>` -> 9-Phase Master Orchestrator  |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Core Architectural Components

### 2.1 Roster Models & Formats
A target streamer entry encapsulates everything needed to track and ingest a creator:

```python
class StreamerTargetRecord(BaseModel):
    streamer_id: str                      # Normalized slug (e.g. "asmongold", "kaicenat")
    display_name: str                     # Human name (e.g. "Asmongold")
    channel_urls: List[str]               # e.g. ["https://twitch.tv/asmon", "https://youtube.com/@zackrawrr"]
    primary_platform: str                 # "TWITCH", "YOUTUBE", "KICK"
    quality_preset: str = "best"          # "best", "1080p", "720p", "audio_only"
    include_chat: bool = True             # Fetch chat replay
    max_recent_vods: int = 5              # Max recent VODs to crawl per sync
    lookback_days: int = 14               # How far back to search
    download_priority: int = 5            # 1 (lowest) to 10 (highest)
    destination_override: Optional[str]   # Custom storage subpath
    tags: List[str] = []                  # ["tier1", "reaction", "gaming"]
    enabled: bool = True
```

The system accepts:
- `roster.yaml` / `roster.json`: Human-editable structured files.
- `roster.csv`: Bulk spreadsheets.
- Internal SQLite table `streamer_targets` inside `catalog.db`.

### 2.2 Homelab Storage Directory Specification
Assets harvested to the homelab server follow a deterministic, filesystem-safe convention:

```
<homelab_root>/
├── catalog.db                           # Central SQLite metadata database
├── roster.yaml                          # Default target roster
└── vods/
    └── <streamer_id>/
        └── <YYYY-MM-DD>_<vod_id>/
            ├── media.mp4                # Downloaded stream video
            ├── chat.json                # Standardized StreamFusion chat format
            ├── metadata.json            # Extracted platform metadata
            ├── thumbnail.jpg            # Preview graphic
            ├── checksums.sha256         # SHA-256 integrity hash
            └── ingest_manifest.json     # Ingest manifest ready for FullSpectrumPipeline
```

### 2.3 The Harvester Engine (`HomelabHarvester`)
Responsible for multi-threaded, resilient retrieval:
- **Concurrency Manager**:
  - `max_concurrent_workers: int = 3`: Maximum simultaneous VOD downloads to avoid ISP/network saturation.
  - `max_per_streamer: int = 1`: Prevents hitting anti-scraping / 429 rate limits against a single creator's channel.
- **Disk Safety Guard**:
  - `min_free_disk_gb: float = 50.0`: Checks available disk space on the homelab destination before initiating a download. Pauses new downloads if free space drops below threshold.
- **Idempotency & Partial Resume**:
  - Checks `catalog.db` before fetching. If status is `HARVESTED` or `COMPLETED`, skips immediately.
  - If a download was interrupted, passes `.part` resume flags to `yt-dlp`.
- **Chat Replay Ingestion**:
  - Uses `chat-downloader` or platform VOD chat APIs to produce standardized `ChatMessage` lists.

### 2.4 VOD Catalog Database (`catalog.db`)
Maintains comprehensive tracking across all harvested items:
- **`streamer_targets`**: Roster records.
- **`harvested_vods`**:
  - `vod_id` (PK)
  - `streamer_id`
  - `platform`
  - `title`
  - `published_at`
  - `duration_sec`
  - `status` (`DISCOVERED`, `QUEUED`, `DOWNLOADING`, `HARVESTED`, `INGESTED`, `ERROR`)
  - `video_path`
  - `chat_path`
  - `file_size_bytes`
  - `error_message`
  - `download_speed_mbps`
  - `harvested_at`

### 2.5 Bridge to `FullSpectrumPipeline`
Once a VOD is harvested on the homelab, it can be passed to the analytical pipeline with zero copying:
```bash
streamfusion harvest ingest-to-pipeline --vod-id twitch_v123456789 --run-shorts --ground-claims
```
This directly reads `media.mp4` and `chat.json` from the homelab mount and triggers the 9-phase synergy pipeline, storing analytical artifacts (matrix, HTML report, shorts) alongside the media.

---

## 3. CLI & JSON-RPC API Specifications

### 3.1 CLI Interface
```bash
# Roster Management
streamfusion harvest roster-add --id asmon --name "Asmongold" --url "https://twitch.tv/asmon" --priority 10
streamfusion harvest roster-import --file streamers.yaml
streamfusion harvest roster-list

# Crawling & Harvesting
streamfusion harvest sync           # Crawls all enabled streamers for new VODs
streamfusion harvest run            # Spawns harvester threads to process download queue
streamfusion harvest status         # Displays active downloads, queue depths, disk space

# Pipeline Trigger
streamfusion harvest ingest-to-pipeline --vod-id <id> [--shorts] [--ground-claims]
```

### 3.2 JSON-RPC 2.0 Methods
- `streamfusion.harvest.syncRoster`: Triggers channel crawl for new VODs.
- `streamfusion.harvest.startHarvester`: Starts the background harvester thread pool.
- `streamfusion.harvest.getHarvesterStatus`: Returns active worker threads, speeds, and queue stats.
- `streamfusion.harvest.listHarvestedVods`: Queries the catalog database with filtering.
- `streamfusion.harvest.triggerPipeline`: Launches `FullSpectrumPipeline` on a harvested VOD.

---

## 4. Definition of Done & Acceptance Criteria

1. **Roster Importer/Exporter**: Supports YAML, JSON, and CSV streamer definitions.
2. **Channel VOD Crawler**: Detects recent VODs across Twitch, YouTube, and Kick without duplicate entries.
3. **Resilient Harvester Thread Pool**: Concurrency control, per-streamer rate limiting, and minimum disk space checks.
4. **Homelab File Hierarchy**: Saves video, chat replay, metadata, thumbnails, and SHA-256 checksums to standard directory layout.
5. **Catalog Database**: SQLite-backed state machine tracking all stages from discovery to analysis.
6. **One-Command Pipeline Handoff**: `streamfusion harvest ingest-to-pipeline` executes `FullSpectrumPipeline` on any harvested asset.
7. **Full Test Suite Passing**: 100% green tests with thorough mocking for network/downloader components.
