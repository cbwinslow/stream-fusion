# Spec 27: Unified Web Dashboard & Real-Time Studio

**Status**: In Progress / Target Complete  
**Implementation Modules**: `src/stream_fusion/web/`, `src/stream_fusion/cli.py`, `src/stream_fusion/models/schemas.py`  
**Test Suite**: `tests/test_dashboard_frontend.py`  
**Registry Schemas**: `DashboardConfig`, `DashboardOverviewStats`, `ScrubberTimelinePayload`, `ShortStudioExportRequest`, `LiveTailSubscriptionRequest`, `LiveStreamActionResponse` exported to `docs/schemas/`  

---

## 1. Executive Summary & Objective

With Specs 01 through 26 complete and verified across 197/197 passing tests, StreamFusion possesses a full-spectrum analytical and autonomous production backend:
- Ingestion, demuxer, whisper transcription, Florence-2 / scene detection visual analysis, and dynamic latency calibration (Specs 01–06).
- Chat NLP, meme bursts, sponsor quantification, voiceprint diarization, chatter profiling, streamer knowledge graph, dense frame OCR, and telemetry (Specs 07–15).
- Unified JSON Schemas, adaptive slang engine, isolated subprocess workers, real-time live ingestion, and web grounding fact-checking (Specs 16–20).
- Autonomous multi-agent vertical short studio, multi-platform live connectors, and co-stream alignment (Specs 21–23).
- Full-spectrum 9-phase master orchestrator (Spec 24).
- Targeted streamer roster harvester and dual-backend catalog (Spec 25).
- 24/7 background scheduler daemon and service orchestration (Spec 26).

**Spec 27** delivers the capstone user-facing interface: **The Unified Web Dashboard & Real-Time Studio**. This provides a responsive web control center and interactive analytical workbench that binds together all 26 backend subsystems into an intuitive, high-performance web application.

```
+--------------------------------------------------------------------------------------------------------+
|                                    StreamFusion Web Dashboard & Studio                                 |
|                                                                                                        |
|  [Tab 1: Homelab Overview] [Tab 2: Multimodal Scrubber] [Tab 3: Short Studio]                          |
|  [Tab 4: Knowledge Explorer] [Tab 5: Live Tail Monitor]            [WS Status: Connected | 60 FPS]     |
+--------------------------------------------------------------------------------------------------------+
                                                    │
                                     (HTTP REST, WebSockets, JSON-RPC)
                                                    ▼
+--------------------------------------------------------------------------------------------------------+
|                                        FastAPI Application Server                                      |
|                                                                                                        |
|   ┌────────────────────────┐  ┌─────────────────────────┐  ┌───────────────────────────────────────┐   |
|   │     REST API Endpoints │  │   WebSocket Feeds       │  │       JSON-RPC 2.0 Gateway            │   |
|   │  • /api/roster         │  │   • /ws/live            │  │   • /api/rpc (AgentRpcDispatcher)     │   |
|   │  • /api/catalog        │  │   • /ws/events          │  └───────────────────────────────────────┘   |
|   │  • /api/daemon         │  │   (LiveEventBroadcaster)│  ┌───────────────────────────────────────┐   |
|   │  • /api/vods/timeline  │  └─────────────────────────┘  │       Static Asset Mount              │   |
|   │  • /api/shorts         │                               │   • / -> static/index.html            │   |
|   │  • /api/knowledge      │                               │   • /assets -> static/css, js         │   |
|   │  • /api/media          │                               └───────────────────────────────────────┘   |
|   └────────────────────────┘                                                                           |
+--------------------------------------------------------------------------------------------------------+
                                                    │
                                                    ▼
+--------------------------------------------------------------------------------------------------------+
|                                      StreamFusion Subsystems                                           |
|   • HarvestCatalog (SQLite / PostgreSQL)     • HomelabDaemon & HomelabScheduler                        |
|   • FullSpectrumPipeline                     • Autonomous Short Producer (Script/Cut/Render)           |
|   • StreamerKnowledgeGraph & WebGrounder     • LiveStreamCoordinator & LiveChatTailer                  |
|   • SponsorQuantifier & ChatNLP              • LiveEventBroadcaster (Pub/Sub Ring Buffer)              |
+--------------------------------------------------------------------------------------------------------+
```

---

## 2. Architecture & Core Views

The Web Dashboard is structured around 5 primary functional views:

### View 1: Homelab & Harvester Overview
- **Roster Manager**: Interactive CRUD interface for target streamer channels (Twitch, YouTube, Kick). Displays handle, platform badge, check frequency, download priority, and enabled toggle.
- **Queue & Download Monitor**: Live progress cards for VODs in `DISCOVERED`, `QUEUED`, `DOWNLOADING`, `HARVESTED`, `ANALYZING`, `ANALYZED`, and `FAILED` states.
- **Storage Gauge**: Visual representation of disk capacity, utilized storage, pruned files, and retention policy status.
- **Daemon Supervisor Controls**: Real-time daemon status indicator (`RUNNING`, `PAUSED`, `STOPPED`, `ERROR`), PID lock indicator, and action buttons (`Start`, `Pause`, `Resume`, `Stop`, `Trigger Instant Crawl`).

### View 2: Multimodal Scrubber & Stream Player
- **Interactive Player**: Synchronized HTML5 video player with seekable controls.
- **Multimodal Timeline Strip**: Time-aligned multi-track scrubber showing:
  - Scene boundary changes from PySceneDetect.
  - Speech transcription segments with word-level highlight timings.
  - Chat intensity heatmap with meme burst flags (`POG_BURST`, `OMEGALUL_BURST`).
  - Sponsor segment overlays with brand identification.
  - Claim timestamps with fact-check verdict colors.
- **Synchronized Chat Waterfall**: Live chat replay window that advances in real-time lockstep with the video scrubber position.
- **Visual Keyframe & OCR Inspector**: Displays detected dense frame OCR text, visual keyframe thumbnails, and bounding boxes for active scenes.

### View 3: Autonomous Short Studio
- **Vertical 9:16 Preview Player**: Mobile device mockup previewing generated short candidates.
- **Camera Layout Inspector**: Visual bounding box overlay showing facecam crop coordinates, game capture framing, and multi-speaker layout splits.
- **Virality Scorecard**: Detailed metric breakdown (Hook Efficacy, Meme Density, Retention Prediction, Emotional Peak).
- **Script & Subtitle Styler**: Three-act breakdown (Hook, Build, Punchline) with customizable dynamic subtitle styles (e.g. `DYNAMIC_WORD_HIGHLIGHT`, `TIKTOK_BOUNCE`, `MINIMAL_CLEAN`).
- **One-Click Render & Export**: Triggers rendering pipeline or auto-publishing to YouTube Shorts / TikTok.

### View 4: Knowledge & Stance Explorer
- **Streamer Claim Fact-Checks**: Searchable table of factual claims extracted by LLM/NLP with truth ratings (`TRUE`, `FALSE`, `MIXED`, `UNVERIFIABLE`), grounding search citations, and confidence scores.
- **Entity Stance Polarity Timeline**: Chronological shift of streamer sentiment towards key entities (e.g. game publishers, competing platforms, drama subjects) with net polarity scores `[-1.0, +1.0]`.
- **Sponsor Quantifier Impression Logs**: Detected brand impressions, screen duration, chatter sentiment during sponsor segments, and estimated CPM performance.

### View 5: Live Tail Monitor
- **Live Stream Receiver**: Low-latency monitor displaying incoming broadcast stream chunks.
- **Live IRC Chat Feed**: Real-time WebSocket waterfall streaming chat messages from active channels.
- **Real-Time Meme Burst Alerter**: Dynamic threshold gauge alerting whenever chat velocity triggers a burst event.
- **Telemetry Gauges**: Real-time FPS, dropped frame counters, audio buffer health, and latency drift calibration readings.

---

## 3. API & Protocol Specifications

### 3.1 REST API Routes
- `GET /api/system/health`: System health, uptime, disk stats, version.
- `GET /api/system/stats`: Comprehensive dashboard overview stats (`DashboardOverviewStats`).
- `GET /api/roster`: List all configured streamers.
- `POST /api/roster`: Create or update a streamer target.
- `DELETE /api/roster/{streamer_id}`: Remove streamer from roster.
- `GET /api/catalog`: List harvested VODs with status, search, and streamer filters.
- `GET /api/catalog/{vod_id}`: Detailed record for a specific VOD.
- `POST /api/catalog/queue`: Manually enqueue a VOD URL.
- `POST /api/catalog/crawl`: Trigger immediate roster crawl.
- `POST /api/catalog/analyze/{vod_id}`: Trigger full spectrum pipeline analysis for a VOD.
- `GET /api/daemon/status`: Retrieve `DaemonStatusReport`.
- `POST /api/daemon/start`: Start daemon background workers.
- `POST /api/daemon/pause`: Pause scheduled queue workers.
- `POST /api/daemon/resume`: Resume scheduled workers.
- `POST /api/daemon/stop`: Gracefully stop daemon.
- `GET /api/vods/{vod_id}/timeline`: Retrieve `ScrubberTimelinePayload` for multimodal player.
- `GET /api/vods/{vod_id}/slices`: Retrieve fusion slices.
- `GET /api/vods/{vod_id}/chat`: Retrieve chat messages for VOD with timestamp filtering.
- `GET /api/shorts`: List short candidates across all or specific VODs.
- `GET /api/shorts/{candidate_id}`: Retrieve detailed short candidate package.
- `POST /api/shorts/{candidate_id}/render`: Render candidate to 9:16 vertical MP4.
- `GET /api/knowledge/claims`: Retrieve extracted streamer claims.
- `GET /api/knowledge/stances`: Retrieve entity stance history.
- `GET /api/knowledge/sponsors`: Retrieve sponsor impact audits.
- `GET /api/media/{vod_id}/video`: Stream video MP4 file with byte-range support.

### 3.2 WebSocket Feeds
- `WS /ws/live`: Real-time duplex feed streaming `StreamFusionEnvelope` events (chat, keyframes, bursts, telemetry).
- `WS /ws/events`: Broadcast feed emitting daemon state transitions, pipeline stage progress, and download updates.

### 3.3 JSON-RPC 2.0 Gateway
- `POST /api/rpc`: Accepts standard JSON-RPC 2.0 payloads and delegates to `AgentRpcDispatcher` for full compatibility with autonomous subagents and CLI automation.

---

## 4. Schema Contracts

All models are defined in `src/stream_fusion/models/schemas.py` and registered in `src/stream_fusion/schema/registry.py`:
1. `DashboardConfig`: Host, port, root path, catalog database URL, CORS, static dir.
2. `DashboardOverviewStats`: Summary counts of streamers, VODs, analyzed VODs, disk space, and active live streams.
3. `ScrubberTimelinePayload`: Aggregated multimodal tracks for playback synchronization.
4. `ShortStudioExportRequest`: Options for rendering and publishing 9:16 shorts.
5. `LiveTailSubscriptionRequest`: Configuration for live WebSocket topic subscriptions.
6. `LiveStreamActionResponse`: Standardized acknowledgement for live stream management actions.

---

## 5. Verification & Testing Protocol

- **Unit & Integration Suite**: `tests/test_dashboard_frontend.py`
  - FastAPI application lifespan and health check routes.
  - Roster CRUD API functionality.
  - VOD catalog listing, detail, and pipeline trigger endpoints.
  - Daemon state inspection and remote control commands.
  - Multimodal scrubber timeline aggregation and chat replay slicing.
  - Short studio candidate queries and render request handling.
  - Knowledge graph claim queries and sponsor impact records.
  - JSON-RPC 2.0 HTTP gateway routing to `AgentRpcDispatcher`.
  - WebSocket client connection, subscribe, and broadcast delivery.
  - CLI `streamfusion dashboard` commands (`serve`, `status`, `open`).
- Maintain 100% green test suite status across all 197 prior tests.
