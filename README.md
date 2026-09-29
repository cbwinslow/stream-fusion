# StreamFusion 🎙️📹💬

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Architecture: Spec-Driven](https://img.shields.io/badge/Architecture-Spec--Driven-green.svg)](docs/spec/01_VISION_AND_SCOPE.md)

**StreamFusion** is an open-source, modular multimodal analysis framework for livestream VODs and chat replay streams (Twitch, YouTube).

It reconstructs what happened on stream by synchronizing **Speaker-Diarized Audio**, **Screen & Visual Context**, and **Audience Chat Reactions** into a unified temporal matrix.

```
                      ┌────────────────────────────────────────────────────────┐
                      │                 RAW STREAM / VOD INPUT                 │
                      └──────────────────────────┬─────────────────────────────┘
                                                 │
                   ┌─────────────────────────────┼─────────────────────────────┐
                   ▼                             ▼                             ▼
         [ Audio Demuxer ]             [ Demuxer / Optimizer ]        [ Chat Replay / IRC ]
                   │                             │                             │
                   ▼                             ▼                             ▼
       Faster-Whisper / WhisperX       Adaptive Frame Optimizer        Twitch / YouTube Rechat
        (Diarized Transcripts)           (Chat-Grounded Keyframes)      (Latency Calibration)
                   │                             │                             │
                   ▼                             ▼                             ▼
        [ Speaker Diarization ]         [ Florence-2 / OCR ]           [ Emote & Sentiment ]
                   │                             │                             │
                   └─────────────────────────────┼─────────────────────────────┘
                                                 │
                                                 ▼
                                     ┌───────────────────────┐
                                     │  FULL-SPECTRUM FUSION │
                                     │     MASTER MATRIX     │
                                     └───────────┬───────────┘
                                                 │
                         ┌───────────────────────┼───────────────────────┐
                         ▼                       ▼                       ▼
                  PostgreSQL 17              ClickHouse               Qdrant
               (Catalog, State, VODs)   (Chat Events, Metrics)   (Vector Embeddings)
```

---

## Key Features

1. **Reaction Diarization:** Distinguishes between the streamer speaking directly into their microphone vs. external audio from videos/games playing on stream.
2. **Visual Screen Understanding:** Uses compact Vision-Language Models (Florence-2 / Moondream2) and OCR to describe active windows, games, and streamer facial expressions.
3. **Calibrated Chat Grounding:** Accounts for the 3–8 second broadcast latency delay, accurately pairing audience emote spikes (`OMEGALUL`, `Pog`, `W`, `L`, `???`) with the precise triggering visual/audio event.
4. **Adaptive Frame Density Optimizer (Spec 30):** Dynamically scales sampling rate from 0.2 FPS (idle) up to 2.0 FPS during chat bursts, saving 50%+ frame compute while maintaining 100% burst visual coverage.
5. **Polyglot Homelab Architecture (Spec 29):** Native bare-metal integrations with PostgreSQL 17 (state/leases), ClickHouse (high-velocity chat events & ASOF joins), and Qdrant (32x BQ vector embeddings).
6. **Unified Web Studio Dashboard (Spec 27):** FastHTML + Tailwind CSS real-time monitoring interface with live WebSocket log streaming, video scrubbing, and vector search inspection.
7. **Autonomous Multi-Agent Short Studio (Spec 21):** Automatic 9:16 vertical short creation, dynamic word-level subtitles, smart cropping, and auto-publishing.
8. **Hardware Optimized for Consumer GPUs:** Sequentially executes stages to run comfortably on a single **NVIDIA RTX 3060 (12GB)** without out-of-memory errors.

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/cbwinslow/stream-fusion.git
cd stream-fusion
pip install -e .
```

To install GPU acceleration and audio/vision dependencies:
```bash
pip install -e ".[audio,vision,ingest,rag,web]"
```

### 2. Configuration (`.env`)

StreamFusion prioritizes native bare-metal instances on your local/homelab network:

```env
POSTGRES_URL=postgresql://streamfusion:password@192.168.10.1:5434/streamfusion
CLICKHOUSE_URL=http://192.168.10.1:8123
QDRANT_URL=http://192.168.10.1:6333
MEDIA_STORAGE_PATH=/home/cbwinslow/workspace/streamfusion/vods
```

*(A standalone, portable `docker-compose.yml` is also included for isolated containerized environments).*

### 3. Usage & CLI

#### Run Full-Spectrum Analysis on a VOD:
```bash
streamfusion analyze "https://www.twitch.tv/videos/2886498935" --out ./output
```

#### Launch the Web Studio Dashboard:
```bash
streamfusion dashboard --host 0.0.0.0 --port 8000
```

#### Search Across Stream Transcripts & Moments (Multimodal RAG):
```bash
streamfusion search "Asmongold reacts to drama beta test" --limit 10
```

---

## Complete Specification Roadmap (Specs 1–30)

StreamFusion is engineered under rigorous **Spec-Driven Development (SDD)**:

| Spec | Title | Core Focus |
|:---:|---|---|
| **01** | [Vision & Scope](docs/spec/01_VISION_AND_SCOPE.md) | High-level roadmap and problem statement |
| **02** | [System Architecture](docs/spec/02_ARCHITECTURE.md) | Modular component decomposition |
| **03** | [Data Contracts & Schemas](docs/spec/03_DATA_CONTRACTS.md) | Standardized JSON/Parquet schemas |
| **04** | [Homelab & Deployment Guide](docs/spec/04_HOMELAB_AND_DEPLOYMENT.md) | Hardware sizing & deployment topology |
| **19** | [Real-Time Audio Diarization](docs/spec/19_REAL_TIME_AUDIO_DIARIZATION_AND_SPEAKER_CLUSTERING.md) | Voiceprint clustering & reaction separation |
| **20** | [Web Grounding & Knowledge Graph](docs/spec/20_WEB_GROUNDING_AND_LIVE_KNOWLEDGE_GRAPH_EXPANSION.md) | Dynamic entity grounding via web search |
| **21** | [Autonomous Multi-Agent Short Studio](docs/spec/21_AUTONOMOUS_MULTI_AGENT_SHORT_PRODUCTION_AND_AUTO_PUBLISHER.md) | Vertical video generation, dynamic captions & publishing |
| **22** | [Multi-Platform Connectors](docs/spec/22_MULTI_PLATFORM_LIVE_STREAM_AND_CHAT_CONNECTORS.md) | Twitch IRC, YouTube Live & Kick integrations |
| **23** | [Co-stream & Cross-Platform Alignment](docs/spec/23_MULTI_STREAM_COSTREAM_AND_CROSS_PLATFORM_ALIGNMENT.md) | Multi-stream clock synchronization |
| **24** | [Full-Spectrum Master Pipeline](docs/spec/24_FULL_SPECTRUM_PIPELINE_UNIFICATION_AND_MASTER_ORCHESTRATOR.md) | End-to-end multi-stage orchestrator |
| **25** | [Targeted Streamer Harvester](docs/spec/25_TARGETED_STREAMER_ROSTER_AND_HOMELAB_HARVESTER.md) | 24/7 background VOD discovery & deduplication |
| **26** | [Homelab Scheduler Daemon](docs/spec/26_HOMELAB_SCHEDULER_DAEMON_AND_SERVICE_ORCHESTRATION.md) | Job leasing, systemd unit templates & priority queues |
| **27** | [Unified Web Dashboard & Studio](docs/spec/27_UNIFIED_WEB_DASHBOARD_AND_MONITORING_FRONTEND.md) | FastHTML + Tailwind CSS real-time interface |
| **28** | [Semantic Vector Search & RAG](docs/spec/28_SEMANTIC_VECTOR_SEARCH_AND_MULTIMODAL_RAG_ENGINE.md) | Qdrant 32x Binary Quantization multimodal retrieval |
| **29** | [Polyglot Homelab Data Stack](docs/spec/29_POLYGLOT_DATA_STACK_POSTGRES_CLICKHOUSE_QDRANT_HOMELAB.md) | Bare-metal PostgreSQL 17, ClickHouse & Qdrant |
| **30** | [Adaptive Frame Density Optimizer](docs/spec/30_ADAPTIVE_FRAME_DENSITY_OPTIMIZER_AND_PRODUCTION_PROFILER.md) | Chat-grounded dynamic sampling & production profiler |

---

## Bare-Metal Homelab Installation Guide

Detailed bare-metal installation instructions for Debian/Ubuntu hosts, PostgreSQL 17 multi-cluster setups, ClickHouse single-binary deployments, and Qdrant systemd configuration can be found in:
* **[Bare-Metal Homelab Installation Guide](docs/homelab/bare_metal_setup.md)**

---

## License

MIT License. See [LICENSE](LICENSE) for details.
