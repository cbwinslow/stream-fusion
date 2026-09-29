# StreamFusion Session Handoff: Ready for Spec 29 (Polyglot Data Stack & Decoupled Homelab)

**Date**: 2026-09-29  
**Current Milestone**: Specs 01–28 **100% Complete & Verified**  
**Test Suite Status**: **220 / 220 tests passing (100% green)**  
**Target Next Spec**: **Spec 29: Polyglot Data Stack (PostgreSQL 17, ClickHouse, Qdrant) & Decoupled Homelab Topology**  

---

## 1. Project State Overview

StreamFusion has achieved a complete, production-grade multimodal livestream intelligence and RAG platform:
- **Specs 01–06**: Ingestion, demuxer, faster-whisper transcription, PySceneDetect + Florence-2 visual comprehension, dynamic broadcast latency calibration, and 1-second fusion matrix.
- **Specs 07–15**: Chat NLP, meme burst detection, sponsor quantifier, stateful resumption, voiceprint library, chatter profiling & grief detection, streamer knowledge graph & claim extraction, dense frame OCR, and telemetry.
- **Specs 16–20**: Unified JSON Schema protocol (85 registered schemas in `docs/schemas/`), self-expanding slang engine, subprocess worker isolation, real-time live ingestion & WebSocket stream tailing, and web grounding fact-checking.
- **Specs 21–24**: Autonomous multi-agent vertical short production, multi-platform live connectors (Twitch, YouTube, Kick), multi-stream co-stream debate synchronization, and full-spectrum master orchestrator (`FullSpectrumPipeline`).
- **Specs 25–27**: Targeted streamer roster ingestion (`roster.py`), dual-backend catalog (`HarvestCatalog`), 24/7 homelab scheduler daemon (`HomelabDaemon`), and unified web dashboard (`StreamFusion Studio` SPA).
- **Spec 28**: Semantic vector search & multimodal RAG knowledge engine (`src/stream_fusion/knowledge/search.py`) with hybrid BM25 + dense search, cross-stream viewpoint comparison, and pluggable vector backends (`postgresql://` with `pgvector`, `json://`, `sqlite://`).

---

## 2. Next Priority: Spec 29 — Polyglot Data Stack & Decoupled Homelab Topology

The goal of **Spec 29** is to scale StreamFusion to handle **millions of raw chat messages** and multi-node homelab deployments by pairing each data model with its ideal engine:

1. **Polyglot Persistence ("Right Tool for the Right Job")**:
   - **PostgreSQL 17**: ACID app state, streamer rosters, harvesting job queues (`FOR UPDATE SKIP LOCKED`), and claim graphs.
   - **ClickHouse**: Raw chat message stream (millions of rows), 10x-12x columnar compression, LowCardinality dictionaries, and native `ASOF JOIN` cross-section alignment.
   - **Qdrant (Rust)**: Multimodal semantic vectors with 32x Binary Quantization (BQ) and payload filtering on streamer/timestamp.
   - **Local NAS / Filesystem**: Raw VOD MP4s, segmented audio WAVs, and rendered video clips.

2. **Decoupled Two-Machine Topology**:
   - **Machine 1 (Homelab 24/7 Data Server)**: Hosts databases, bare-metal storage, and the harvesting daemon.
   - **Machine 2 (Client / GPU Workstation)**: Executes heavy CUDA inference (WhisperX, Florence-2 vision, OCR, FFmpeg short rendering) and serves the interactive Web Studio UI.

3. **Deployment Assets**:
   - `docker-compose.yml`: Portable reference deployment with `postgres:17-alpine`, `clickhouse/clickhouse-server:latest`, and `qdrant/qdrant:latest`.
   - `docs/homelab/bare_metal_setup.md`: Bare-metal Linux installation guide and systemd service units.

---

## 3. Implementation Checklist for Next Agent

1. **Virtual Environment & Baseline Verification**:
   - Path: `.venv\Scripts\python.exe`
   - Run tests: `.venv\Scripts\pytest.exe` (must confirm 220/220 passing green before starting).

2. **Pydantic Configuration Updates (`src/stream_fusion/models/schemas.py`)**:
   - Add `postgres_url`, `clickhouse_url`, `qdrant_url`, and `media_storage_path` to `DashboardConfig`.
   - Update schema registry (`src/stream_fusion/schema/registry.py`) and export updated schema to `docs/schemas/DashboardConfig.schema.json`.

3. **Implement Qdrant Vector Storage (`src/stream_fusion/knowledge/search.py`)**:
   - Implement `QdrantVectorStorage(BaseVectorStorage)` supporting collection creation, point upsert with payloads, cosine distance search, and Binary Quantization settings.
   - Update `create_vector_storage()` factory to route `qdrant://...`.
   - Provide graceful fallback / mock mode for zero-dependency local testing when Qdrant daemon is offline.

4. **Implement ClickHouse Chat Storage (`src/stream_fusion/knowledge/clickhouse.py`)**:
   - Implement `ClickHouseChatStorage` with `chat_events` and `speech_segments` tables.
   - Add batch insert for raw chat messages.
   - Add `asof_align_chat_reactions(vod_id, window_sec)` helper executing `ASOF JOIN` across speech and chat reaction spikes.
   - Provide in-memory / fallback execution for testing when ClickHouse daemon is offline.

5. **Create Homelab & Deployment Files**:
   - `docker-compose.yml` (PostgreSQL 17, ClickHouse, Qdrant, StreamFusion server).
   - `docs/homelab/bare_metal_setup.md` (Native Debian/Ubuntu installation instructions, systemd unit files, port references).

6. **Add Test Suite (`tests/test_polyglot_data_stack.py`)**:
   - Test `QdrantVectorStorage` CRUD, binary quantization parameters, and payload filtering.
   - Test `ClickHouseChatStorage` schema generation, batch insert, and `ASOF JOIN` query generation.
   - Test `DashboardConfig` network endpoint resolution.
   - Run full test suite: verify all 220 existing + new tests pass 100% green.

7. **Deliver Completion Handoff**:
   - Write `docs/spec/session_handoff_spec29_completed.md` following the Spec Protocol.
