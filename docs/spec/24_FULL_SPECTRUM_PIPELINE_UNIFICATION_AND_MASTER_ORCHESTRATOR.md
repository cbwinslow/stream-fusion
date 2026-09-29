# StreamFusion: Full-Spectrum Pipeline Unification & Master Synergy Orchestrator (Spec 24)

## 1. Overview & North Star Goals

Over Specs 01 through 23, StreamFusion developed deep, specialized capabilities across multimodal streaming:
- **Audio & Diarization**: Faster-Whisper, PyAnnote diarization, biometric Voiceprint profiles, and prosody.
- **Vision & Keyframing**: Florence-2 dense captions, screen OCR, facecam bounding box detection, and bounded frame buffering.
- **Chat & Community NLP**: Dynamic latency cross-correlation ($\Delta t$), adaptive slang learning, chatter safety profiling, and meme burst cascades.
- **Knowledge & Grounding**: Claim extraction, web fact-checking citations, and longitudinal stance shift tracking.
- **Production & Monetization**: Brand sponsor quantification, multi-agent vertical short studio (Director, Editor, Policy, Copywriter, Publisher), and multi-stream co-host synchronization.

However, operating these capabilities previously required running separate CLI commands or decoupled scripts. **Spec 24 creates the Top-Level Synergy Orchestrator (`FullSpectrumPipeline`)**—a unified, production-grade engine that ties together every subsystem into a single, cohesive end-to-end analytical pipeline.

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   INPUT: RAW STREAM VOD OR REPLAY                                │
│                     media.mp4 (Local/Twitch/YouTube)  +  chat_replay.json                        │
└───────────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                                │
                                                ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                            FullSpectrumPipeline (MASTER ORCHESTRATOR)                            │
│                                                                                                  │
│   [Stage 1: Ingestion & Demuxing]                                                                │
│   Extracts 16kHz mono audio WAV + Scene-change keyframe image series                             │
│                                               │                                                  │
│   [Stage 2: Worker-Isolated Audio & Diarization] (GPU)                                           │
│   Transcribes word timings + tags STREAMER/VIDEO + Voiceprint creator matching                   │
│                                               │                                                  │
│   [Stage 3: Worker-Isolated Screen Vision & OCR] (GPU)                                           │
│   Dense Florence-2 captions, screen OCR, facecam detection with bounded buffer                   │
│                                               │                                                  │
│   [Stage 4: Adaptive Chat Calibration & Slang Discovery]                                         │
│   Computes latency Δt + updates persistent AdaptiveLexiconStore + ChatterProfiler                │
│                                               │                                                  │
│   [Stage 5: Multimodal Fusion Matrix & Highlight Detection]                                      │
│   Synchronizes audio/vision/calibrated chat + computes Take Agreement Index (A)                  │
│                                               │                                                  │
│   [Stage 6: Sponsor & Brand Quantifier]                                                          │
│   Detects brand mentions + calculates sponsor delta sentiment + SponsorImpactReport              │
│                                               │                                                  │
│   [Stage 7: Knowledge Graph Extraction, Web Grounding & Stance Shifts]                           │
│   Extracts claims + verifies via LiveWebGroundingEngine + records in TemporalStanceShiftTracker  │
│                                               │                                                  │
│   [Stage 8: Autonomous Multi-Agent Short Studio]                                                 │
│   Director -> Editor -> Policy -> Copywriter -> Publisher renders 9:16 Shorts with subtitles     │
│                                               │                                                  │
│   [Stage 9: Full-Spectrum Synergy Manifest & Event Bus Dispatch]                                 │
│   Outputs full_spectrum_manifest.json + Emits universal StreamFusionEnvelopes                    │
└───────────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                                │
                                                ▼
                     Unified FullSpectrumManifest, Parquet, JSONL, & HTML Report
```

### North Star Goals:
1. **End-to-End Multimodal Synergy**: Run a complete stream analysis in one command that automatically populates the knowledge graph, trains the adaptive slang dictionary, audits brand sponsors, fact-checks streamer claims against live web sources, and produces ready-to-upload 9:16 vertical shorts.
2. **Subsystem Fault Tolerance**: Non-critical downstream stages (e.g. web search timeout or social copy generation) run behind isolated safety guards. A failure in an external web lookup or video encoder never fails the entire pipeline; stages report status (`SUCCESS`, `SKIPPED`, `DEGRADED`).
3. **Strict Resource Management**: Guarantees zero GPU VRAM leakage via sequential subprocess worker isolation (`WorkerIsolationManager`) and zero disk bloat via bounded frame buffering (`BoundedFrameBuffer`).
4. **Rich Unified Manifest (`FullSpectrumManifest`)**: Emits a single master JSON bundle linking the analytical matrix, sponsor impact report, grounded claims, stance shift history, adaptive lexicon additions, and vertical short packages.
5. **100% Backward Compatibility**: All 165 existing tests pass without regressions.

---

## 2. Pipeline Execution Stages & Architecture

### 2.1 Configuration: `FullSpectrumConfig`
A unified configuration model controlling all stage parameters:
- `isolate_gpu_workers: bool = True`
- `bounded_buffer: bool = True`
- `enable_adaptive_slang: bool = True`
- `enable_web_grounding: bool = True`
- `enable_stance_tracking: bool = True`
- `enable_sponsor_quantifier: bool = True`
- `enable_short_production: bool = True`
- `short_candidate_count: int = 3`
- `dry_run_shorts: bool = False`
- `stance_history_db: Path = Path("./stance_history.json")`
- `adaptive_lexicon_db: Path = Path("./adaptive_lexicon.json")`

### 2.2 Stage Lifecycle & Degraded Fallbacks
Each stage is measured and wrapped in telemetry:
- If `enable_web_grounding` encounters network timeouts, claims are flagged as `UNVERIFIABLE` and stage marks `DEGRADED`.
- If `enable_short_production` fails or `ffmpeg` is missing, stage marks `DEGRADED`, preserving analytical matrices and data exports.

---

## 3. Subsystem Layout

```
src/stream_fusion/orchestration/
├── __init__.py                 # Public exports
├── full_spectrum.py            # FullSpectrumPipeline & FullSpectrumOrchestrator
└── manifest.py                 # FullSpectrumManifest builder & serializer
```

---

## 4. Acceptance Criteria & Definition of Done

1. `FullSpectrumPipeline` seamlessly connects all 8 core intelligence stages into a single unified execution flow.
2. Resource management: sequential worker isolation and bounded frame buffers ensure zero VRAM leakage and bounded disk usage.
3. Fault tolerance: optional stage errors do not crash the master pipeline.
4. Generates a comprehensive `FullSpectrumManifest` summarizing all multimodal outputs.
5. CLI command `streamfusion run-all` / `streamfusion full-spectrum` with comprehensive options.
6. JSON-RPC 2.0 method `streamfusion.runFullSpectrum`.
7. Full test coverage in `tests/test_full_spectrum_pipeline.py`.
8. All existing 165 tests + new tests pass (100% green).
