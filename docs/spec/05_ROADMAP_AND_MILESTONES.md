# StreamFusion: Roadmap & Milestones Specification

## Milestone Overview

```mermaid
gantt
    title StreamFusion Development Milestones
    dateFormat YYYY-MM-DD
    section Phase 1: Core
    M1: Specs, Contracts & Demo CLI            :done, m1, 2026-09-27, 2026-09-27
    M2: Ingest & Real Chat Replay Connector    :active, m2, 2026-09-28, 2026-10-02
    section Phase 2: AI Engines
    M3: WhisperX Audio & Diarization Engine    :m3, 2026-10-03, 2026-10-08
    M4: Florence-2 Vision & OCR Engine         :m4, 2026-10-09, 2026-10-14
    section Phase 3: Alignment & Output
    M5: Dynamic Latency Calibration & Matrix   :m5, 2026-10-15, 2026-10-19
    M6: Interactive Dashboard & Highlight Cut  :m6, 2026-10-20, 2026-10-25
    section Phase 4: Release
    M7: PyPI Packaging & Community Release     :m7, 2026-10-26, 2026-10-31
```

---

## Detailed Milestone Deliverables & Acceptance Criteria

### Milestone 1: Architecture, Data Contracts & Functional Demo [COMPLETED]
* **Deliverables:**
  * Complete SDD suite (`01_NORTH_STAR_AND_REQUIREMENTS.md`, `02_ARCHITECTURE_AND_STACK.md`, `03_DATA_CONTRACTS.md`, `04_PIPELINE_MODULES.md`, `05_ROADMAP_AND_MILESTONES.md`).
  * Pydantic v2 data models for `ChatMessage`, `AudioSegment`, `VisualKeyframe`, and `FusionSlice`.
  * Working CLI (`streamfusion demo`) that aligns a simulated reaction segment and generates an interactive HTML dashboard.
  * Unit test suite passing with 100% green tests.
* **Acceptance Criteria:** `pytest` passes and CLI outputs valid HTML report without errors.

---

### Milestone 2: Live Ingest & Real Chat Replay Connector
* **Deliverables:**
  * `stream_fusion.ingest.downloader`: Wrapper for `yt-dlp` supporting time-slicing (`--download-sections`).
  * `stream_fusion.ingest.chat_replay`: Parser for live Twitch JSON and YouTube live chat format, extracting emotes (Twitch, BTTV, 7TV).
  * Demuxer producing 16kHz mono audio and video segment.
* **Acceptance Criteria:** Running `streamfusion ingest <url> --duration 120` downloads exact 2-minute audio, video, and chat JSON to disk.

---

### Milestone 3: Faster-Whisper + Diarization Engine
* **Deliverables:**
  * `stream_fusion.audio.transcriber`: `faster-whisper` integration with FP16/INT8 compute type.
  * `stream_fusion.audio.diarizer`: `pyannote.audio` speaker clustering.
  * Streamer voice anchor profile matcher tagging `STREAMER` vs `EXTERNAL_VIDEO`.
  * Explicit VRAM cleanup ensuring 0 bytes leaked after audio phase.
* **Acceptance Criteria:** Processes 5 minutes of mixed reaction audio in <30 seconds on RTX 3060, using <4.0 GB VRAM.

---

### Milestone 4: Adaptive Keyframing & Florence-2 Vision Engine
* **Deliverables:**
  * `stream_fusion.vision.sampler`: `PySceneDetect` boundary detector extracting scene-change keyframes.
  * `stream_fusion.vision.florence`: Microsoft Florence-2 integration for `<MORE_DETAILED_CAPTION>` and `<OCR>`.
  * Explicit VRAM cleanup ensuring model is unloaded before fusion phase.
* **Acceptance Criteria:** Analyzes all keyframes in <200ms per frame on RTX 3060, using <2.5 GB VRAM.

---

### Milestone 5: Dynamic Broadcast Latency Calibration & Fusion Matrix
* **Deliverables:**
  * `stream_fusion.chat.calibrator`: Cross-correlation engine between audio trigger bursts and chat message density to compute real broadcast lag.
  * `stream_fusion.fusion.matrix`: High-speed Polars/Pandas bucket alignment joining audio, visual, and calibrated chat.
  * Agreement score index and multi-signal highlight detector.
* **Acceptance Criteria:** Automatically calculates latency offset $\Delta t$ within ±0.5s of manual ground truth.

---

### Milestone 6: Interactive Dashboard & Highlight Clipper
* **Deliverables:**
  * Standalone HTML report with responsive scrubber and video sync.
  * Automated 9:16 vertical clip export (ffmpeg filter cropping facecam and reaction window with auto-captions).
  * Export to Premiere/DaVinci EDL/XML.
* **Acceptance Criteria:** 1-click generation of ready-to-upload YouTube Short for top highlight moments.

---

### Milestone 7: Open Source Distribution & Packaging
* **Deliverables:**
  * Public GitHub repository with clean CI workflow (`.github/workflows/test.yml`).
  * PyPI publication: `pip install stream-fusion`.
  * Pre-built documentation website.
* **Acceptance Criteria:** Clean `pip install stream-fusion` on a fresh machine executes `streamfusion --help`.
