# StreamFusion: North Star & Requirements Specification

## 1. The North Star

> **North Star Metric:**  
> **"Zero-Friction Multimodal Stream Reconstruction"**: Given any raw livestream URL (Twitch or YouTube) or local VOD file, StreamFusion must fully autonomously ingest, separate streamer speech from external video audio with **>90% precision**, extract visual screen context on scene transitions in **<200ms per frame**, automatically calculate broadcast latency offset ($\Delta t$) within **±0.5s error**, and output a temporally aligned analytical matrix and interactive dashboard running on consumer-grade hardware (**RTX 3060 12GB**) with **zero Out-of-Memory (OOM) failures**.

---

## 2. Core Personas & User Journeys

### Persona 1: Stream Video Editor (e.g., Editing for Asmongold TV / YouTube Shorts)
* **Goal:** Turn an 8-hour stream into 3 viral YouTube Shorts and 1 long-form react video without manually watching 8 hours of footage.
* **Journey:**
  1. Editor runs `streamfusion process <twitch_vod_url> --output-dir ./edits`.
  2. StreamFusion downloads and processes the VOD in the background.
  3. Editor opens the interactive HTML report (`report.html`), which highlights:
     * Timestamps where streamer paused an external video.
     * Peak audience chat velocity (`OMEGALUL`, `Pog`, `W`, `L`).
     * Streamer statement vs. audience sentiment agreement score.
  4. Editor clicks "Export EDL/XML" (or cut-list) and imports directly into Adobe Premiere or DaVinci Resolve.

### Persona 2: Creator Economy & Game Studio Analyst
* **Goal:** Quantify audience engagement during a 2-hour sponsored gameplay stream or reveal trailer reaction.
* **Journey:**
  1. Analyst inputs the stream segment where the game trailer was watched.
  2. StreamFusion extracts the screen OCR (game title, trailer text) and aligns it second-by-second with the chat sentiment curve.
  3. Analyst exports a CSV/Parquet dataset showing exact timestamps where chat sentiment went negative (e.g. price announcement, microtransactions) vs. positive (gameplay mechanics).

### Persona 3: Multimodal AI Researcher
* **Goal:** Extract paired multimodal datasets `(Screen Context + Streamer Audio) -> Real-time Audience Reaction` for fine-tuning foundation vision-language models.
* **Journey:**
  1. Researcher runs StreamFusion in batch mode across 50 streams.
  2. Output is emitted as standardized JSONL training triples ready for Hugging Face datasets.

---

## 3. Functional Requirements

### FR-1: Intelligent Ingestion
* **FR-1.1:** Must support both Twitch and YouTube livestream VOD URLs as well as pre-downloaded local MP4/MKV files.
* **FR-1.2:** Must support segment downloading (`--start-time`, `--duration`) to allow rapid testing and sub-clip analysis without downloading multi-gigabyte files.
* **FR-1.3:** Must pull complete timestamped chat replay with emote metadata (including 7TV, BTTV, and FrankerFaceZ emotes).

### FR-2: Reaction Audio Diarization (Streamer vs. Video)
* **FR-2.1:** Must extract audio demuxed at 16kHz mono.
* **FR-2.2:** Must transcribe spoken speech with word-level timestamps using `faster-whisper`.
* **FR-2.3:** Must perform speaker diarization (`pyannote.audio`) and classify segments into `STREAMER` (primary broadcast mic) vs. `EXTERNAL_VIDEO` (content played on stream) vs. `CO_HOST`.
* **FR-2.4:** Must support an acoustic "Streamer Anchor Profile" to deterministically identify the streamer's voice based on spectral audio properties.

### FR-3: Screen & Vision Context Parsing
* **FR-3.1:** Must not waste compute analyzing every 60fps frame; must use **adaptive scene change detection** (via `PySceneDetect` or dynamic perceptual hashing).
* **FR-3.2:** Must extract screen text (video titles, browser tabs, in-game UI) using unified OCR.
* **FR-3.3:** Must generate dense visual captions describing the active window and streamer facecam state.

### FR-4: Dynamic Broadcast Latency Calibration
* **FR-4.1:** Must not rely solely on a hardcoded latency offset.
* **FR-4.2:** Must compute the cross-correlation between audio/visual trigger events (e.g., streamer laughing or loud speech spike) and subsequent chat message density bursts to determine the exact stream latency delay $\Delta t \in [2.5s, 8.5s]$.

### FR-5: Unified Fusion Matrix & Highlighting
* **FR-5.1:** Must aggregate all modalities into synchronized time slices (default 2.0s).
* **FR-5.2:** Must compute rolling chat velocity, dominant emote distribution, and sentiment polarity.
* **FR-5.3:** Must calculate a Streamer-Chat "Agreement Score" (how strongly chat agrees or disagrees with streamer takes).
* **FR-5.4:** Must flag candidate "Spike / Highlight" moments based on multi-signal thresholding.

### FR-6: Export & Interoperability
* **FR-6.1:** Must generate a zero-dependency, self-contained interactive HTML dashboard with scrubbable timelines.
* **FR-6.2:** Must export machine-readable datasets in Apache Parquet and JSONL instruction-tuning triples for AI training.
* **FR-6.3:** Must automatically render detected highlight spikes into 9:16 vertical video shorts (cropping streamer facecam on top and content on bottom).

---

## 4. Non-Functional Requirements & Hardware Constraints

* **NFR-1 (Compute Boundary):** Must execute on a single **NVIDIA RTX 3060 (12GB VRAM)** on Windows/Linux without OOM exceptions.
* **NFR-2 (Memory Lifecycle):** Peak VRAM usage must not exceed 8.0 GB at any point in the pipeline. All neural models must be sequentially loaded, used, and explicitly evicted from GPU memory.
* **NFR-3 (Execution Speed):** Total pipeline processing time for a 1-hour stream must not exceed **20 minutes** on an RTX 3060.
* **NFR-4 (Packaging):** Must be installable via `uv` / `pip` with clean CLI entrypoints (`streamfusion`).

---

## 5. Detailed Criteria for Completion (Definition of Done)

| Phase | Deliverable | Status | Criteria for Completion (DoD) |
|---|---|---|---|
| **Phase 1: Spec & Contracts** | SDD Documentation & Schemas | **DONE** | All Pydantic data schemas pass strict validation; CLI options defined; specs approved. |
| **Phase 2: Ingestion & Chat** | Ingestion Module & Chat Analyzer | **DONE** | Ingests real Twitch VOD slices via `yt-dlp` and full chat replay with 7TV emotes via `TwitchDownloaderCLI`. |
| **Phase 3: Audio Diarization** | Audio Engine & Prosody | **DONE** | Faster-Whisper + Diarization runs on RTX 3060 with CUDA; RMS prosodic loudness & burst detection tested. |
| **Phase 4: Screen Vision** | Vision Engine & OCR | **DONE** | PySceneDetect samples keyframes; Florence-2 extracts OCR and scene descriptions in <200ms/frame with zero OOM. |
| **Phase 5: Matrix & Latency** | Dynamic Calibration & Fusion | **DONE** | Discrete cross-correlation auto-tunes stream lag ($\Delta t = 4.5s$ on real Asmon stream); Take Agreement Index calculated. |
| **Phase 6: Multi-Export & Shorts** | HTML, Parquet, JSONL & Clipper | **DONE** | Exports interactive HTML report, Apache Parquet matrix, Hugging Face training triples, and renders 9:16 vertical shorts. |
| **Phase 7: Packaging & Release** | PyPI & Open Source | **ACTIVE** | `uv.lock` deterministic builds, clean CLI help, 100% test coverage across 23 tests. |
