# StreamFusion: System Architecture Specification

## 1. High-Level Pipeline

```mermaid
flowchart TD
    subgraph Ingestion
        A[Stream VOD URL / File] --> B[yt-dlp Video/Audio Extractor]
        A --> C[Chat Replay Downloader]
        B --> D[Demuxed Audio: 16kHz WAV]
        B --> E[Demuxed Video: MP4]
        C --> F[Raw Chat Events: JSONL]
    end

    subgraph AudioEngine [Audio Processing Engine]
        D --> G[ASR: Faster-Whisper]
        D --> H[Speaker Diarization: PyAnnote]
        G & H --> I[Diarized Transcripts: Streamer vs External Video]
    end

    subgraph VisionEngine [Screen & Vision Processing Engine]
        E --> J[Scene Change Detector / Keyframe Sampler]
        J --> K[Screen OCR: PaddleOCR / EasyOCR]
        J --> L[Local VLM: Florence-2 / Moondream2]
        K & L --> M[Timestamped Visual Summaries]
    end

    subgraph ChatEngine [Chat NLP & Sentiment Engine]
        F --> N[Broadcast Latency Calibrator: -t seconds]
        N --> O[Emote Frequency & Sentiment Scorer]
        O --> P[Chat Velocity & Polarity Windows]
    end

    subgraph FusionEngine [Temporal Fusion & Alignment Matrix]
        I & M & P --> Q[Time-Bucket Aggregator: 1s to 5s buckets]
        Q --> R[Unified Fusion Matrix: DataFrame / Parquet]
    end

    subgraph Export [Export & Analytics]
        R --> S[Interactive HTML Dashboard]
        R --> T[AI Highlight / Cut-List Generator]
        R --> U[Training Dataset JSONL]
    end
```

---

## 2. Memory-Constrained Hardware Strategy (RTX 3060 12GB)

To operate seamlessly on consumer GPUs without out-of-memory (OOM) errors, StreamFusion adopts a **Phased Sequential Pipeline**:

### Phase A: Audio Ingestion & Extraction (GPU: 2.5 – 3.5 GB)
1. Load `faster-whisper` (medium or large-v3-turbo) in `float16` or `int8`.
2. Extract word-level timestamps.
3. Run speaker diarization (`pyannote.audio` pipeline).
4. Save diarized intervals to local cache (`cache/audio_segments.parquet`).
5. **Explicitly teardown & garbage collect:** `del model; torch.cuda.empty_cache()`.

### Phase B: Vision & Screen Processing (GPU: 2.0 – 4.0 GB)
1. Sample frames using `ffmpeg` scene detection (threshold: 0.3) or static stride (every 2.0 seconds).
2. Load lightweight VLM (Microsoft Florence-2-base / large or Moondream2).
3. Batch-infer keyframes for scene description and screen text extraction.
4. Save frame descriptions to local cache (`cache/visual_keyframes.parquet`).
5. **Explicitly teardown & garbage collect:** `del model; torch.cuda.empty_cache()`.

### Phase C: Chat Parsing & Latency Alignment (CPU-bound)
1. Vectorize chat replay messages using Polars/Pandas.
2. Group into discrete temporal buckets (default: 2.0s windows).
3. Compute message velocity (msgs/sec), emote proportions (OMEGALUL, Pog, L, W, ???), and sentiment score.
4. Apply latency compensation offset: $T_{event} = T_{chat} - \Delta_{latency}$ (where $\Delta_{latency} \approx 4.0s$).

### Phase D: Matrix Fusion & Export (CPU-bound)
1. Perform outer join on time-bucket keys.
2. Interpolate visual state across speech and chat spikes.
3. Emit final Parquet dataset and interactive HTML report.
