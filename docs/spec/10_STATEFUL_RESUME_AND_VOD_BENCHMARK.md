# StreamFusion: Stateful Checkpoint Resumption & Real VOD Benchmark (Spec 10)

## 1. Overview & Objectives
Full stream processing on real 6–10 hour broadcasts requires resilience against interruptions (crashes, system reboots, out-of-memory errors). In addition, all features must be verified on real-world stream data (such as the 60-second Asmongold test sample with real twitch chat and MP4 video) to prove practical effectiveness.

This specification details:
1. **Virtual Time-Slice Checkpointing & Resume Engine**:
   - Atomic partition writing of intermediate audio, vision, and chat slices.
   - Resumption manifest (`checkpoint_manifest.json`) verifying completed chunks via SHA-256 or chunk state flags.
   - Zero redundant compute when restarting an interrupted run.
2. **Real VOD End-to-End Benchmark Execution**:
   - Verification with `asmon_sample_60s.mp4` and `sample_asmon_chat.json`.
   - Pipeline run yielding all multimodal outputs:
     - Calibrated rolling latency curve
     - High-agreement highlight detection
     - 9:16 vertical short export with animated karaoke `.ass` subtitles
     - Interactive HTML report and Parquet/JSONL export.

---

## 2. Checkpoint-Resume Architecture

```
Cache Directory: cache/{stream_id}/
├── checkpoint_manifest.json
├── chunks/
│   ├── chunk_0000_audio.json
│   ├── chunk_0000_vision.json
│   ├── chunk_0000_chat.json
│   ├── chunk_0001_audio.json
│   └── ...
└── final/
    ├── stream_fusion_aligned.parquet
    └── report.html
```

### 2.1 Chunk Manifest Contract
```json
{
  "stream_id": "asmon_vod_123",
  "chunk_duration_sec": 300.0,
  "total_chunks_expected": 24,
  "completed_chunks": [0, 1, 2, 3],
  "status": "IN_PROGRESS",
  "last_updated": "2026-09-27T13:30:00Z"
}
```

### 2.2 Resumption Workflow
1. When `StreamFusionPipeline.run()` or `run_chunked()` starts, it inspects `cache_dir/{stream_id}/checkpoint_manifest.json`.
2. Any chunk listed in `completed_chunks` is deserialized directly from disk cache without invoking heavy models (Whisper / Florence-2).
3. Processing starts precisely at `first_incomplete_chunk`.
4. Once all chunks complete, the fusion matrix stitches the slices into a continuous dataset.

---

## 3. Real VOD Benchmark Test Suite
- Real media validation:
  - Input: `asmon_sample_60s.mp4` + `sample_asmon_chat.json`
  - Automated integration test `tests/test_real_asmon_benchmark.py` running in headless/mock or real mode.
  - Generates verifiable vertical clip and checks subtitle alignment and facecam box detection.

---

## 4. Deliverables & Acceptance Criteria
- Module: `src/stream_fusion/checkpoint/manager.py` (`CheckpointManager`, `ChunkState`).
- Integration with `src/stream_fusion/pipeline.py`.
- Benchmark test: `tests/test_checkpoint_manager.py` and `tests/test_real_asmon_benchmark.py`.
