# StreamFusion: Subprocess Worker Isolation & Bounded Buffering (Spec 18)

## 1. North Star & System Objectives
Full-length broadcast streams commonly span 6 to 10+ hours. In traditional monolithic Python pipelines, long-running processes experience two fatal failure modes:
1. **CUDA Memory Fragmentation & VRAM Leakage**: Python garbage collection and `torch.cuda.empty_cache()` do not release the underlying CUDA driver contexts, cuBLAS workspaces, or PyAnnote/CTranslate2 allocations back to the OS. Over hours of continuous chunk execution, memory pool fragmentation causes non-deterministic CUDA Out-Of-Memory (OOM) crashes.
2. **Unbounded Disk Bloat from Keyframe Extraction**: Sampling keyframes at 1 frame per 2 seconds over a 10-hour VOD generates 18,000 raw image files (~15–25 GB), risking disk exhaustion and high I/O latency.

**Spec 18** establishes an enterprise-grade execution harness:
- **Subprocess Worker Isolation (`SubprocessWorkerManager`)**: Executes heavy AI inference workloads (CTranslate2 Faster-Whisper, PyAnnote Diarization, Microsoft Florence-2) in isolated child processes with standardized JSON IPC (`StreamFusionEnvelope`). Upon worker exit, the operating system unconditionally reclaims 100% of GPU VRAM and process RAM.
- **Bounded Rolling Buffering (`BoundedFrameBuffer`)**: Processes video in sliding time windows (e.g. 30–60s), extracting keyframes on-the-fly, extracting multimodal features, and immediately purging image files from disk to maintain an $O(1)$ constant disk footprint ($\le 50\text{ MB}$).

---

## 2. Architecture & Data Flow

```
                      StreamPipeline (Main Process)
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│  BoundedFrameBuffer (Streaming) │       │  SubprocessWorkerManager (IPC)  │
│  - Window size: 30–60s          │       │  - Zero VRAM Leakage            │
│  - Extracts window frames       │       │  - JSON Envelope Protocol       │
│  - Feeds Vision Worker          │       │  - Supervises Timeouts/Crashes  │
│  - Purges JPEGs immediately     │       │  - OS Reclaims 100% CUDA Pools  │
└─────────────────────────────────┘       └─────────────────────────────────┘
         │                                                 │
         │                                                 ▼
         │                               ┌─────────────────────────────────┐
         │                               │     Child Worker Subprocess     │
         │                               │  (Whisper / Diarizer / Florence)│
         └──────────────────────────────►│                                 │
                                         │  Reads: task_input.json (env)   │
                                         │  Runs: GPU Model Inference      │
                                         │  Writes: result.json (env)      │
                                         │  Exits cleanly (Exit code 0)    │
                                         └─────────────────────────────────┘
```

---

## 3. Worker Protocol & IPC Specifications

Child worker processes communicate with the main pipeline via standard JSON envelopes ([Spec 16](file:///C:/Users/blain/Documents/stream-fusion/docs/spec/16_UNIFIED_JSON_SCHEMA_AND_AGENT_COMMUNICATION.md)):

### 3.1 Task Input Envelope (`WorkerTaskInput`)
```json
{
  "version": "1.0",
  "task_type": "AUDIO_TRANSCRIPTION",
  "media_path": "chunk_0.wav",
  "config": {
    "whisper_model": "base",
    "compute_type": "float16",
    "device": "cuda"
  },
  "timeout_sec": 300.0
}
```

### 3.2 Task Output Envelope (`StreamFusionEnvelope`)
```json
{
  "version": "1.0",
  "message_id": "c1f8021b-410a-4fa0-b389-c454e9bc3861",
  "stream_id": "asmon_sample_60s",
  "event_type": "STAGE_COMPLETE",
  "producer": "stream_fusion.workers.audio",
  "payload": {
    "segments": [ ... ]
  },
  "telemetry": {
    "duration_ms": 4820.5,
    "stage": "worker_audio_transcription",
    "ram_mb": 420.1,
    "vram_mb": 1120.0
  }
}
```

---

## 4. Bounded Frame Buffering Contract

Let media duration be $T$.
1. Rather than extracting all $N = \frac{T}{\Delta t_{\text{sample}}}$ frames upfront:
2. Slice into streaming windows $W_k = [t_k, t_k + \Delta t_W]$.
3. For each window $W_k$:
   - Demux only frames within $[t_k, t_k + \Delta t_W]$ to a temporary window folder.
   - Run vision feature and OCR extraction.
   - Persist extracted `VisualKeyframe` records to JSON/Parquet.
   - Unlink all image files in $W_k$.
4. **Guaranteed Bound**: Maximum disk usage is bounded by $\frac{\Delta t_W}{\Delta t_{\text{sample}}} \times \text{AvgImageSize} \approx 30 \times 1.2\text{ MB} \approx 36\text{ MB}$, independent of total stream length $T$.

---

## 5. Verification & Quality Gates
1. **Zero VRAM Leakage Verification**:
   - Measure VRAM in main process before and after launching isolated worker tasks.
   - Confirm baseline VRAM returns to 0 MB (or initial idle) upon worker process exit.
2. **Subprocess Resilience & Timeout Handling**:
   - Supervise worker execution with configurable timeout.
   - Test worker crash / error propagation with proper exit codes and error envelopes.
3. **Bounded Disk Usage Verification**:
   - Verify that all temporary frame files in each window are purged and unlinked.
4. **Full Test Suite & Benchmark**:
   - All tests pass with 0 regressions.
