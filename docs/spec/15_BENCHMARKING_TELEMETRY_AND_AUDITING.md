# StreamFusion: Continuous Benchmarking, Telemetry & Recursive Auditing Framework (Spec 15)

## 1. North Star & Objectives
To reliably scale StreamFusion across full-length 6–10 hour broadcasts and maintain production-grade performance, we require an automated, reusable telemetry, benchmarking, and auditing framework. This system monitors every execution, tracks pipeline speed and success rates, logs audit trails, detects regressions against historical baselines, and provides actionable feedback for continuous codebase improvement.

### Core Objectives
1. **Stage-by-Stage Precision Telemetry**:
   - Instrument all pipeline stages (demux, transcription, diarization, vision OCR, chat alignment, fusion matrix, video clipping).
   - Record wall-clock duration, CPU utilization, peak RAM, VRAM consumption, and throughput (Real-Time Factor, FPS, msgs/sec).
2. **Quality & Grounding Metrics**:
   - Track OCR density, chat latency calibration cross-correlation confidence, highlight agreement rate, and claim extraction yield.
3. **Persistent Audit Store & Historical Run Registry**:
   - Store every run's execution manifest, commit hash, configuration, and telemetry in an atomic SQLite/JSONL audit database (`audit.db`).
4. **Automated Regression & Anomaly Detector**:
   - Compare current runs against a designated baseline or rolling average ($N=5$).
   - Flag anomalies:
     * `SPEED_REGRESSION`: Stage duration $> 1.25 \times \text{baseline}$.
     * `MEMORY_LEAK_WARNING`: Peak memory $> 1.30 \times \text{baseline}$.
     * `QUALITY_DROP`: Metric decrease $> 15\%$ (e.g. fewer keyframe OCR blocks or calibration confidence drop).
5. **Actionable Feedback & Recursive Optimization Reports**:
   - Generate human-readable Markdown/HTML audit cards with bottleneck diagnosis and concrete code optimization recommendations.

---

## 2. Telemetry & Metrics Data Contracts

### 2.1 Stage Telemetry Schema (`StageTelemetry`)
```json
{
  "stage_name": "speech_transcription",
  "start_time_iso": "2026-09-27T16:10:00Z",
  "end_time_iso": "2026-09-27T16:10:08Z",
  "duration_sec": 8.12,
  "cpu_percent_peak": 45.2,
  "ram_mb_peak": 1240.5,
  "vram_mb_peak": 2350.0,
  "throughput_metric_name": "real_time_factor",
  "throughput_value": 0.135,
  "status": "SUCCESS",
  "error_message": null
}
```

### 2.2 Complete Audit Run Record (`AuditRunRecord`)
```json
{
  "run_id": "audit_20260927_161000",
  "git_commit": "6ffd370",
  "git_dirty": false,
  "timestamp": "2026-09-27T16:10:00Z",
  "stream_id": "asmon_sample_60s",
  "media_duration_sec": 60.0,
  "total_pipeline_duration_sec": 24.3,
  "overall_real_time_factor": 0.405,
  "stages": { ... },
  "quality_metrics": {
    "total_chat_messages": 736,
    "calibrated_latency_sec": 4.5,
    "ocr_blocks_extracted": 30,
    "claims_extracted": 2,
    "highlights_count": 1
  },
  "regressions_detected": [],
  "recommendations": []
}
```

---

## 3. Regression Detection Mathematics

For any metric $M$ in the current run $R$ compared to baseline $B$:

### 3.1 Relative Change Formulation
$$\Delta M = \frac{M_R - M_B}{M_B}$$

### 3.2 Regression Trigger Matrix
| Metric Type | Trigger Condition | Severity | Action |
|---|---|---|---|
| Stage Duration ($t_{\text{stage}}$) | $\Delta t > +25\%$ | `WARNING` | Log stage bottleneck recommendation |
| Total Pipeline Duration | $\Delta t_{\text{total}} > +30\%$ | `CRITICAL` | Flag performance regression |
| Peak RAM / VRAM | $\Delta \text{Mem} > +30\%$ | `WARNING` | Check unclosed tensors / leaked buffers |
| OCR / Extraction Yield | $\Delta \text{Yield} < -15\%$ | `WARNING` | Verify detector confidence threshold |
| Chat Calibration Confidence | $\Delta \text{Conf} < -20\%$ | `WARNING` | Check cross-correlation window drift |

---

## 4. Audit Engine Architecture

```
Execution Trigger (CLI / Pipeline)
  │
  ├──► Telemetry Collector (Scoped Context Managers)
  │      ├── Wall-Clock Timers (high-precision perf_counter)
  │      ├── Process Memory Poller (psutil RSS / VMS)
  │      └── Torch CUDA Memory Allocator (if CUDA available)
  │
  ├──► Quality Metric Aggregator
  │      └── Yield count, calibration confidence, highlight agreements
  │
  ├──► Baseline Comparator Engine
  │      ├── Loads active baseline from audit.db
  │      ├── Evaluates ΔM against threshold rules
  │      └── Synthesizes bottleneck diagnosis & recommendations
  │
  └──► Storage & Export
         ├── Appends record to SQLite audit.db & audit.jsonl
         └── Generates CLI Rich Table + Markdown Audit Feedback Card
```

---

## 5. Acceptance Criteria & Deliverables
* **Modules**:
  * `src/stream_fusion/monitoring/telemetry.py` (`TelemetryCollector`, `StageTimer`, `SystemResourceProbe`).
  * `src/stream_fusion/monitoring/audit.py` (`AuditStore`, `AuditRunRecord`, `RegressionComparator`).
* **CLI Commands**:
  * `streamfusion audit benchmark`: Runs benchmark suite on standard sample, recording telemetry.
  * `streamfusion audit history`: Lists past execution runs with throughput and status.
  * `streamfusion audit compare <run_id_a> <run_id_b>`: Displays diff table with regression flags.
* **Unit Tests**: `tests/test_audit_framework.py` verifying metric collection, persistence, and regression alerting.
