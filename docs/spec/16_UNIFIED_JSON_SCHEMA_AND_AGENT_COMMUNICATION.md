# StreamFusion: Unified JSON Schema & Agent Communication Protocol (Spec 16)

## 1. North Star & Architectural Objectives

As StreamFusion expands into multi-modal livestream analysis (speech, vision, chat, claims, sponsors, griefing detection, telemetry), both internal modules and external autonomous agents require an unambiguous, strictly validated data interchange standard.

### Core Objectives
1. **Universal JSON Schema Standards (Pydantic V2)**:
   - Single source of truth for all schemas across the pipeline.
   - Full JSON-Schema (Draft-07 / 2020-12) automatic generation for frontends, downstream agents, and external microservices.
   - CLI automated schema generation and validation (`streamfusion schema export`, `streamfusion schema validate`, `streamfusion schema list`).
2. **Unified Message Bus Envelope (`StreamFusionEnvelope`)**:
   - Universal event format for inter-module pub/sub, streaming event logs (NDJSON), and external message brokers.
   - Strict versioning (`"version": "1.0"`), unique `message_id`, UTC ISO-8601 timestamps, trace correlation IDs, explicit `event_type` enums, generic typed payload, and optional lightweight stage telemetry.
3. **Storage & Serialization Adapters**:
   - **JSONL Event Stream (`JsonlStreamAdapter`)**: Memory-bounded line-delimited streaming reader and writer with on-the-fly schema validation.
   - **SQLite JSON1 Store (`SqliteJsonStore`)**: Structured persistence with native SQLite JSON1 indices for rapid filtering on payload attributes (`json_extract(payload, '$.polarity')`).
   - **Parquet-JSON Bridge (`ParquetJsonBridge`)**: Seamless, lossless bidirectional conversion between columnar Parquet analytics matrices and schema-validated JSON records.
4. **Agentic Communication Protocol & RPC Hooks**:
   - JSON-RPC 2.0 compliant dispatcher (`AgentRpcDispatcher`) allowing autonomous agents (e.g. Antigravity subagents, LangGraph, CrewAI, AutoGen) to inspect, query, and filter pipeline states and stream results via stdin/stdout or CLI.
   - Standard query methods: `streamfusion.queryEvents`, `streamfusion.getStreamSummary`, `streamfusion.queryClaims`, `streamfusion.queryHighlights`, `streamfusion.queryChatters`.

---

## 2. Universal Envelope Data Contracts

### 2.1 Stream Event Types (`StreamEventType`)
```python
from enum import Enum

class StreamEventType(str, Enum):
    STAGE_START = "STAGE_START"
    STAGE_COMPLETE = "STAGE_COMPLETE"
    AUDIO_SEGMENT = "AUDIO_SEGMENT"
    KEYFRAME_ANALYSIS = "KEYFRAME_ANALYSIS"
    FUSION_SLICE = "FUSION_SLICE"
    CHAT_BURST = "CHAT_BURST"
    HIGHLIGHT_MOMENT = "HIGHLIGHT_MOMENT"
    SPONSOR_DETECTED = "SPONSOR_DETECTED"
    CLAIM_EXTRACTED = "CLAIM_EXTRACTED"
    SAFETY_ALERT = "SAFETY_ALERT"
    WEB_READ_ALONG = "WEB_READ_ALONG"
    AUDIT_BENCHMARK = "AUDIT_BENCHMARK"
    PIPELINE_COMPLETE = "PIPELINE_COMPLETE"
    ERROR = "ERROR"
```

### 2.2 Telemetry Metadata Header (`EnvelopeTelemetry`)
```json
{
  "duration_ms": 142.5,
  "stage": "chat_nlp",
  "ram_mb": 512.4,
  "vram_mb": 1024.0
}
```

### 2.3 Universal Envelope Contract (`StreamFusionEnvelope[T]`)
```json
{
  "version": "1.0",
  "message_id": "b3f892cb-2831-419b-a359-f81d599bdf91",
  "timestamp": "2026-09-28T06:30:00.123456Z",
  "stream_id": "asmon_sample_60s",
  "event_type": "HIGHLIGHT_MOMENT",
  "trace_id": "trace-77401-abc",
  "parent_id": null,
  "producer": "stream_fusion.export.clipper",
  "payload": {
    "start_sec": 12.0,
    "end_sec": 24.0,
    "highlight_score": 0.94,
    "reason": "Dramatic chat spike and streamer laughter"
  },
  "telemetry": {
    "duration_ms": 32.1,
    "stage": "clipper",
    "ram_mb": 420.0,
    "vram_mb": 0.0
  },
  "metadata": {
    "category": "reaction"
  }
}
```

---

## 3. Schema Registry Architecture

The `SchemaRegistry` maintains a catalog mapping event types and model names to their Pydantic classes:

| Schema Name | Target Class | Module |
|---|---|---|
| `ChatMessage` | `ChatMessage` | `stream_fusion.models.schemas` |
| `AudioSegment` | `AudioSegment` | `stream_fusion.models.schemas` |
| `VisualKeyframe` | `VisualKeyframe` | `stream_fusion.models.schemas` |
| `FusionSlice` | `FusionSlice` | `stream_fusion.models.schemas` |
| `StreamAnalysisResult` | `StreamAnalysisResult` | `stream_fusion.models.schemas` |
| `MemeBurstEvent` | `MemeBurstEvent` | `stream_fusion.models.schemas` |
| `ChatterProfile` | `ChatterProfile` | `stream_fusion.models.schemas` |
| `SponsorImpactReport` | `SponsorImpactReport` | `stream_fusion.models.schemas` |
| `StreamerClaim` | `StreamerClaim` | `stream_fusion.models.schemas` |
| `ChatterSafetyVerdict` | `ChatterSafetyVerdict` | `stream_fusion.models.schemas` |
| `ReadAlongSegment` | `ReadAlongSegment` | `stream_fusion.models.schemas` |
| `AuditRunRecord` | `AuditRunRecord` | `stream_fusion.monitoring.audit` |
| `StreamFusionEnvelope` | `StreamFusionEnvelope` | `stream_fusion.schema.envelope` |

### CLI Schema Commands:
- `streamfusion schema list`: Prints tabular overview of all registered schemas.
- `streamfusion schema export --out-dir docs/schemas/ [--format json|yaml]`: Exports standalone `.json` JSON-Schema definitions for each registered model.
- `streamfusion schema validate --model-name FusionSlice --file input.json`: Validates a local JSON document against the target schema.

---

## 4. Storage & Persistence Adapters

### 4.1 JsonlStreamAdapter
Streaming newline-delimited JSON reader/writer:
- Streaming generator: reads arbitrarily large files line-by-line with bounded memory ($O(1)$ RAM).
- Optional payload model deserialization and validation.
- Filtering by `event_type`, `stream_id`, or timestamp range.

### 4.2 SqliteJsonStore
Stores envelopes in a SQLite table:
```sql
CREATE TABLE IF NOT EXISTS stream_events (
    message_id TEXT PRIMARY KEY,
    version TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    stream_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    trace_id TEXT,
    producer TEXT,
    payload_json TEXT NOT NULL,
    telemetry_json TEXT,
    metadata_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_stream_events_stream_type ON stream_events(stream_id, event_type);
CREATE INDEX IF NOT EXISTS idx_stream_events_timestamp ON stream_events(timestamp);
```
Supports queries with JSON1 expressions:
```sql
SELECT message_id, payload_json FROM stream_events
WHERE stream_id = ? AND json_extract(payload_json, '$.stance') = 'APPROVAL';
```

### 4.3 ParquetJsonBridge
Lossless conversion between Parquet tabular data and validated `FusionSlice` / `AudioSegment` JSON envelopes.

---

## 5. Agentic Communication Protocol & RPC

Autonomous AI agents communicate with StreamFusion via standard JSON-RPC 2.0:
```json
{
  "jsonrpc": "2.0",
  "id": "agent-req-001",
  "method": "streamfusion.queryEvents",
  "params": {
    "stream_id": "asmon_sample_60s",
    "event_type": "CLAIM_EXTRACTED",
    "filters": {
      "stance": "APPROVAL"
    }
  }
}
```

Response format:
```json
{
  "jsonrpc": "2.0",
  "id": "agent-req-001",
  "result": {
    "count": 2,
    "events": [ ... ]
  }
}
```

CLI commands:
- `streamfusion agent rpc`: Interactive or piped JSON-RPC stdio server.
- `streamfusion agent query --db audit.db --stream-id asmon_sample_60s --event-type CLAIM_EXTRACTED`: Direct CLI agent query outputting formatted or pure JSON.

---

## 6. Verification & Quality Gates
1. Unit tests covering:
   - Universal envelope serialization and deserialization with generic payloads.
   - Schema registry listing, export, and validation against valid and malformed JSON.
   - JSONL stream writer and reader with chunking and filtering.
   - SQLite JSON store insertion and JSON1 attribute extraction queries.
   - Parquet to JSON envelope bridge.
   - Agent RPC dispatcher dispatching queries, reporting errors, and returning JSON-RPC 2.0 payloads.
   - CLI commands `streamfusion schema ...` and `streamfusion agent ...`.
2. Regression verification against existing 90-test test suite.
