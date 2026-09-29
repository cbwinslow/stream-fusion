# Spec 29: Polyglot Data Stack (PostgreSQL 17, ClickHouse, Qdrant) & Decoupled Homelab Topology

**Status**: Ready for Implementation  
**Target Modules**:
- `src/stream_fusion/models/schemas.py`: Pydantic network configuration & connection strings.
- `src/stream_fusion/knowledge/search.py`: `QdrantVectorStorage(BaseVectorStorage)` adapter with Binary Quantization.
- `src/stream_fusion/knowledge/clickhouse.py`: `ClickHouseChatStorage` for high-throughput raw chat logs and `ASOF JOIN` cross-section analytics.
- `docker-compose.yml`: Reproducible container stack with Postgres 17, ClickHouse, Qdrant, and StreamFusion.
- `docs/homelab/bare_metal_setup.md`: Bare-metal Linux installation guide, systemd service units, and performance tuning.
- `tests/test_polyglot_data_stack.py`: Verification suite with zero-dependency mocking.

---

## 1. Executive Summary & Objective

In Specs 01 through 28, StreamFusion built an end-to-end multimodal livestream harvesting, transcription, chat alignment, and semantic RAG engine. However, as the system scales to **dozens of streamers 24/7** and accumulates **tens of millions of raw chat messages**, a single monolithic database creates severe bottlenecks:
- Traditional relational databases (PostgreSQL/SQLite) choke on 50M+ row aggregations and raw chat emote scans.
- Vector databases are terrible at relational ACID job queues and fine-grained state management.
- Analytical columnar databases (ClickHouse) excel at time-series and `ASOF JOIN`s, but lack ACID transaction locking and native mature conversational RAG vector indexes.

**Spec 29** establishes the **Polyglot Data Stack & Decoupled Homelab Architecture**:
1. **The Right Tool for the Right Data Model**:
   - **PostgreSQL 17**: ACID app state, streamer rosters, harvesting job queues (`FOR UPDATE SKIP LOCKED`), and claim graphs.
   - **ClickHouse**: Raw chat message stream (millions of rows), 10x-12x columnar compression, and native `ASOF JOIN` cross-section alignment.
   - **Qdrant (Rust)**: Multimodal semantic vectors with 32x Binary Quantization (BQ) and payload filtering on streamer/timestamp.
   - **Local NAS / Filesystem**: Raw VOD MP4s, segmented audio WAVs, and rendered video clips.
2. **Two-Machine Client-Server Topology**:
   - **Machine 1 (Homelab 24/7 Server)**: Hosts databases, bare-metal storage, and the harvesting daemon.
   - **Machine 2 (Client / GPU Workstation)**: Executes heavy CUDA inference (WhisperX, Florence-2 vision, OCR, FFmpeg short rendering) and serves the interactive Web Studio UI.
3. **Reproducibility & Dual Deployment**:
   - Bare-metal systemd setup for maximum bare-metal Linux performance.
   - Portable `docker-compose.yml` for instant reference reproduction.
   - 100% Pydantic/`.env` configurable network endpoints.

---

## 2. Decoupled Two-Machine Topology

```
┌─────────────────────────────────────────────────────────────────┐
│              MACHINE 1: HOMELAB DATA SERVER (24/7)               │
│                  (Debian/Ubuntu Bare-Metal / NAS)                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────────────────────┐  ┌───────────────────────────────┐  │
│  │   PostgreSQL 17        │  │     ClickHouse Server         │  │
│  │   Port: 5432           │  │     Port: 8123 (HTTP) / 9000  │  │
│  │   • Streamer Rosters   │  │     • Millions of Chat Rows   │  │
│  │   • Harvest Job Queues │  │     • Emote Velocity Counters │  │
│  │   • Scheduler Leases   │  │     • ASOF Cross-Stream Joins │  │
│  │   • Verified Claims    │  │     • 10x Columnar LZ4/ZSTD   │  │
│  └────────────────────────┘  └───────────────────────────────┘  │
│                                                                 │
│  ┌────────────────────────┐  ┌───────────────────────────────┐  │
│  │   Qdrant (Rust)        │  │     Homelab Storage / NAS     │  │
│  │   Port: 6333 (HTTP)    │  │     SMB / NFS / Local Mount   │  │
│  │   • Speech Vectors     │  │     • Raw VOD MP4 Files       │  │
│  │   • Chat Burst Vectors │  │     • Audio WAV Tracks        │  │
│  │   • 32x Binary Quant   │  │     • Rendered 9:16 Shorts    │  │
│  │   • Sub-10ms ANN Query │  │     • Keyframe Thumbnails     │  │
│  └────────────────────────┘  └───────────────────────────────┘  │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │   StreamFusion Harvester Daemon (Background Worker)       │  │
│  │   • 24/7 Live Stream Sniffing & yt-dlp Video Download     │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                ▲
                                │ 1GbE / 10GbE / ZeroTier LAN
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│           MACHINE 2: CLIENT / GPU WORKSTATION (Inference)       │
│                  (Windows 11 / Linux with NVIDIA RTX)           │
├─────────────────────────────────────────────────────────────────┤
│  • CUDA GPU Audio Extraction: Faster-Whisper / WhisperX         │
│  • Visual Understanding: Florence-2 / Qwen2-VL / Screen OCR     │
│  • Vector Embedding Generation: LocalEmbedder / Sentence-Trans  │
│  • Viral Video Production: FFmpeg 9:16 Auto-Crop & Subtitles    │
│  • StreamFusion Web Studio SPA: Interactive UI & RAG Chat       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Polyglot Data Model Division & Schemas

### 3.1 Model A: PostgreSQL 17 (Relational & ACID State)
- **Table `streamer_roster`**:
  ```sql
  CREATE TABLE IF NOT EXISTS streamer_roster (
      streamer_id VARCHAR(64) PRIMARY KEY,
      display_name VARCHAR(128) NOT NULL,
      twitch_channel VARCHAR(64),
      youtube_channel_id VARCHAR(64),
      kick_channel VARCHAR(64),
      priority_tier VARCHAR(16) DEFAULT 'standard',
      is_active BOOLEAN DEFAULT TRUE,
      created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
  );
  ```
- **Table `harvest_jobs`**:
  ```sql
  CREATE TABLE IF NOT EXISTS harvest_jobs (
      job_id VARCHAR(64) PRIMARY KEY,
      vod_id VARCHAR(128) NOT NULL,
      streamer_id VARCHAR(64) REFERENCES streamer_roster(streamer_id),
      status VARCHAR(32) NOT NULL DEFAULT 'QUEUED', -- QUEUED, DOWNLOADING, PROCESSING, COMPLETED, FAILED
      worker_id VARCHAR(64),
      leased_until TIMESTAMP WITH TIME ZONE,
      created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
      updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
  );
  ```

### 3.2 Model B: ClickHouse (High-Volume Raw Chat & ASOF Cross-Section)
- **Table `chat_events`**:
  ```sql
  CREATE TABLE IF NOT EXISTS chat_events (
      streamer_id LowCardinality(String),
      vod_id String,
      timestamp_ms UInt64,
      timestamp_sec Float32,
      chatter_username LowCardinality(String),
      message_text String,
      emotes Array(LowCardinality(String)),
      sentiment_score Float32,
      is_subscriber UInt8,
      burst_flag UInt8
  ) ENGINE = MergeTree()
  ORDER BY (streamer_id, vod_id, timestamp_ms)
  SETTINGS index_granularity = 8192;
  ```
- **Table `speech_segments` (For ASOF Correlation)**:
  ```sql
  CREATE TABLE IF NOT EXISTS speech_segments (
      streamer_id LowCardinality(String),
      vod_id String,
      start_ms UInt64,
      end_ms UInt64,
      speaker_id LowCardinality(String),
      transcript String
  ) ENGINE = MergeTree()
  ORDER BY (streamer_id, vod_id, start_ms);
  ```
- **The `ASOF JOIN` Cross-Section Query**:
  ```sql
  -- Align chat reactions happening within 0-8 seconds after a streamer speaks
  SELECT 
      s.vod_id,
      s.start_ms,
      s.transcript,
      c.chatter_username,
      c.message_text,
      c.emotes
  FROM speech_segments s
  ASOF LEFT JOIN chat_events c
    ON s.vod_id = c.vod_id 
   AND s.streamer_id = c.streamer_id
   AND s.start_ms <= c.timestamp_ms
  WHERE s.vod_id = {vod_id:String}
    AND c.timestamp_ms <= s.start_ms + 8000;
  ```

### 3.3 Model C: Qdrant (Multimodal Semantic Vectors with Binary Quantization)
- **Collection**: `stream_fusion_vectors`
- **Vector Parameters**:
  - Size: 128 (default) or 768 / 1024
  - Distance: Cosine
  - **Quantization**: Binary Quantization (BQ) enabled for 32x memory compression:
    ```json
    {
      "quantization_config": {
        "binary": {
          "always_ram": true
        }
      }
    }
    ```
- **Payload Schema**:
  ```json
  {
    "chunk_id": "vod123_speech_140_155",
    "vod_id": "vod123",
    "streamer_id": "asmongold",
    "content_type": "speech",
    "text": "The problem with Deadlock matchmaking is high MMR stacks...",
    "timestamp_sec": 142.5,
    "end_timestamp_sec": 157.0,
    "metadata": {
      "speaker_id": "SPEAKER_00",
      "chat_burst_dominant_emote": "KEKW"
    }
  }
  ```

---

## 4. Configuration Contracts (`DashboardConfig` & `.env`)

In [`src/stream_fusion/models/schemas.py`](file:///C:/Users/blain/Documents/stream-fusion/src/stream_fusion/models/schemas.py):
```python
class DashboardConfig(BaseModel):
    # Existing settings...
    homelab_root: str = Field(default="./homelab_archive")
    vector_db_url: Optional[str] = Field(default=None) # e.g. "qdrant://192.168.1.100:6333" or "postgresql://..."
    
    # Spec 29 Polyglot Stack Endpoints:
    postgres_url: Optional[str] = Field(
        default=None, 
        description="PostgreSQL 17 connection string (e.g. postgresql://user:pass@192.168.1.100:5432/streamfusion)"
    )
    clickhouse_url: Optional[str] = Field(
        default=None, 
        description="ClickHouse HTTP/native endpoint (e.g. http://192.168.1.100:8123)"
    )
    qdrant_url: Optional[str] = Field(
        default=None, 
        description="Qdrant REST/gRPC endpoint (e.g. http://192.168.1.100:6333)"
    )
    media_storage_path: Optional[str] = Field(
        default=None,
        description="NAS / SMB / Local filesystem mount for heavy video/audio blobs"
    )
```

---

## 5. Docker Compose Reference Deployment (`docker-compose.yml`)

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:17-alpine
    container_name: streamfusion-postgres
    restart: unless-stopped
    environment:
      POSTGRES_DB: streamfusion
      POSTGRES_USER: streamfusion
      POSTGRES_PASSWORD: streamfusion_secure_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  clickhouse:
    image: clickhouse/clickhouse-server:latest
    container_name: streamfusion-clickhouse
    restart: unless-stopped
    ports:
      - "8123:8123"  # HTTP interface
      - "9000:9000"  # Native interface
    volumes:
      - clickhouse_data:/var/lib/clickhouse
    ulimits:
      nofile:
        soft: 262144
        hard: 262144

  qdrant:
    image: qdrant/qdrant:latest
    container_name: streamfusion-qdrant
    restart: unless-stopped
    ports:
      - "6333:6333"  # REST API
      - "6334:6334"  # gRPC
    volumes:
      - qdrant_data:/qdrant/storage

  streamfusion-server:
    build: .
    container_name: streamfusion-server
    restart: unless-stopped
    depends_on:
      - postgres
      - clickhouse
      - qdrant
    environment:
      - POSTGRES_URL=postgresql://streamfusion:streamfusion_secure_password@postgres:5432/streamfusion
      - CLICKHOUSE_URL=http://clickhouse:8123
      - QDRANT_URL=http://qdrant:6333
      - HOMELAB_STORAGE_PATH=/data
    ports:
      - "8000:8000"
    volumes:
      - nas_media:/data

volumes:
  postgres_data:
  clickhouse_data:
  qdrant_data:
  nas_media:
```

---

## 6. Bare-Metal Homelab Installation (Native Linux Services)

For users who want pure bare-metal performance without Docker container virtualization:

1. **PostgreSQL 17**:
   ```bash
   sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
   wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
   sudo apt-get update && sudo apt-get install -y postgresql-17
   ```
2. **ClickHouse (Official single-binary installer)**:
   ```bash
   curl https://clickhouse.com/ | sh
   sudo ./clickhouse install
   sudo systemctl enable --now clickhouse-server
   ```
3. **Qdrant (Static Rust Binary)**:
   ```bash
   curl -sL https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-gnu.tar.gz | tar -xz
   sudo mv qdrant /usr/local/bin/
   # Run as systemd unit or direct background process
   ```

---

## 7. Verification & Zero-Dependency Testing Protocol

To ensure local tests remain fast and require zero external daemons:
1. `tests/test_polyglot_data_stack.py` uses lightweight mock/in-memory adapters when external network services are offline.
2. `create_vector_storage()` seamlessly supports `qdrant://`, `postgresql://`, `json://`, and fallback `sqlite://`.
3. The existing 220 tests across Specs 01–28 must continue to pass 100% green without regression.
