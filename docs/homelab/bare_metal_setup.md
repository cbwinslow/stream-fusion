# StreamFusion Bare-Metal Homelab Installation & Deployment Guide

This guide details the bare-metal Linux installation, configuration, and systemd service orchestration for the **StreamFusion Polyglot Data Stack** (Spec 29).

---

## 1. Decoupled Two-Machine Topology

StreamFusion separates 24/7 background harvesting and analytical ingestion from heavy GPU inference:

```
┌─────────────────────────────────────────────────────────────────┐
│              MACHINE 1: HOMELAB DATA SERVER (24/7)               │
│                  (Debian/Ubuntu Bare-Metal / NAS)                │
├─────────────────────────────────────────────────────────────────┤
│  • PostgreSQL 17 (Port 5432): Relational state & harvest jobs   │
│  • ClickHouse (Ports 8123/9000): Raw chat streams & ASOF joins  │
│  • Qdrant (Port 6333): 32x Binary Quantized semantic vectors    │
│  • Local Storage / NAS: Raw VODs, WAV tracks, clips             │
│  • StreamFusion Harvester Daemon: 24/7 Livestream listener      │
└─────────────────────────────────────────────────────────────────┘
                                ▲
                                │ 1GbE / 10GbE / ZeroTier LAN
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│           MACHINE 2: CLIENT / GPU WORKSTATION (Inference)       │
│                  (Windows 11 / Linux with NVIDIA RTX)           │
├─────────────────────────────────────────────────────────────────┤
│  • WhisperX / Faster-Whisper CUDA Speech Transcription          │
│  • Florence-2 / Qwen2-VL Screen & Visual Comprehension          │
│  • Screen OCR & Gamer Handle Recognition                        │
│  • 9:16 Short Rendering & Subtitle Generation (FFmpeg)          │
│  • StreamFusion Studio SPA Dashboard                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Port Reference Table

| Service | Port | Protocol | Purpose |
|---|---|---|---|
| **PostgreSQL 17** | `5434` (or `5432` default) | TCP | Relational state, streamer roster, job leases (`FOR UPDATE SKIP LOCKED`) *(Cluster `17-main` uses `5434` when coexisting with `16-main` on `5432`)* |
| **ClickHouse HTTP** | `8123` | HTTP | High-throughput batch chat ingestion & ASOF joins |
| **ClickHouse Native** | `9000` | TCP | Native ClickHouse binary interface |
| **Qdrant REST** | `6333` | HTTP | Vector index, Binary Quantization search & payload filtering |
| **Qdrant gRPC** | `6334` | gRPC | High-performance binary vector streaming |
| **StreamFusion API** | `8000` | HTTP/WS | StreamFusion Web Studio & REST API |

> [!IMPORTANT]
> **Strict Bare-Metal Priority**: Primary production services run natively bare-metal on the homelab host. Docker containers associated with other applications (e.g. Langfuse, Grafana, custom web apps) MUST NEVER be repurposed or touched. The included `docker-compose.yml` is provided strictly as a clean, portable reference stack for external users or isolated testing environments.

---

## 3. Direct NIC / High-Speed Interconnect Topology

When running heavy video processing and vector indexing between a GPU workstation and a homelab storage/database server, a dedicated point-to-point NIC link (e.g. 1 Gbps / 10 Gbps) provides maximum throughput without saturating the home LAN:

- **Workstation Direct NIC (e.g., `wp2-eno2`)**: `192.168.10.2 / 24`
- **Homelab Server Direct NIC**: `192.168.10.1 / 24`
- **Storage Subsystem**: Dedicated partition with fluid multi-terabyte capacity (e.g. `/home/cbwinslow/workspace/streamfusion/vods`). Raw high-bitrate media files (1080p60 VODs, WAV tracks, demuxed keyframes) reside entirely on homelab storage, preserving 100% of local workstation SSD capacity.


## 3. Database Installation on Machine 1 (Debian / Ubuntu)

### 3.1 PostgreSQL 17 with pgvector

```bash
# 1. Add PostgreSQL official repository
sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
sudo apt-get update
sudo apt-get install -y postgresql-17 postgresql-17-pgvector

# 2. Configure user and database
sudo -u postgres psql -c "CREATE USER streamfusion WITH PASSWORD 'streamfusion_secure_password';"
sudo -u postgres psql -c "CREATE DATABASE streamfusion OWNER streamfusion;"
sudo -u postgres psql -d streamfusion -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 3. Allow LAN access (in /etc/postgresql/17/main/postgresql.conf)
# listen_addresses = '*'
# In /etc/postgresql/17/main/pg_hba.conf:
# host all streamfusion 192.168.1.0/24 md5

sudo systemctl restart postgresql
```

### 3.2 ClickHouse Server (Official Single-Binary Installer)

```bash
# 1. Download and run official quick installer
curl https://clickhouse.com/ | sh
sudo ./clickhouse install

# 2. Allow remote connections in /etc/clickhouse-server/config.xml
# Uncomment: <listen_host>0.0.0.0</listen_host>

# 3. Start and enable systemd service
sudo systemctl enable --now clickhouse-server

# 4. Verify ClickHouse HTTP port
curl "http://localhost:8123/ping"
# Output should be: Ok.
```

### 3.3 Qdrant (Rust Vector Database)

```bash
# 1. Download static release binary
curl -sL https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-gnu.tar.gz | tar -xz
sudo mv qdrant /usr/local/bin/qdrant
sudo chmod +x /usr/local/bin/qdrant

# 2. Create qdrant system user and data directory
sudo useradd -r -s /bin/false qdrant
sudo mkdir -p /var/lib/qdrant /etc/qdrant
sudo chown -R qdrant:qdrant /var/lib/qdrant
```

Create `/etc/systemd/system/qdrant.service`:

```ini
[Unit]
Description=Qdrant Vector Database
After=network.target

[Service]
Type=simple
User=qdrant
Group=qdrant
WorkingDirectory=/var/lib/qdrant
ExecStart=/usr/local/bin/qdrant
Restart=always
RestartSec=5
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

Enable and start Qdrant:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now qdrant
curl "http://localhost:6333/collections"
```

---

## 4. StreamFusion Harvester Daemon Service

Create `/etc/systemd/system/streamfusion-harvester.service`:

```ini
[Unit]
Description=StreamFusion 24/7 Homelab Harvester Daemon
After=network.target postgresql.service clickhouse-server.service qdrant.service

[Service]
Type=simple
User=streamfusion
Group=streamfusion
WorkingDirectory=/opt/stream-fusion
Environment="PATH=/opt/stream-fusion/.venv/bin:/usr/local/bin:/usr/bin"
Environment="POSTGRES_URL=postgresql://streamfusion:streamfusion_secure_password@localhost:5432/streamfusion"
Environment="CLICKHOUSE_URL=http://localhost:8123"
Environment="QDRANT_URL=http://localhost:6333"
Environment="HOMELAB_STORAGE_PATH=/mnt/storage/streamfusion"
ExecStart=/opt/stream-fusion/.venv/bin/stream-fusion daemon start --homelab-root /mnt/storage/streamfusion
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

---

## 5. Client / Workstation Configuration (`.env`)

On Machine 2 (GPU workstation running analysis and Studio web app), configure connection strings to point to Machine 1's LAN IP:

```bash
# Machine 1 LAN IP: 192.168.1.100
POSTGRES_URL="postgresql://streamfusion:streamfusion_secure_password@192.168.1.100:5432/streamfusion"
CLICKHOUSE_URL="http://192.168.1.100:8123"
QDRANT_URL="http://192.168.1.100:6333"
MEDIA_STORAGE_PATH="//192.168.1.100/streamfusion_media"
```

Launch the StreamFusion Studio SPA:
```bash
stream-fusion dashboard --port 8000
```
