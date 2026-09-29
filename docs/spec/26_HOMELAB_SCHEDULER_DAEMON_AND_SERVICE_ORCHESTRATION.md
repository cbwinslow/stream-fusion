# Spec 26: Homelab 24/7 Scheduler Daemon & Service Orchestration

**Status**: Complete & Verified (2026-09-29)  
**Implementation Modules**: `src/stream_fusion/harvester/scheduler.py`, `src/stream_fusion/harvester/daemon.py`, `deploy/`  
**Test Suite**: `tests/test_homelab_scheduler_daemon.py` (8/8 passing, 197/197 overall test suite green)  
**Registry Schemas**: `DaemonConfig`, `DaemonStatusReport`, `DaemonJobRecord`, `DaemonState` exported to `docs/schemas/`  

---

## 1. Executive Summary & Objective

Spec 25 delivered the targeted streamer roster and homelab storage harvester engine (`HomelabHarvester`), dual-backend catalog (`HarvestCatalog`), and the pipeline bridge (`ingest_to_pipeline`). However, operating StreamFusion continuously on a dedicated homelab server or NAS requires an autonomous, self-contained **24/7 Scheduler Daemon & Service Orchestrator**.

This subsystem turns StreamFusion into an "always-on" background service that:
1. **Cron & Periodic Scheduling**:
   - Automatically crawls target streamer channels on configurable intervals (e.g. every 30 minutes) to discover newly published VODs and queue them.
   - Polls the download queue at high frequency (e.g. every 15 seconds) to dispatch multi-threaded downloads.
   - Automatically triggers the 9-Phase `FullSpectrumPipeline` when VODs complete downloading (`auto_analyze=True`), managing single-concurrency pipeline execution to eliminate GPU VRAM / CPU thrashing.
2. **Robust Lifecycle & Graceful Shutdown**:
   - Handles OS termination signals (`SIGINT`, `SIGTERM`, Windows console control events) cooperatively using thread cancellation tokens.
   - Ensures active downloads and active pipeline runs cleanly finalize or checkpoint without leaving orphaned temporary files or corrupting SQLite/PostgreSQL databases.
   - Manages an atomic PID file (`daemon.pid`) to prevent multiple conflicting daemon instances from conflicting over the storage directory.
3. **Telemetry, Heartbeat & Remote IPC**:
   - Emits structured heartbeat reports (`DaemonStatusReport`) detailing uptime, queue depths, active download bandwidth, active pipeline stage, free disk space, and recent errors.
   - Supports remote control via JSON-RPC 2.0 (`streamfusion.daemon.*`) and CLI (`streamfusion daemon *`) to inspect status, trigger instant crawls, pause, resume, and gracefully stop.
4. **Housekeeping & Retention Policy**:
   - Optionally enforces retention rules (e.g. purge or archive multi-gigabyte raw video files after 14 days) while permanently preserving extracted multimodal Parquet matrices, JSON manifests, and 9:16 vertical shorts.
5. **Self-Contained Project Packaging**:
   - All code, scripts, configuration templates, and orchestration files reside strictly inside the project root (`./config/`, `./deploy/`, `./logs/`, `./homelab_storage/`).
   - Includes cross-platform service runners: Docker Compose (`deploy/docker-compose.homelab.yml`), Linux systemd (`deploy/systemd/streamfusion-daemon.service`), and Windows scheduled tasks (`deploy/windows/run_daemon.bat`, `deploy/windows/install_scheduled_task.ps1`).

```
+-----------------------------------------------------------------------------------------+
|                           StreamFusion Homelab Daemon Supervisor                        |
|                                                                                         |
|  [PID Lock File: ./daemon.pid]        [Rotating File Logger: ./logs/daemon.log]         |
|  [State Machine: STOPPED -> STARTING -> RUNNING <-> PAUSED -> DRAINING -> STOPPED]      |
+-----------------------------------------------------------------------------------------+
                                             │
                      ┌──────────────────────┼──────────────────────┐
                      ▼                      ▼                      ▼
+---------------------------+ +----------------------------+ +----------------------------+
|   Periodic Channel Crawl  | |    Download Queue Worker   | |    Auto-Pipeline Runner    |
| • Polls Roster (30m)      | | • Multi-threaded (yt-dlp)  | | • Isolated Concurrency (1) |
| • Discovers New VODs      | | • Per-streamer rate limits | | • 9-Phase Full Spectrum    |
| • Queues into Catalog DB  | | • Disk space threshold     | | • Updates status: ANALYZED |
+---------------------------+ +----------------------------+ +----------------------------+
                      │                      │                      │
                      └──────────────────────┼──────────────────────┘
                                             ▼
+-----------------------------------------------------------------------------------------+
|                                Homelab Storage & Database                               |
|   • SQLite (local) or PostgreSQL (clustered)                                            |
|   • Storage: ./homelab_storage/vods/<streamer_id>/<vod_id>/                             |
|   • Retention Housekeeper: optional pruning of raw video after N days                   |
+-----------------------------------------------------------------------------------------+
                                             │
                                             ▼
+-----------------------------------------------------------------------------------------+
|                                Remote Control & Inspection                              |
|   • CLI: streamfusion daemon {run|status|stop|pause|resume|trigger-crawl}               |
|   • JSON-RPC: streamfusion.daemon.getStatus / triggerCrawl / pause / resume             |
|   • Service manifests: Docker Compose, Systemd unit, Windows Task Scheduler             |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Core Architectural Components

### 2.1 Daemon Configuration (`DaemonConfig`)
Configurable via YAML/JSON file (`config/daemon.yaml`) or CLI overrides:

```python
class DaemonConfig(BaseModel):
    crawl_interval_minutes: int = 30
    download_poll_interval_seconds: int = 15
    auto_analyze: bool = True
    max_concurrent_downloads: int = 2
    max_concurrent_pipelines: int = 1
    min_free_disk_gb: float = 50.0
    retention_days: Optional[int] = None
    pid_file: str = "daemon.pid"
    log_file: str = "logs/daemon.log"
    homelab_root: str = "./homelab_storage"
    catalog_db_url: str = "sqlite:///homelab_storage/catalog.db"
    enable_shorts: bool = True
    enable_web_grounding: bool = True
    enable_adaptive_slang: bool = True
    enable_sponsor_quantifier: bool = True
    dry_run_shorts: bool = False
```

### 2.2 Telemetry & State Tracking (`DaemonStatusReport`)
```python
class DaemonState(str, Enum):
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    DRAINING = "DRAINING"
    ERROR = "ERROR"

class DaemonStatusReport(BaseModel):
    state: DaemonState
    pid: Optional[int]
    uptime_seconds: float
    started_at: Optional[datetime]
    last_crawl_at: Optional[datetime]
    next_crawl_at: Optional[datetime]
    active_downloads: List[Dict[str, Any]]
    active_pipeline_vod: Optional[str]
    queued_vods_count: int
    harvested_vods_count: int
    analyzed_vods_count: int
    error_vods_count: int
    free_disk_gb: float
    recent_errors: List[str]
```

### 2.3 Scheduler Tasks & Execution Cycle
The scheduler loop runs asynchronously or across coordinated background threads:
- **`_crawl_task`**: Triggers `ChannelVodCrawler.sync_all(catalog, auto_queue=True)` at every `crawl_interval_minutes` or on-demand.
- **`_download_task`**: Calls `HomelabHarvester.process_queue()` periodically if there are queued VODs and active workers < limit.
- **`_pipeline_task`**: If `auto_analyze=True` and no pipeline is currently running, queries the catalog for VODs in `HARVESTED` or `READY_FOR_ANALYSIS` and executes `ingest_to_pipeline()`.
- **`_housekeeping_task`**: If `retention_days` is configured, periodically removes `media.mp4` for VODs analyzed older than `retention_days`.

### 2.4 Self-Contained Deployment Files
- `deploy/docker-compose.homelab.yml`: Standalone compose environment with PostgreSQL and StreamFusion worker daemon volume-mounting project storage.
- `deploy/systemd/streamfusion-daemon.service`: Linux systemd service unit template.
- `deploy/windows/run_daemon.bat`: Simple launcher for Windows CMD/PowerShell.
- `deploy/windows/install_scheduled_task.ps1`: Automated PowerShell setup for Windows Task Scheduler to run unattended on boot.

---

## 3. Acceptance Criteria
1. `DaemonConfig`, `DaemonState`, and `DaemonStatusReport` registered in `SchemaRegistry` and exported to `docs/schemas/`.
2. `HomelabDaemon` starts, checks PID lock, runs scheduled crawls, downloads, and pipeline handoffs cooperatively.
3. Graceful shutdown halts all threads without data loss or stuck processes.
4. CLI commands (`streamfusion daemon run`, `status`, `stop`, `pause`, `resume`, `trigger-crawl`) function cleanly.
5. JSON-RPC endpoints (`streamfusion.daemon.*`) return valid JSON-RPC 2.0 payloads.
6. 100% test pass on `tests/test_homelab_scheduler_daemon.py` and overall project test suite.
