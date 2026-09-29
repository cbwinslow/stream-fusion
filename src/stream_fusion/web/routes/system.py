"""System Health, Telemetry & Storage Routes (Spec 27)."""

from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import time
from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, Request

from stream_fusion.models.schemas import (
    DaemonState,
    DashboardOverviewStats,
)

router = APIRouter(prefix="/api/system", tags=["System"])
_server_start_time = time.time()


@router.get("/health")
def get_system_health(request: Request) -> Dict[str, Any]:
    """Returns basic system health, server uptime, and software version."""
    uptime = time.time() - _server_start_time
    return {
        "status": "healthy",
        "service": "StreamFusion Studio",
        "version": "0.1.0",
        "uptime_seconds": round(uptime, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/storage")
def get_system_storage(request: Request) -> Dict[str, Any]:
    """Returns filesystem disk usage metrics for homelab storage root."""
    app_state = request.app.state
    root = Path(getattr(app_state, "homelab_root", "./homelab_storage")).resolve()
    root.mkdir(parents=True, exist_ok=True)

    usage = shutil.disk_usage(str(root))
    total_gb = usage.total / (1024 ** 3)
    used_gb = usage.used / (1024 ** 3)
    free_gb = usage.free / (1024 ** 3)
    percent_used = (usage.used / usage.total) * 100 if usage.total > 0 else 0.0

    return {
        "path": str(root),
        "total_gb": round(total_gb, 2),
        "used_gb": round(used_gb, 2),
        "free_gb": round(free_gb, 2),
        "percent_used": round(percent_used, 1),
    }


@router.get("/stats", response_model=DashboardOverviewStats)
def get_overview_stats(request: Request) -> DashboardOverviewStats:
    """Aggregates comprehensive dashboard stats across catalog, daemon, and pipelines."""
    app_state = request.app.state
    catalog = getattr(app_state, "catalog", None)
    daemon = getattr(app_state, "daemon", None)
    root = Path(getattr(app_state, "homelab_root", "./homelab_storage")).resolve()

    # Query catalog
    streamers_count = 0
    total_vods = 0
    harvested_vods = 0
    analyzed_vods = 0
    queued_vods = 0
    downloading_vods = 0

    if catalog:
        try:
            targets = catalog.list_streamer_targets()
            streamers_count = len(targets)
            stats = catalog.get_catalog_stats()
            total_vods = stats.get("total_vods", 0)
            status_counts = stats.get("status_counts", {})
            harvested_vods = status_counts.get("HARVESTED", 0)
            analyzed_vods = status_counts.get("ANALYZED", 0)
            queued_vods = status_counts.get("QUEUED", 0) + status_counts.get("DISCOVERED", 0)
            downloading_vods = status_counts.get("DOWNLOADING", 0)
        except Exception:
            pass

    # Storage calculation
    total_storage_bytes = 0
    free_gb = 0.0
    try:
        usage = shutil.disk_usage(str(root))
        total_storage_bytes = usage.used
        free_gb = usage.free / (1024 ** 3)
    except Exception:
        pass

    # Daemon state
    daemon_state = "STOPPED"
    daemon_pid = None
    if daemon:
        try:
            daemon_state = daemon.status.value if hasattr(daemon.status, "value") else str(daemon.status)
            report = daemon.get_status_report()
            daemon_pid = report.pid
        except Exception:
            pass

    # Live streams
    live_count = len(getattr(app_state, "active_live_streams", {}))

    return DashboardOverviewStats(
        streamer_count=streamers_count,
        total_vods_count=total_vods,
        harvested_vods_count=harvested_vods,
        analyzed_vods_count=analyzed_vods,
        queued_vods_count=queued_vods,
        downloading_vods_count=downloading_vods,
        total_storage_bytes=total_storage_bytes,
        free_storage_gb=round(free_gb, 2),
        daemon_state=daemon_state,
        daemon_pid=daemon_pid,
        active_live_streams=live_count,
        total_shorts_count=getattr(app_state, "total_shorts_count", 0),
        total_claims_count=getattr(app_state, "total_claims_count", 0),
        last_updated_at=datetime.now(timezone.utc),
    )
