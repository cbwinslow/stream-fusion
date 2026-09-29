"""Daemon Supervisor & Scheduler Control Routes (Spec 27)."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, Request

from stream_fusion.models.schemas import (
    DaemonState,
    DaemonStatusReport,
    LiveStreamActionResponse,
)

router = APIRouter(prefix="/api/daemon", tags=["Daemon"])


@router.get("/status", response_model=DaemonStatusReport)
def get_daemon_status(request: Request) -> DaemonStatusReport:
    """Returns real-time status, health, and worker queue depths of the homelab daemon."""
    daemon = getattr(request.app.state, "daemon", None)
    if daemon:
        return daemon.get_status_report()

    # Fallback status report when daemon instance is not in-process
    catalog = getattr(request.app.state, "catalog", None)
    queued_count = 0
    downloading_count = 0
    harvested_count = 0
    analyzed_count = 0
    if catalog:
        try:
            stats = catalog.get_catalog_stats()
            sc = stats.get("status_counts", {})
            queued_count = sc.get("QUEUED", 0) + sc.get("DISCOVERED", 0)
            downloading_count = sc.get("DOWNLOADING", 0)
            harvested_count = sc.get("HARVESTED", 0)
            analyzed_count = sc.get("ANALYZED", 0)
        except Exception:
            pass

    return DaemonStatusReport(
        state=DaemonState.STOPPED,
        queued_vods_count=queued_count,
        downloading_vods_count=downloading_count,
        harvested_vods_count=harvested_count,
        analyzed_vods_count=analyzed_vods_count,
    )


@router.post("/start", response_model=LiveStreamActionResponse)
def start_daemon(request: Request) -> LiveStreamActionResponse:
    """Starts the 24/7 background scheduler daemon."""
    daemon = getattr(request.app.state, "daemon", None)
    if not daemon:
        raise HTTPException(
            status_code=503,
            detail="Daemon instance not configured on this server.",
        )

    if daemon.status == DaemonState.RUNNING:
        return LiveStreamActionResponse(
            status="WARNING",
            action="START_DAEMON",
            message="Daemon is already running.",
        )

    try:
        daemon.start(foreground=False)
        started = True
    except Exception as e:
        started = False

    return LiveStreamActionResponse(
        status="OK" if started else "ERROR",
        action="START_DAEMON",
        message="Daemon supervisor started successfully." if started else "Failed to start daemon supervisor.",
    )


@router.post("/pause", response_model=LiveStreamActionResponse)
def pause_daemon(request: Request) -> LiveStreamActionResponse:
    """Pauses daemon download and crawler workers."""
    daemon = getattr(request.app.state, "daemon", None)
    if not daemon:
        raise HTTPException(status_code=503, detail="Daemon instance not configured.")

    paused = daemon.pause()
    return LiveStreamActionResponse(
        status="OK" if paused else "WARNING",
        action="PAUSE_DAEMON",
        message="Daemon paused." if paused else "Daemon is not in RUNNING state.",
    )


@router.post("/resume", response_model=LiveStreamActionResponse)
def resume_daemon(request: Request) -> LiveStreamActionResponse:
    """Resumes paused daemon workers."""
    daemon = getattr(request.app.state, "daemon", None)
    if not daemon:
        raise HTTPException(status_code=503, detail="Daemon instance not configured.")

    resumed = daemon.resume()
    return LiveStreamActionResponse(
        status="OK" if resumed else "WARNING",
        action="RESUME_DAEMON",
        message="Daemon resumed." if resumed else "Daemon is not in PAUSED state.",
    )


@router.post("/stop", response_model=LiveStreamActionResponse)
def stop_daemon(request: Request) -> LiveStreamActionResponse:
    """Gracefully terminates daemon workers and releases PID lockfile."""
    daemon = getattr(request.app.state, "daemon", None)
    if not daemon:
        raise HTTPException(status_code=503, detail="Daemon instance not configured.")

    stopped = daemon.stop()
    return LiveStreamActionResponse(
        status="OK" if stopped else "WARNING",
        action="STOP_DAEMON",
        message="Daemon stopped." if stopped else "Daemon was not running.",
    )


@router.post("/crawl", response_model=LiveStreamActionResponse)
def trigger_daemon_crawl(request: Request) -> LiveStreamActionResponse:
    """Instructs daemon to execute an immediate channel crawl out-of-band."""
    daemon = getattr(request.app.state, "daemon", None)
    if not daemon:
        raise HTTPException(status_code=503, detail="Daemon instance not configured.")

    job = daemon.trigger_immediate_crawl()
    target_id = job.get("job_id") if isinstance(job, dict) else getattr(job, "job_id", None)
    return LiveStreamActionResponse(
        status="OK",
        action="TRIGGER_CRAWL",
        target_id=target_id,
        message="Immediate crawl dispatched to daemon supervisor.",
    )


@router.get("/logs")
def get_daemon_logs(
    lines: int = Query(100, ge=10, le=1000), request: Request = None
) -> Dict[str, Any]:
    """Reads tail of daemon log file."""
    log_file = Path(getattr(request.app.state, "daemon_log_file", "logs/daemon.log"))
    if not log_file.exists():
        return {"log_file": str(log_file), "lines": []}

    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
            tail = [line.rstrip() for line in all_lines[-lines:]]
            return {"log_file": str(log_file), "total_lines": len(all_lines), "lines": tail}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read daemon logs: {e}")
