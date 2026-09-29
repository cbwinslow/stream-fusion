"""Roster & Streamer Target Management Routes (Spec 27)."""

from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, Request

from stream_fusion.harvester.roster import RosterLoader
from stream_fusion.models.schemas import StreamerTargetRecord

router = APIRouter(prefix="/api/roster", tags=["Roster"])


@router.get("", response_model=List[StreamerTargetRecord])
def list_roster_streamers(
    enabled_only: bool = Query(False, description="Filter for enabled streamers only"),
    request: Request = None,
) -> List[StreamerTargetRecord]:
    """Retrieves all tracked streamer targets from the catalog."""
    catalog = getattr(request.app.state, "catalog", None)
    if not catalog:
        raise HTTPException(status_code=500, detail="Catalog database is not initialized.")

    return catalog.list_streamer_targets(enabled_only=enabled_only)


@router.get("/{streamer_id}", response_model=StreamerTargetRecord)
def get_roster_streamer(streamer_id: str, request: Request) -> StreamerTargetRecord:
    """Retrieves single streamer target by ID."""
    catalog = getattr(request.app.state, "catalog", None)
    if not catalog:
        raise HTTPException(status_code=500, detail="Catalog database is not initialized.")

    target = catalog.get_streamer_target(streamer_id)
    if not target:
        raise HTTPException(status_code=404, detail=f"Streamer target '{streamer_id}' not found.")
    return target


@router.post("", response_model=StreamerTargetRecord)
def upsert_roster_streamer(
    record: StreamerTargetRecord, request: Request
) -> StreamerTargetRecord:
    """Creates or updates a target streamer configuration."""
    catalog = getattr(request.app.state, "catalog", None)
    if not catalog:
        raise HTTPException(status_code=500, detail="Catalog database is not initialized.")

    catalog.upsert_streamer_target(record)
    return record


@router.delete("/{streamer_id}")
def delete_roster_streamer(streamer_id: str, request: Request) -> Dict[str, Any]:
    """Deletes a streamer target from the catalog."""
    catalog = getattr(request.app.state, "catalog", None)
    if not catalog:
        raise HTTPException(status_code=500, detail="Catalog database is not initialized.")

    deleted = catalog.delete_streamer_target(streamer_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Streamer target '{streamer_id}' not found.")
    return {"status": "OK", "streamer_id": streamer_id, "message": "Streamer target removed."}


@router.post("/sync")
def sync_roster_from_file(
    roster_path: Optional[str] = None, request: Request = None
) -> Dict[str, Any]:
    """Synchronizes catalog streamer targets with a local YAML roster configuration."""
    catalog = getattr(request.app.state, "catalog", None)
    if not catalog:
        raise HTTPException(status_code=500, detail="Catalog database is not initialized.")

    p = Path(roster_path or "config/roster.yaml")
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Roster config file not found: {p}")

    records = RosterLoader.load(p)
    synced = []
    for r in records:
        catalog.upsert_streamer_target(r)
        synced.append(r)

    return {
        "status": "OK",
        "synced_count": len(synced),
        "streamer_ids": [t.streamer_id for t in synced],
    }

