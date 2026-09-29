"""Catalog, VODs, & Harvester Queue Routes (Spec 27)."""

import asyncio
from datetime import datetime, timezone
import threading
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request
from pydantic import BaseModel, Field

from stream_fusion.harvester.crawler import ChannelVodCrawler
from stream_fusion.models.schemas import (
    HarvestedVodRecord,
    HarvestStatus,
    LiveStreamActionResponse,
)

router = APIRouter(prefix="/api/catalog", tags=["Catalog"])


class EnqueueVodRequest(BaseModel):
    streamer_id: str
    vod_id: str
    platform: str = "twitch"
    title: str = "Manual Ingestion"
    url: Optional[str] = None
    duration_sec: float = 0.0


@router.get("", response_model=List[HarvestedVodRecord])
def list_harvested_vods(
    streamer_id: Optional[str] = Query(None, description="Filter by streamer ID"),
    status: Optional[str] = Query(None, description="Filter by status (e.g. QUEUED, HARVESTED, ANALYZED)"),
    search: Optional[str] = Query(None, description="Keyword search in title"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    request: Request = None,
) -> List[HarvestedVodRecord]:
    """Retrieves harvested VODs from catalog with filtering and search."""
    catalog = getattr(request.app.state, "catalog", None)
    if not catalog:
        raise HTTPException(status_code=500, detail="Catalog database is not initialized.")

    if search:
        return catalog.search_vods(query=search, streamer_id=streamer_id, limit=limit)

    st_enum = None
    if status:
        try:
            st_enum = HarvestStatus(status.upper())
        except ValueError:
            pass

    return catalog.list_harvested_vods(
        streamer_id=streamer_id,
        status=st_enum,
        limit=limit,
        offset=offset,
    )


@router.get("/{vod_id}", response_model=HarvestedVodRecord)
def get_harvested_vod(vod_id: str, request: Request) -> HarvestedVodRecord:
    """Retrieves details of a single harvested VOD."""
    catalog = getattr(request.app.state, "catalog", None)
    if not catalog:
        raise HTTPException(status_code=500, detail="Catalog database is not initialized.")

    vod = catalog.get_harvested_vod(vod_id)
    if not vod:
        raise HTTPException(status_code=404, detail=f"VOD '{vod_id}' not found in catalog.")
    return vod


@router.post("/queue", response_model=HarvestedVodRecord)
def enqueue_vod(req: EnqueueVodRequest, request: Request) -> HarvestedVodRecord:
    """Manually enqueues a VOD into the download catalog."""
    catalog = getattr(request.app.state, "catalog", None)
    if not catalog:
        raise HTTPException(status_code=500, detail="Catalog database is not initialized.")

    # Ensure streamer target exists
    target = catalog.get_streamer_target(req.streamer_id)
    if not target:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot enqueue VOD: streamer target '{req.streamer_id}' does not exist.",
        )

    record = HarvestedVodRecord(
        vod_id=req.vod_id,
        streamer_id=req.streamer_id,
        platform=req.platform,
        title=req.title,
        duration_sec=req.duration_sec,
        status=HarvestStatus.QUEUED,
        raw_metadata={"manual_url": req.url} if req.url else {},
    )
    catalog.upsert_harvested_vod(record)
    return record


@router.post("/crawl")
def trigger_channel_crawl(
    streamer_id: Optional[str] = None, request: Request = None
) -> Dict[str, Any]:
    """Triggers an immediate channel crawl to discover new VODs."""
    catalog = getattr(request.app.state, "catalog", None)
    crawler = getattr(request.app.state, "crawler", None) or ChannelVodCrawler()
    if not catalog:
        raise HTTPException(status_code=500, detail="Catalog database is not initialized.")

    if streamer_id:
        target = catalog.get_streamer_target(streamer_id)
        if not target:
            raise HTTPException(status_code=404, detail=f"Streamer '{streamer_id}' not found.")
        new_vods = crawler.sync_streamer(target, catalog=catalog, auto_queue=True)
        return {
            "status": "OK",
            "streamer_id": streamer_id,
            "discovered_vods": len(new_vods),
            "vod_ids": [v.vod_id for v in new_vods],
        }

    results = crawler.sync_all(catalog=catalog, auto_queue=True)
    total = sum(len(v) for v in results.values())
    return {
        "status": "OK",
        "total_discovered": total,
        "details": {s_id: len(v) for s_id, v in results.items()},
    }


def _run_pipeline_worker(catalog: Any, vod_id: str, homelab_root: str):
    """Background worker executing the FullSpectrumPipeline on a VOD."""
    from stream_fusion.harvester.bridge import ingest_to_pipeline
    from stream_fusion.models.schemas import FullSpectrumConfig

    vod = catalog.get_harvested_vod(vod_id)
    if not vod or not vod.video_path:
        catalog.update_vod_status(vod_id, HarvestStatus.FAILED, error_message="Missing video_path for analysis")
        return

    catalog.update_vod_status(vod_id, HarvestStatus.ANALYZING)
    cfg = FullSpectrumConfig(
        enable_shorts=True,
        enable_web_grounding=True,
        enable_adaptive_slang=True,
        enable_sponsor_quantifier=True,
        dry_run_shorts=True,
    )
    try:
        manifest = ingest_to_pipeline(vod=vod, config=cfg, output_dir=f"{homelab_root}/analyzed/{vod_id}")
        catalog.update_vod_status(
            vod_id,
            HarvestStatus.ANALYZED,
            metadata_path=manifest.manifest_path,
        )
    except Exception as e:
        catalog.update_vod_status(vod_id, HarvestStatus.FAILED, error_message=str(e))


@router.post("/analyze/{vod_id}", response_model=LiveStreamActionResponse)
def trigger_vod_analysis(
    vod_id: str, background_tasks: BackgroundTasks, request: Request
) -> LiveStreamActionResponse:
    """Dispatches the FullSpectrumPipeline for a harvested VOD in the background."""
    catalog = getattr(request.app.state, "catalog", None)
    if not catalog:
        raise HTTPException(status_code=500, detail="Catalog database is not initialized.")

    vod = catalog.get_harvested_vod(vod_id)
    if not vod:
        raise HTTPException(status_code=404, detail=f"VOD '{vod_id}' not found.")

    if vod.status in (HarvestStatus.ANALYZING,):
        return LiveStreamActionResponse(
            status="WARNING",
            action="ANALYZE",
            target_id=vod_id,
            message="VOD analysis is already in progress.",
        )

    root = getattr(request.app.state, "homelab_root", "./homelab_storage")
    background_tasks.add_task(_run_pipeline_worker, catalog, vod_id, root)

    return LiveStreamActionResponse(
        status="OK",
        action="ANALYZE",
        target_id=vod_id,
        message=f"FullSpectrumPipeline analysis scheduled for VOD '{vod_id}'.",
        details={"streamer_id": vod.streamer_id, "title": vod.title},
    )
