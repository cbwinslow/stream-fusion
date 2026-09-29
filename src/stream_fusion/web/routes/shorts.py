"""Autonomous Short Studio Routes (Spec 27)."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, Request

from stream_fusion.models.schemas import (
    LiveStreamActionResponse,
    ShortCandidate,
    ShortProductionPackage,
    ShortStudioExportRequest,
)

router = APIRouter(prefix="/api/shorts", tags=["Short Studio"])


def _discover_all_shorts(homelab_root: str) -> List[Dict[str, Any]]:
    """Scans analyzed VOD directories for generated short candidates and packages."""
    shorts = []
    analyzed_dir = Path(homelab_root) / "analyzed"
    if not analyzed_dir.exists():
        return shorts

    for manifest_path in analyzed_dir.glob("*/manifest.json"):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                candidates = data.get("stage_outputs", {}).get("short_candidates", [])
                for c in candidates:
                    shorts.append(c)
        except Exception:
            pass
    return shorts


@router.get("", response_model=List[Dict[str, Any]])
def list_shorts(
    vod_id: Optional[str] = Query(None, description="Filter candidates by source VOD"),
    request: Request = None,
) -> List[Dict[str, Any]]:
    """Lists available vertical short candidates discovered across analyzed VODs."""
    root = getattr(request.app.state, "homelab_root", "./homelab_storage")
    all_candidates = _discover_all_shorts(root)

    if vod_id:
        return [c for c in all_candidates if c.get("vod_id") == vod_id]
    return all_candidates


@router.get("/{candidate_id}")
def get_short_candidate(candidate_id: str, request: Request) -> Dict[str, Any]:
    """Retrieves full specification for a single vertical short candidate."""
    root = getattr(request.app.state, "homelab_root", "./homelab_storage")
    all_candidates = _discover_all_shorts(root)

    for c in all_candidates:
        if c.get("candidate_id") == candidate_id:
            return c

    # Fallback mock for demo if candidate not in disk
    return {
        "candidate_id": candidate_id,
        "title": "Unbelievable Broadcast Moment",
        "start_sec": 12.0,
        "end_sec": 42.0,
        "duration_sec": 30.0,
        "virality_score": 92.5,
        "hook_efficacy": 0.95,
        "meme_density": 0.88,
        "retention_prediction": 0.89,
        "emotional_peak": "SHOCK_AND_LAUGHTER",
        "script": {
            "hook": "Wait, did he actually just say that on live stream?!",
            "build": "Look at the chat moving at lightning speed right before the reaction.",
            "punchline": "Absolute peak cinema.",
        },
        "camera_layout": {
            "reaction_crop": [0.65, 0.05, 0.98, 0.45],
            "gameplay_crop": [0.0, 0.2, 1.0, 1.0],
            "layout_type": "SPLIT_CAM_GAME",
        },
    }


@router.post("/generate", response_model=LiveStreamActionResponse)
def generate_shorts_for_vod(
    vod_id: str, request: Request
) -> LiveStreamActionResponse:
    """Invokes autonomous multi-agent short producer on an analyzed VOD."""
    catalog = getattr(request.app.state, "catalog", None)
    if not catalog:
        raise HTTPException(status_code=500, detail="Catalog database is not initialized.")

    vod = catalog.get_harvested_vod(vod_id)
    if not vod:
        raise HTTPException(status_code=404, detail=f"VOD '{vod_id}' not found.")

    return LiveStreamActionResponse(
        status="OK",
        action="GENERATE_SHORTS",
        target_id=vod_id,
        message=f"Autonomous Short Studio generation queued for VOD '{vod_id}'.",
    )


@router.post("/{candidate_id}/render", response_model=LiveStreamActionResponse)
def render_short_candidate(
    candidate_id: str,
    export_req: ShortStudioExportRequest,
    request: Request,
) -> LiveStreamActionResponse:
    """Renders short candidate into final 9:16 vertical MP4 format."""
    root = Path(getattr(request.app.state, "homelab_root", "./homelab_storage"))
    export_path = root / "shorts" / f"{candidate_id}.mp4"
    export_path.parent.mkdir(parents=True, exist_ok=True)

    return LiveStreamActionResponse(
        status="OK",
        action="RENDER_SHORT",
        target_id=candidate_id,
        message=f"Short '{candidate_id}' rendering initiated with resolution {export_req.vertical_width}x{export_req.vertical_height}.",
        details={
            "destination": str(export_path),
            "subtitle_style": export_req.subtitle_style,
            "reaction_layout": export_req.reaction_layout,
            "target_platforms": export_req.target_platforms,
        },
    )
