"""Multimodal Scrubber, Video Playback & Timeline Routes (Spec 27)."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse

from stream_fusion.models.schemas import ScrubberTimelinePayload

router = APIRouter(prefix="/api/vods", tags=["Player & Timeline"])


def _load_manifest_for_vod(catalog: Any, vod_id: str, homelab_root: str) -> Optional[Dict[str, Any]]:
    """Attempts to find and load the FullSpectrumManifest JSON for a given VOD."""
    # Check if vod has metadata_path in catalog
    vod = catalog.get_harvested_vod(vod_id)
    if vod and vod.metadata_path and Path(vod.metadata_path).exists():
        try:
            with open(vod.metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # Fallback: check homelab_root/analyzed/<vod_id>/manifest.json
    cand = Path(homelab_root) / "analyzed" / vod_id / "manifest.json"
    if cand.exists():
        try:
            with open(cand, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


@router.get("/{vod_id}/timeline", response_model=ScrubberTimelinePayload)
def get_vod_timeline(vod_id: str, request: Request) -> ScrubberTimelinePayload:
    """Aggregates all multimodal analytical tracks for synchronized player scrubbing."""
    catalog = getattr(request.app.state, "catalog", None)
    homelab_root = getattr(request.app.state, "homelab_root", "./homelab_storage")
    if not catalog:
        raise HTTPException(status_code=500, detail="Catalog database is not initialized.")

    vod = catalog.get_harvested_vod(vod_id)
    if not vod:
        raise HTTPException(status_code=404, detail=f"VOD '{vod_id}' not found.")

    manifest = _load_manifest_for_vod(catalog, vod_id, homelab_root)

    # Initialize empty track containers
    scenes: List[Dict[str, Any]] = []
    speech_segments: List[Dict[str, Any]] = []
    chat_bursts: List[Dict[str, Any]] = []
    sponsors: List[Dict[str, Any]] = []
    claims: List[Dict[str, Any]] = []
    waveform: List[float] = []
    keyframes: List[Dict[str, Any]] = []
    slices_count = 0

    if manifest:
        # Load from stage outputs recorded in manifest
        stage_outputs = manifest.get("stage_outputs", {})
        scenes = stage_outputs.get("scenes", [])
        speech_segments = stage_outputs.get("speech_segments", [])
        chat_bursts = stage_outputs.get("meme_bursts", [])
        sponsors = stage_outputs.get("sponsor_segments", [])
        claims = stage_outputs.get("claims", [])
        keyframes = stage_outputs.get("keyframes", [])
        waveform = stage_outputs.get("audio_waveform", [])
        slices_count = manifest.get("total_slices", 0)

    # If no analyzed manifest, build a default timeline based on VOD metadata
    duration = vod.duration_sec if vod.duration_sec > 0 else 60.0
    if not waveform:
        # Generate 64 normalized amplitude values for waveform track visualization
        import math
        waveform = [round(abs(math.sin(i * 0.25) * 0.8 + 0.2), 2) for i in range(64)]

    video_url = f"/api/media/{vod_id}/video" if vod.video_path and Path(vod.video_path).exists() else None

    return ScrubberTimelinePayload(
        vod_id=vod.vod_id,
        title=vod.title,
        streamer_id=vod.streamer_id,
        duration_sec=duration,
        video_url=video_url,
        slices_count=slices_count,
        scenes=scenes,
        speech_segments=speech_segments,
        chat_bursts=chat_bursts,
        sponsors=sponsors,
        claims=claims,
        waveform=waveform,
        keyframes=keyframes,
    )


@router.get("/{vod_id}/chat")
def get_vod_chat_messages(
    vod_id: str,
    start_sec: float = Query(0.0, ge=0.0),
    end_sec: Optional[float] = Query(None, ge=0.0),
    limit: int = Query(200, ge=1, le=1000),
    request: Request = None,
) -> Dict[str, Any]:
    """Retrieves chat messages for a VOD within an optional time offset window."""
    catalog = getattr(request.app.state, "catalog", None)
    if not catalog:
        raise HTTPException(status_code=500, detail="Catalog database is not initialized.")

    vod = catalog.get_harvested_vod(vod_id)
    if not vod:
        raise HTTPException(status_code=404, detail=f"VOD '{vod_id}' not found.")

    if not vod.chat_path or not Path(vod.chat_path).exists():
        return {"vod_id": vod_id, "messages": [], "count": 0}

    messages = []
    try:
        with open(vod.chat_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            raw_list = data if isinstance(data, list) else data.get("messages", [])
            for item in raw_list:
                ts = float(item.get("timestamp_offset", item.get("time_in_seconds", 0.0)))
                if ts >= start_sec and (end_sec is None or ts <= end_sec):
                    messages.append(item)
                    if len(messages) >= limit:
                        break
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read chat log: {e}")

    return {
        "vod_id": vod_id,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "count": len(messages),
        "messages": messages,
    }


@router.get("/{vod_id}/slices")
def get_vod_slices(
    vod_id: str,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    request: Request = None,
) -> Dict[str, Any]:
    """Retrieves 1-second synchronized multimodal FusionSlice records."""
    catalog = getattr(request.app.state, "catalog", None)
    homelab_root = getattr(request.app.state, "homelab_root", "./homelab_storage")
    manifest = _load_manifest_for_vod(catalog, vod_id, homelab_root)
    if not manifest:
        return {"vod_id": vod_id, "slices": [], "total_slices": 0}

    parquet_path = manifest.get("slices_parquet_path")
    slices: List[Dict[str, Any]] = []
    total = 0

    if parquet_path and Path(parquet_path).exists():
        try:
            import pandas as pd
            df = pd.read_parquet(parquet_path)
            total = len(df)
            sliced = df.iloc[offset : offset + limit]
            slices = sliced.to_dict(orient="records")
        except Exception:
            pass

    return {
        "vod_id": vod_id,
        "offset": offset,
        "limit": limit,
        "total_slices": total,
        "slices": slices,
    }


@router.get("/media/{vod_id}/video")
def stream_vod_video(vod_id: str, request: Request):
    """Streams local MP4 video file with HTTP range headers."""
    catalog = getattr(request.app.state, "catalog", None)
    if not catalog:
        raise HTTPException(status_code=500, detail="Catalog database is not initialized.")

    vod = catalog.get_harvested_vod(vod_id)
    if not vod or not vod.video_path or not Path(vod.video_path).exists():
        raise HTTPException(status_code=404, detail=f"Video file for VOD '{vod_id}' not found on disk.")

    return FileResponse(
        path=vod.video_path,
        media_type="video/mp4",
        filename=Path(vod.video_path).name,
    )
