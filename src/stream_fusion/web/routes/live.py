"""Live Tail Monitor & Real-Time Stream Ingestion Routes (Spec 27)."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from stream_fusion.models.schemas import (
    LivePlatform,
    LiveStreamActionResponse,
    LiveStreamConfig,
    LiveStreamStatus,
    LiveTailHealthMetrics,
)

router = APIRouter(prefix="/api/live", tags=["Live Monitor"])


class StartLiveTailRequest(BaseModel):
    channel_name: str
    platform: str = "twitch"
    buffer_duration_sec: float = 300.0
    auto_chat: bool = True


@router.get("/status")
def get_live_status(request: Request) -> Dict[str, Any]:
    """Retrieves health metrics and active live stream coordinators."""
    active_coordinators = getattr(request.app.state, "active_live_streams", {})
    channels = []

    for name, coord in active_coordinators.items():
        try:
            status = coord.get_status()
            channels.append({
                "channel_name": name,
                "platform": status.config.platform.value if hasattr(status.config.platform, "value") else str(status.config.platform),
                "state": status.state.value if hasattr(status.state, "value") else str(status.state),
                "health": status.health.model_dump() if hasattr(status.health, "model_dump") else {},
                "total_chat_messages": status.total_chat_messages,
                "meme_bursts_detected": status.meme_bursts_detected,
            })
        except Exception:
            channels.append({
                "channel_name": name,
                "platform": "twitch",
                "state": "RUNNING",
            })

    return {
        "active_streams_count": len(channels),
        "streams": channels,
    }


@router.post("/start", response_model=LiveStreamActionResponse)
async def start_live_tail(req: StartLiveTailRequest, request: Request) -> LiveStreamActionResponse:
    """Spawns a real-time live ingestion tailer and IRC chat listener."""
    app_state = request.app.state
    if not hasattr(app_state, "active_live_streams"):
        app_state.active_live_streams = {}

    chan = req.channel_name.lower()
    if chan in app_state.active_live_streams:
        return LiveStreamActionResponse(
            status="WARNING",
            action="START_LIVE_TAIL",
            target_id=chan,
            message=f"Live stream tailing already running for channel '{chan}'.",
        )

    # Ingest coordinator creation
    from stream_fusion.ingest.live_coordinator import LiveStreamCoordinator

    platform_enum = LivePlatform.TWITCH
    if req.platform.lower() == "youtube":
        platform_enum = LivePlatform.YOUTUBE
    elif req.platform.lower() == "kick":
        platform_enum = LivePlatform.KICK

    config = LiveStreamConfig(
        channel_name=chan,
        platform=platform_enum,
        buffer_duration_sec=req.buffer_duration_sec,
    )
    broadcaster = getattr(app_state, "broadcaster", None)
    coord = LiveStreamCoordinator(config=config, broadcaster=broadcaster)
    await coord.start()
    app_state.active_live_streams[chan] = coord

    return LiveStreamActionResponse(
        status="OK",
        action="START_LIVE_TAIL",
        target_id=chan,
        message=f"Real-time live tailing initiated for {req.platform.upper()} channel '{chan}'.",
        details={"channel_name": chan, "platform": req.platform},
    )


@router.post("/stop/{channel_name}", response_model=LiveStreamActionResponse)
async def stop_live_tail(channel_name: str, request: Request) -> LiveStreamActionResponse:
    """Terminates an active live stream ingestion coordinator."""
    app_state = request.app.state
    active_coordinators = getattr(app_state, "active_live_streams", {})
    chan = channel_name.lower()

    if chan not in active_coordinators:
        return LiveStreamActionResponse(
            status="WARNING",
            action="STOP_LIVE_TAIL",
            target_id=chan,
            message=f"No active live stream found for channel '{chan}'.",
        )

    coord = active_coordinators.pop(chan)
    try:
        await coord.stop()
    except Exception:
        pass

    return LiveStreamActionResponse(
        status="OK",
        action="STOP_LIVE_TAIL",
        target_id=chan,
        message=f"Live stream tailing terminated for channel '{chan}'.",
    )
