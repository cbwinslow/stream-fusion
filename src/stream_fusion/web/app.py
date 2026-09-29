"""FastAPI Application Factory & Service Assembly (Spec 27)."""

from contextlib import asynccontextmanager
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from stream_fusion.harvester.catalog import HarvestCatalog
from stream_fusion.harvester.daemon import HomelabDaemon
from stream_fusion.models.schemas import DashboardConfig
from stream_fusion.monitoring.broadcaster import LiveEventBroadcaster
from stream_fusion.schema.agent_rpc import AgentRpcDispatcher
from stream_fusion.web.routes.catalog import router as catalog_router
from stream_fusion.web.routes.daemon import router as daemon_router
from stream_fusion.web.routes.knowledge import router as knowledge_router
from stream_fusion.web.routes.live import router as live_router
from stream_fusion.web.routes.player import router as player_router
from stream_fusion.web.routes.roster import router as roster_router
from stream_fusion.web.routes.rpc import router as rpc_router
from stream_fusion.web.routes.search import router as search_router
from stream_fusion.web.routes.shorts import router as shorts_router
from stream_fusion.web.routes.system import router as system_router
from stream_fusion.web.websockets import WebSocketHub

logger = logging.getLogger(__name__)


def create_app(
    config: Optional[DashboardConfig] = None,
    catalog: Optional[HarvestCatalog] = None,
    daemon: Optional[HomelabDaemon] = None,
    broadcaster: Optional[LiveEventBroadcaster] = None,
    search_engine: Optional[Any] = None,
) -> FastAPI:
    """Builds and configures the StreamFusion Web Dashboard & Studio FastAPI application."""
    config = config or DashboardConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup phase
        logger.info(f"StreamFusion Studio starting on {config.host}:{config.port}...")
        yield
        # Shutdown phase
        logger.info("StreamFusion Studio shutting down...")
        # Clean up any active live streams
        active_streams = getattr(app.state, "active_live_streams", {})
        for name, coord in list(active_streams.items()):
            try:
                coord.stop()
            except Exception:
                pass
        active_streams.clear()

    app = FastAPI(
        title="StreamFusion Studio",
        description="Unified Multimodal Livestream Intelligence & Autonomous Production Dashboard",
        version="0.1.0",
        lifespan=lifespan,
    )

    # State injection
    app.state.config = config
    app.state.homelab_root = config.homelab_root
    app.state.catalog = catalog or HarvestCatalog(database_url=config.catalog_db_url)
    app.state.daemon = daemon
    app.state.broadcaster = broadcaster or LiveEventBroadcaster()
    app.state.ws_hub = WebSocketHub(app.state.broadcaster)
    app.state.rpc_dispatcher = AgentRpcDispatcher()
    app.state.search_engine = search_engine
    app.state.active_live_streams = {}
    if daemon:
        app.state.rpc_dispatcher._active_daemon = daemon

    # CORS configuration
    if config.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Register API Routers
    app.include_router(system_router)
    app.include_router(roster_router)
    app.include_router(catalog_router)
    app.include_router(daemon_router)
    app.include_router(player_router)
    app.include_router(shorts_router)
    app.include_router(knowledge_router)
    app.include_router(search_router)
    app.include_router(live_router)
    app.include_router(rpc_router)

    # Direct media deep-link streaming route
    @app.get("/api/media/{vod_id}/video")
    def stream_direct_media_video(vod_id: str, request: Request):
        """Direct media endpoint supporting deep-linked video timestamps."""
        catalog = getattr(app.state, "catalog", None)
        if catalog:
            vod = catalog.get_harvested_vod(vod_id)
            if vod and vod.video_path and Path(vod.video_path).exists():
                return FileResponse(
                    path=vod.video_path,
                    media_type="video/mp4",
                    filename=Path(vod.video_path).name,
                )
        homelab_root = getattr(app.state, "homelab_root", "./homelab_storage")
        cand = Path(homelab_root) / "raw" / f"{vod_id}.mp4"
        if cand.exists():
            return FileResponse(path=str(cand), media_type="video/mp4", filename=cand.name)
        raise HTTPException(status_code=404, detail=f"Video for VOD '{vod_id}' not found.")

    # WebSocket Endpoints
    @app.websocket("/ws/live")
    async def websocket_live_endpoint(websocket: WebSocket):
        """Duplex WebSocket endpoint for real-time live events and chat streaming."""
        hub: WebSocketHub = app.state.ws_hub
        await hub.handle_client(websocket, event_types=["*"], replay_count=30)

    @app.websocket("/ws/events")
    async def websocket_events_endpoint(websocket: WebSocket):
        """Duplex WebSocket endpoint for pipeline progress and daemon heartbeat notifications."""
        hub: WebSocketHub = app.state.ws_hub
        await hub.handle_client(websocket, event_types=["*"], replay_count=10)

    # Static Assets & Frontend Single-Page Application
    static_dir = Path(config.static_dir) if config.static_dir else Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/", response_class=FileResponse)
        def serve_index_page():
            index_path = static_dir / "index.html"
            if index_path.exists():
                return FileResponse(str(index_path))
            return HTMLResponse("<h1>StreamFusion Studio</h1><p>UI asset index.html not found.</p>")
    else:
        @app.get("/", response_class=HTMLResponse)
        def fallback_index():
            return HTMLResponse("""
            <!DOCTYPE html>
            <html>
                <head><title>StreamFusion Studio</title></head>
                <body style="font-family:sans-serif; background:#0f172a; color:#f8fafc; padding:40px;">
                    <h1>StreamFusion Studio API is Running</h1>
                    <p>API documentation available at <a href="/docs" style="color:#38bdf8;">/docs</a></p>
                </body>
            </html>
            """)

    return app


def run_server(
    config: Optional[DashboardConfig] = None,
    catalog: Optional[HarvestCatalog] = None,
    daemon: Optional[HomelabDaemon] = None,
) -> None:
    """Launches uvicorn ASGI server to host StreamFusion Studio."""
    import uvicorn

    config = config or DashboardConfig()
    app = create_app(config=config, catalog=catalog, daemon=daemon)
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        reload=config.reload,
        log_level="info",
    )
