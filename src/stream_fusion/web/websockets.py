"""WebSocket Feeds & Real-Time Event Hub (Spec 27)."""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Set
from fastapi import WebSocket, WebSocketDisconnect

from stream_fusion.monitoring.broadcaster import LiveEventBroadcaster
from stream_fusion.schema.envelope import StreamEventType, StreamFusionEnvelope

logger = logging.getLogger(__name__)


class WebSocketHub:
    """Manages active browser WebSocket connections and bridges to LiveEventBroadcaster."""

    def __init__(self, broadcaster: Optional[LiveEventBroadcaster] = None):
        self.broadcaster = broadcaster or LiveEventBroadcaster()
        self._active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    @property
    def connection_count(self) -> int:
        return len(self._active_connections)

    async def connect(self, websocket: WebSocket) -> None:
        """Accepts and registers incoming WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self._active_connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        """Removes a disconnected WebSocket."""
        async with self._lock:
            self._active_connections.discard(websocket)

    async def broadcast_json(self, message: Dict[str, Any]) -> None:
        """Sends a JSON message to all active WebSocket clients."""
        text = json.dumps(message)
        async with self._lock:
            dead = set()
            for ws in self._active_connections:
                try:
                    await ws.send_text(text)
                except Exception:
                    dead.add(ws)
            self._active_connections.difference_update(dead)

    async def handle_client(
        self,
        websocket: WebSocket,
        event_types: Optional[List[str]] = None,
        replay_count: int = 25,
    ) -> None:
        """Main bi-directional pump for a connected WebSocket client."""
        await self.connect(websocket)
        sub, client_queue = await self.broadcaster.register_client(
            transport="WEBSOCKET",
            event_types=event_types or ["*"],
            replay_count=replay_count,
        )

        async def send_pump():
            try:
                while True:
                    msg = await client_queue.get()
                    await websocket.send_text(msg)
                    client_queue.task_done()
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.debug(f"WebSocket client send loop terminated: {e}")

        async def receive_pump():
            try:
                while True:
                    data = await websocket.receive_text()
                    try:
                        parsed = json.loads(data)
                        action = parsed.get("action", "").lower()
                        if action == "ping":
                            pong = {
                                "event_type": "PONG",
                                "timestamp": asyncio.get_event_loop().time(),
                                "payload": {"message": "pong", "connections": self.connection_count},
                            }
                            await websocket.send_text(json.dumps(pong))
                        elif action == "subscribe":
                            new_types = parsed.get("event_types", ["*"])
                            sub.event_types = new_types
                            ack = {
                                "event_type": "SUBSCRIBED",
                                "payload": {"event_types": new_types},
                            }
                            await websocket.send_text(json.dumps(ack))
                    except json.JSONDecodeError:
                        pass
            except WebSocketDisconnect:
                pass
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.debug(f"WebSocket client receive loop terminated: {e}")

        sender_task = asyncio.create_task(send_pump())
        receiver_task = asyncio.create_task(receive_pump())

        try:
            done, pending = await asyncio.wait(
                [sender_task, receiver_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
        finally:
            await self.broadcaster.unregister_client(sub.client_id)
            await self.disconnect(websocket)
