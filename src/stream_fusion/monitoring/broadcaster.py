"""Universal Live Event Broadcaster supporting WebSocket and SSE (Spec 19)."""

import asyncio
import base64
import hashlib
import json
import logging
import struct
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid

from stream_fusion.models.schemas import LiveClientSubscription
from stream_fusion.schema.envelope import StreamFusionEnvelope

logger = logging.getLogger(__name__)

WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class LiveEventBroadcaster:
    """Universal real-time event distribution bus.
    
    Broadcasts typed StreamFusionEnvelope instances over WebSocket and HTTP Server-Sent Events (SSE).
    Includes an in-memory rolling replay ring-buffer and per-client topic filtering.
    """

    def __init__(self, replay_capacity: int = 250):
        self.replay_capacity = replay_capacity
        self._replay_buffer: List[StreamFusionEnvelope[Any]] = []
        self._subscriptions: Dict[str, LiveClientSubscription] = {}
        # Mapping client_id -> asyncio.Queue[str]
        self._client_queues: Dict[str, asyncio.Queue[str]] = {}
        self._lock = asyncio.Lock()
        self._running = False
        self._server: Optional[asyncio.AbstractServer] = None

    @property
    def replay_buffer_size(self) -> int:
        return len(self._replay_buffer)

    @property
    def active_subscribers_count(self) -> int:
        return len(self._subscriptions)

    def get_replay_events(
        self,
        count: int = 50,
        event_types: Optional[List[str]] = None,
    ) -> List[StreamFusionEnvelope[Any]]:
        """Retrieves recent events matching optional type filter."""
        events = self._replay_buffer[-count:]
        if not event_types or "*" in event_types:
            return list(events)
        types_set = set(event_types)
        return [e for e in events if e.event_type in types_set]

    async def register_client(
        self,
        transport: str = "WEBSOCKET",
        event_types: Optional[List[str]] = None,
        replay_count: int = 0,
    ) -> Tuple[LiveClientSubscription, asyncio.Queue[str]]:
        """Registers a new subscriber and sends initial replay if requested."""
        types = event_types or ["*"]
        sub = LiveClientSubscription(
            client_id=str(uuid.uuid4()),
            transport=transport,
            event_types=types,
            replay_count=replay_count,
        )
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=500)

        async with self._lock:
            self._subscriptions[sub.client_id] = sub
            self._client_queues[sub.client_id] = q

        # Enqueue replay events
        if replay_count > 0:
            replays = self.get_replay_events(replay_count, types)
            for rev in replays:
                try:
                    q.put_nowait(rev.model_dump_json())
                except asyncio.QueueFull:
                    break

        return sub, q

    async def unregister_client(self, client_id: str) -> None:
        """Removes a client subscription."""
        async with self._lock:
            self._subscriptions.pop(client_id, None)
            self._client_queues.pop(client_id, None)

    async def broadcast(self, envelope: StreamFusionEnvelope[Any]) -> int:
        """Broadcasts an envelope to all matching active subscribers."""
        payload_str = envelope.model_dump_json()

        async with self._lock:
            # Append to replay buffer
            self._replay_buffer.append(envelope)
            if len(self._replay_buffer) > self.replay_capacity:
                self._replay_buffer.pop(0)

            delivered = 0
            for cid, sub in list(self._subscriptions.items()):
                if "*" in sub.event_types or envelope.event_type in sub.event_types:
                    q = self._client_queues.get(cid)
                    if q:
                        try:
                            q.put_nowait(payload_str)
                            delivered += 1
                        except asyncio.QueueFull:
                            # Drop message if client is lagging behind
                            pass
            return delivered

    def emit_event(
        self,
        event_type: str,
        payload: Any,
        stream_id: Optional[str] = None,
    ) -> StreamFusionEnvelope[Any]:
        """Convenience method to construct and broadcast an envelope synchronously/async."""
        env = StreamFusionEnvelope[Any](
            event_type=event_type,
            stream_id=stream_id,
            payload=payload,
        )
        # Schedule broadcast if event loop is running
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.broadcast(env))
        except RuntimeError:
            # Synchronous append to replay buffer if no loop active
            self._replay_buffer.append(env)
            if len(self._replay_buffer) > self.replay_capacity:
                self._replay_buffer.pop(0)
        return env

    # --- HTTP SSE & WebSocket Transport Server ---

    @staticmethod
    def encode_ws_frame(message: str) -> bytes:
        """Encodes a text message into an unmasked RFC 6455 WebSocket frame."""
        data = message.encode("utf-8")
        length = len(data)
        header = bytearray([0x81])  # FIN + Text Frame

        if length <= 125:
            header.append(length)
        elif length <= 65535:
            header.append(126)
            header.extend(struct.pack("!H", length))
        else:
            header.append(127)
            header.extend(struct.pack("!Q", length))

        return bytes(header) + data

    @staticmethod
    def decode_ws_frame(data: bytes) -> Tuple[Optional[str], Optional[int], bytes]:
        """Decodes an incoming masked RFC 6455 WebSocket frame.
        
        Returns (decoded_text, opcode, remaining_bytes).
        """
        if len(data) < 2:
            return None, None, data

        b1 = data[0]
        b2 = data[1]
        opcode = b1 & 0x0F
        masked = (b2 & 0x80) != 0
        payload_len = b2 & 0x7F

        idx = 2
        if payload_len == 126:
            if len(data) < 4:
                return None, None, data
            payload_len = struct.unpack("!H", data[2:4])[0]
            idx = 4
        elif payload_len == 127:
            if len(data) < 10:
                return None, None, data
            payload_len = struct.unpack("!Q", data[2:10])[0]
            idx = 10

        mask_key = None
        if masked:
            if len(data) < idx + 4:
                return None, None, data
            mask_key = data[idx:idx + 4]
            idx += 4

        if len(data) < idx + payload_len:
            return None, None, data

        payload = bytearray(data[idx:idx + payload_len])
        if masked and mask_key:
            for i in range(len(payload)):
                payload[i] ^= mask_key[i % 4]

        remaining = data[idx + payload_len:]
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            text = payload.decode("utf-8", errors="replace")

        return text, opcode, remaining

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handles incoming HTTP SSE and WebSocket upgrade requests."""
        try:
            request_line = await reader.readline()
            if not request_line:
                writer.close()
                return

            req_str = request_line.decode("utf-8", errors="replace").strip()
            headers: Dict[str, str] = {}
            while True:
                line = await reader.readline()
                if not line or line == b"\r\n":
                    break
                h_str = line.decode("utf-8", errors="replace").strip()
                if ":" in h_str:
                    k, v = h_str.split(":", 1)
                    headers[k.strip().lower()] = v.strip()

            parts = req_str.split()
            if len(parts) < 2:
                writer.close()
                return

            method, path = parts[0], parts[1]

            # 1. Server-Sent Events (SSE) Endpoint
            if path.startswith("/events") or "text/event-stream" in headers.get("accept", ""):
                # Query params parsing
                replay = 0
                events = ["*"]
                if "?" in path:
                    query = path.split("?", 1)[1]
                    for pair in query.split("&"):
                        if "=" in pair:
                            qk, qv = pair.split("=", 1)
                            if qk == "replay":
                                replay = int(qv) if qv.isdigit() else 0
                            elif qk == "events":
                                events = [e.strip() for e in qv.split(",") if e.strip()]

                sub, queue = await self.register_client(
                    transport="SSE",
                    event_types=events,
                    replay_count=replay,
                )

                response_header = (
                    "HTTP/1.1 200 OK\r\n"
                    "Content-Type: text/event-stream\r\n"
                    "Cache-Control: no-cache\r\n"
                    "Connection: keep-alive\r\n"
                    "Access-Control-Allow-Origin: *\r\n\r\n"
                )
                writer.write(response_header.encode("utf-8"))
                await writer.drain()

                try:
                    while self._running:
                        msg = await queue.get()
                        frame = f"data: {msg}\n\n"
                        writer.write(frame.encode("utf-8"))
                        await writer.drain()
                finally:
                    await self.unregister_client(sub.client_id)
                    writer.close()
                return

            # 2. WebSocket Upgrade Request
            if headers.get("upgrade", "").lower() == "websocket":
                ws_key = headers.get("sec-websocket-key")
                if not ws_key:
                    writer.close()
                    return

                accept_val = base64.b64encode(
                    hashlib.sha1((ws_key + WS_GUID).encode("utf-8")).digest()
                ).decode("utf-8")

                resp = (
                    "HTTP/1.1 101 Switching Protocols\r\n"
                    "Upgrade: websocket\r\n"
                    "Connection: Upgrade\r\n"
                    f"Sec-WebSocket-Accept: {accept_val}\r\n\r\n"
                )
                writer.write(resp.encode("utf-8"))
                await writer.drain()

                sub, queue = await self.register_client(transport="WEBSOCKET")

                # Writer loop
                async def sender():
                    try:
                        while self._running:
                            msg = await queue.get()
                            frame = self.encode_ws_frame(msg)
                            writer.write(frame)
                            await writer.drain()
                    except Exception:
                        pass

                sender_task = asyncio.create_task(sender())

                # Reader loop
                buffer = b""
                try:
                    while self._running and not reader.at_eof():
                        chunk = await reader.read(4096)
                        if not chunk:
                            break
                        buffer += chunk
                        while True:
                            text, opcode, remaining = self.decode_ws_frame(buffer)
                            if opcode is None:
                                break
                            buffer = remaining
                            if opcode == 0x8:  # Close
                                return
                            if opcode == 0x9:  # Ping
                                pong_frame = bytearray([0x8A, 0x00])
                                writer.write(pong_frame)
                                await writer.drain()
                                continue
                            if text:
                                # Handle subscription messages
                                try:
                                    cmd = json.loads(text)
                                    if isinstance(cmd, dict) and cmd.get("action") == "subscribe":
                                        sub.event_types = cmd.get("events", ["*"])
                                        req_replay = cmd.get("replay_count", 0)
                                        if req_replay > 0:
                                            for rev in self.get_replay_events(req_replay, sub.event_types):
                                                writer.write(self.encode_ws_frame(rev.model_dump_json()))
                                                await writer.drain()
                                except json.JSONDecodeError:
                                    pass
                finally:
                    sender_task.cancel()
                    await self.unregister_client(sub.client_id)
                    writer.close()
                return

            # Default fallback: 404
            writer.write(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")
            await writer.drain()
            writer.close()

        except Exception as e:
            logger.debug(f"Connection error: {e}")
            try:
                writer.close()
            except Exception:
                pass

    async def start_server(self, host: str = "0.0.0.0", port: int = 8765) -> None:
        """Starts the combined WebSocket and SSE HTTP server."""
        if self._running:
            return
        self._running = True
        self._server = await asyncio.start_server(self._handle_connection, host, port)
        logger.info(f"LiveEventBroadcaster listening on ws://{host}:{port} and http://{host}:{port}/events")

    async def stop_server(self) -> None:
        """Gracefully terminates the server."""
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
