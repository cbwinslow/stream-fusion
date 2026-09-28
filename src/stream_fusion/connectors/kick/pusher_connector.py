"""Kick Live Chat Connector via Pusher Channels Protocol (Spec 22)."""

import asyncio
import base64
import json
import logging
import os
import ssl
import struct
import time
from typing import Any, Dict, List, Optional, Tuple

from stream_fusion.connectors.base import BaseChatConnector
from stream_fusion.connectors.kick.resolver import KickChannelResolver
from stream_fusion.connectors.normalizer import MessageNormalizer
from stream_fusion.models.schemas import (
    ChatMessage,
    LivePlatform,
    PlatformCapabilities,
)
from stream_fusion.nlp.adaptive_slang import RollingBurstDetector

logger = logging.getLogger(__name__)

# Default Kick Pusher credentials
DEFAULT_PUSHER_KEY = "eb1d5f283081a78b932c"
DEFAULT_PUSHER_CLUSTER = "us2"


class KickChatConnector(BaseChatConnector):
    """Pusher WebSocket client for tailing live Kick.com chat rooms."""

    def __init__(
        self,
        channel_name: str,
        chatroom_id: Optional[int] = None,
        pusher_key: str = DEFAULT_PUSHER_KEY,
        pusher_cluster: str = DEFAULT_PUSHER_CLUSTER,
        resolver: Optional[KickChannelResolver] = None,
        burst_detector: Optional[RollingBurstDetector] = None,
        stream_start_ms: Optional[float] = None,
        max_reconnect_attempts: int = 10,
        base_reconnect_delay: float = 2.0,
    ):
        super().__init__(
            channel_name=channel_name,
            platform=LivePlatform.KICK,
            burst_detector=burst_detector,
            stream_start_ms=stream_start_ms,
            max_reconnect_attempts=max_reconnect_attempts,
            base_reconnect_delay=base_reconnect_delay,
        )
        self.resolver = resolver or KickChannelResolver()
        self.chatroom_id = self.resolver.resolve(channel_name, manual_override=chatroom_id)
        self.pusher_key = pusher_key
        self.pusher_cluster = pusher_cluster
        self.channel_topic = f"chatrooms.{self.chatroom_id}.v2"

        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._subscribed: bool = False

    @property
    def capabilities(self) -> PlatformCapabilities:
        return PlatformCapabilities(
            platform=LivePlatform.KICK,
            supports_emotes=True,
            supports_badges=True,
            supports_superchats=False,
            supports_bits=False,
            supports_subscriptions=True,
            requires_api_key=False,
            supports_anonymous=True,
        )

    @property
    def is_subscribed(self) -> bool:
        return self._subscribed

    async def _send_ws_frame(self, text: str, opcode: int = 0x1) -> None:
        """Sends a masked RFC 6455 client WebSocket frame."""
        if not self._writer:
            return

        payload = text.encode("utf-8")
        payload_len = len(payload)
        header = bytearray()
        # FIN bit set + text opcode
        header.append(0x80 | (opcode & 0x0F))

        # Client-to-server frames MUST be masked (0x80)
        mask_key = os.urandom(4)
        if payload_len < 126:
            header.append(0x80 | payload_len)
        elif payload_len <= 65535:
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", payload_len))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack("!Q", payload_len))

        header.extend(mask_key)

        masked_payload = bytearray(payload_len)
        for i in range(payload_len):
            masked_payload[i] = payload[i] ^ mask_key[i % 4]

        self._writer.write(header + masked_payload)
        await self._writer.drain()

    async def _read_ws_frame(self) -> Optional[Tuple[Optional[str], int]]:
        """Reads and decodes an RFC 6455 server-to-client frame (unmasked from server)."""
        if not self._reader:
            return None

        # Read 2 byte header
        head = await self._reader.readexactly(2)
        b1, b2 = head[0], head[1]
        opcode = b1 & 0x0F
        masked = (b2 & 0x80) != 0
        payload_len = b2 & 0x7F

        if payload_len == 126:
            ext = await self._reader.readexactly(2)
            payload_len = struct.unpack("!H", ext)[0]
        elif payload_len == 127:
            ext = await self._reader.readexactly(8)
            payload_len = struct.unpack("!Q", ext)[0]

        mask_key = None
        if masked:
            mask_key = await self._reader.readexactly(4)

        payload = await self._reader.readexactly(payload_len)
        if masked and mask_key:
            unmasked = bytearray(payload_len)
            for i in range(payload_len):
                unmasked[i] = payload[i] ^ mask_key[i % 4]
            payload = bytes(unmasked)

        if opcode == 0x8:  # Close frame
            return None, 0x8

        if opcode == 0x9:  # Ping frame
            await self._send_ws_frame("", opcode=0xA)
            return "", 0x9

        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            text = payload.decode("utf-8", errors="replace")

        return text, opcode

    async def ingest_raw_event(self, event_name: str, data: Any) -> Optional[ChatMessage]:
        """Manually ingests a Pusher event (used for testing, mock feeds, and simulated live streams)."""
        if event_name == "pusher:connection_established":
            # Send subscription
            sub_payload = json.dumps({
                "event": "pusher:subscribe",
                "data": {"channel": self.channel_topic},
            })
            if self._writer:
                await self._send_ws_frame(sub_payload)
            return None

        if event_name == "pusher_internal:subscription_succeeded":
            self._subscribed = True
            logger.info(f"[Kick] Subscribed to topic: {self.channel_topic}")
            return None

        if event_name == "pusher:ping":
            pong = json.dumps({"event": "pusher:pong", "data": {}})
            if self._writer:
                await self._send_ws_frame(pong)
            return None

        if event_name == "App\\Events\\ChatMessageEvent":
            msg = MessageNormalizer.from_kick_pusher(data, self.stream_start_ms)
            if msg:
                await self.ingest_normalized_message(msg)
                return msg

        return None

    async def _connect_and_listen(self) -> None:
        """Connects to Kick's Pusher cluster, performs handshake, and processes frames."""
        host = f"ws-{self.pusher_cluster}.pusher.com"
        port = 443
        path = f"/app/{self.pusher_key}?protocol=7&client=js&version=7.6.0&flash=false"

        ssl_ctx = ssl.create_default_context()
        self._reader, self._writer = await asyncio.open_connection(host, port, ssl=ssl_ctx)

        # RFC 6455 Client Handshake
        sec_key = base64.b64encode(os.urandom(16)).decode()
        handshake_req = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {sec_key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) StreamFusion/0.1\r\n"
            "\r\n"
        )
        self._writer.write(handshake_req.encode())
        await self._writer.drain()

        # Read HTTP handshake response
        handshake_response = await self._reader.readline()
        if b"101" not in handshake_response:
            raise ConnectionError(f"WebSocket upgrade rejected: {handshake_response.decode().strip()}")

        # Consume remaining handshake headers
        while True:
            line = await self._reader.readline()
            if not line or line == b"\r\n":
                break

        # Main frame listening loop
        while self._running:
            frame_res = await self._read_ws_frame()
            if frame_res is None:
                break
            text, opcode = frame_res
            if opcode == 0x8:  # Closed
                break
            if opcode != 0x1 or not text:
                continue

            try:
                frame_json = json.loads(text)
                event_name = frame_json.get("event", "")
                data_val = frame_json.get("data")
                await self.ingest_raw_event(event_name, data_val)
            except json.JSONDecodeError:
                continue

    async def _disconnect(self) -> None:
        """Closes the WebSocket writer and resets state."""
        self._subscribed = False
        if self._writer:
            try:
                await self._send_ws_frame("", opcode=0x8)
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
            self._writer = None
            self._reader = None
