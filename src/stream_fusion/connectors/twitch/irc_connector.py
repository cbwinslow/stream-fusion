"""Twitch Live Chat Connector via RFC 1459 IRC and IRCv3 Tags (Spec 22)."""

import asyncio
import logging
import ssl
import time
from typing import Any, Callable, Dict, List, Optional

from stream_fusion.chat.live_irc import TwitchIrcParser
from stream_fusion.connectors.base import BaseChatConnector
from stream_fusion.connectors.normalizer import MessageNormalizer
from stream_fusion.models.schemas import (
    ChatMessage,
    LivePlatform,
    PlatformCapabilities,
)
from stream_fusion.nlp.adaptive_slang import RollingBurstDetector

logger = logging.getLogger(__name__)


class TwitchChatConnector(BaseChatConnector):
    """Asynchronous client for tailing live Twitch chat streams via IRCv3."""

    def __init__(
        self,
        channel_name: str,
        burst_detector: Optional[RollingBurstDetector] = None,
        anonymous: bool = True,
        nick: Optional[str] = None,
        oauth: Optional[str] = None,
        stream_start_ms: Optional[float] = None,
        max_reconnect_attempts: int = 10,
        base_reconnect_delay: float = 2.0,
    ):
        super().__init__(
            channel_name=channel_name.lstrip("#").lower(),
            platform=LivePlatform.TWITCH,
            burst_detector=burst_detector,
            stream_start_ms=stream_start_ms,
            max_reconnect_attempts=max_reconnect_attempts,
            base_reconnect_delay=base_reconnect_delay,
        )
        self.anonymous = anonymous
        self.nick = nick or f"justinfan{int(time.time() % 100000)}"
        self.oauth = oauth or "SCHMOOPIIE"
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None

    @property
    def capabilities(self) -> PlatformCapabilities:
        return PlatformCapabilities(
            platform=LivePlatform.TWITCH,
            supports_emotes=True,
            supports_badges=True,
            supports_superchats=False,
            supports_bits=True,
            supports_subscriptions=True,
            requires_api_key=False,
            supports_anonymous=True,
        )

    async def ingest_raw_line(self, line: str) -> Optional[ChatMessage]:
        """Manually ingests a single IRC wire line (used for unit tests and simulated feeds)."""
        msg = TwitchIrcParser.parse_line(line)
        if not msg:
            return None

        if msg.command == "PING":
            if self._writer:
                token = msg.trailing or ":tmi.twitch.tv"
                self._writer.write(f"PONG {token}\r\n".encode())
                await self._writer.drain()
            return None

        if msg.command == "PRIVMSG" and msg.trailing is not None:
            chat_msg = MessageNormalizer.from_twitch_irc(
                tags=msg.tags,
                prefix=msg.prefix,
                content=msg.trailing,
                stream_start_ms=self.stream_start_ms,
            )
            await self.ingest_normalized_message(chat_msg)
            return chat_msg

        return None

    async def _connect_and_listen(self) -> None:
        """Connects to Twitch IRC server, requests IRCv3 capabilities, and reads lines."""
        host = "irc.chat.twitch.tv"
        port = 6697
        ssl_ctx = ssl.create_default_context()

        self._reader, self._writer = await asyncio.open_connection(host, port, ssl=ssl_ctx)

        # Twitch IRC Handshake
        self._writer.write(b"CAP REQ :twitch.tv/tags twitch.tv/commands\r\n")
        if self.anonymous:
            self._writer.write(f"PASS SCHMOOPIIE\r\nNICK {self.nick}\r\n".encode())
        else:
            self._writer.write(f"PASS {self.oauth}\r\nNICK {self.nick}\r\n".encode())
        self._writer.write(f"JOIN #{self.channel_name}\r\n".encode())
        await self._writer.drain()

        while self._running and not self._reader.at_eof():
            line_bytes = await self._reader.readline()
            if not line_bytes:
                break
            line_str = line_bytes.decode("utf-8", errors="replace").strip("\r\n")
            if line_str.startswith("PING"):
                token = line_str.split(" ", 1)[1] if " " in line_str else ":tmi.twitch.tv"
                self._writer.write(f"PONG {token}\r\n".encode())
                await self._writer.drain()
                continue

            await self.ingest_raw_line(line_str)

    async def _disconnect(self) -> None:
        """Closes socket connection."""
        if self._writer:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
            self._writer = None
            self._reader = None
