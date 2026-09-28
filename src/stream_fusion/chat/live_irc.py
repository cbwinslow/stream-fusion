"""Live Twitch IRC Parser and WebSocket Chat Stream Tailer (Spec 19)."""

import asyncio
from datetime import datetime, timezone
import logging
import re
import ssl
import time
from typing import Any, Callable, Dict, List, Optional
import uuid

from stream_fusion.models.schemas import ChatEmote, ChatMessage
from stream_fusion.nlp.adaptive_slang import RollingBurstDetector

logger = logging.getLogger(__name__)


class TwitchIrcMessage:
    """Structured representation of an IRC RFC 1459 / IRCv3 Twitch message."""

    def __init__(
        self,
        raw_line: str,
        tags: Dict[str, str],
        prefix: Optional[str],
        command: str,
        params: List[str],
        trailing: Optional[str],
    ):
        self.raw_line = raw_line
        self.tags = tags
        self.prefix = prefix
        self.command = command
        self.params = params
        self.trailing = trailing

    def __repr__(self) -> str:
        return f"<TwitchIrcMessage command={self.command} params={self.params} trailing={self.trailing[:20] if self.trailing else ''}>"


class TwitchIrcParser:
    """Parses Twitch IRC raw wire lines with IRCv3 tags into structured messages."""

    TAG_UNESCAPE = {
        r"\:": ";",
        r"\s": " ",
        r"\\": "\\",
        r"\r": "\r",
        r"\n": "\n",
    }

    @classmethod
    def unescape_tag_value(cls, val: str) -> str:
        res = val
        for escaped, orig in cls.TAG_UNESCAPE.items():
            res = res.replace(escaped, orig)
        return res

    @classmethod
    def parse_line(cls, line: str) -> Optional[TwitchIrcMessage]:
        """Parses a single IRC line."""
        if not line:
            return None
        line = line.strip("\r\n")
        if not line:
            return None

        tags: Dict[str, str] = {}
        idx = 0

        # 1. Parse IRCv3 tags
        if line.startswith("@"):
            space_idx = line.find(" ")
            if space_idx == -1:
                return None
            raw_tags = line[1:space_idx]
            for tag_item in raw_tags.split(";"):
                if "=" in tag_item:
                    k, v = tag_item.split("=", 1)
                    tags[k] = cls.unescape_tag_value(v)
                else:
                    tags[tag_item] = ""
            idx = space_idx + 1

        # Skip whitespace
        while idx < len(line) and line[idx] == " ":
            idx += 1

        # 2. Parse prefix
        prefix: Optional[str] = None
        if idx < len(line) and line[idx] == ":":
            space_idx = line.find(" ", idx)
            if space_idx == -1:
                return None
            prefix = line[idx + 1:space_idx]
            idx = space_idx + 1

        while idx < len(line) and line[idx] == " ":
            idx += 1

        # 3. Parse command & params
        remaining = line[idx:]
        params: List[str] = []
        trailing: Optional[str] = None

        if " :" in remaining:
            parts, trailing = remaining.split(" :", 1)
            tokens = parts.strip().split()
        else:
            tokens = remaining.strip().split()

        if not tokens:
            return None

        command = tokens[0].upper()
        if len(tokens) > 1:
            params = tokens[1:]

        return TwitchIrcMessage(
            raw_line=line,
            tags=tags,
            prefix=prefix,
            command=command,
            params=params,
            trailing=trailing,
        )

    @classmethod
    def to_chat_message(
        cls,
        irc_msg: TwitchIrcMessage,
        stream_start_ms: Optional[float] = None,
    ) -> Optional[ChatMessage]:
        """Converts an IRC PRIVMSG message to a StreamFusion ChatMessage model."""
        if irc_msg.command != "PRIVMSG" or irc_msg.trailing is None:
            return None

        tags = irc_msg.tags
        msg_id = tags.get("id") or str(uuid.uuid4())

        # Author Name
        author_name = tags.get("display-name")
        if not author_name and irc_msg.prefix:
            author_name = irc_msg.prefix.split("!")[0]
        if not author_name:
            author_name = "anonymous"

        # User ID
        user_id = tags.get("user-id") or author_name.lower()

        # Timestamp offset
        tmi_sent_ts = tags.get("tmi-sent-ts")
        if tmi_sent_ts:
            try:
                sent_ms = float(tmi_sent_ts)
                if stream_start_ms:
                    timestamp_offset = max(0.0, (sent_ms - stream_start_ms) / 1000.0)
                else:
                    timestamp_offset = sent_ms / 1000.0
            except ValueError:
                timestamp_offset = time.time()
        else:
            timestamp_offset = time.time()

        # Badges
        badges: List[str] = []
        raw_badges = tags.get("badges")
        if raw_badges:
            badges = [b.strip() for b in raw_badges.split(",") if b.strip()]

        # Emotes
        content = irc_msg.trailing
        emotes: List[ChatEmote] = []
        raw_emotes = tags.get("emotes")
        if raw_emotes:
            # Format: emote_id:start-end,start-end/emote_id2:start-end
            for emote_group in raw_emotes.split("/"):
                if ":" in emote_group:
                    emote_id, spans = emote_group.split(":", 1)
                    span_list = spans.split(",")
                    # extract name from first span
                    name = "emote"
                    if span_list and "-" in span_list[0]:
                        try:
                            s, e = map(int, span_list[0].split("-", 1))
                            name = content[s:e + 1]
                        except Exception:
                            name = emote_id
                    emotes.append(ChatEmote(id=emote_id, name=name, count=len(span_list)))

        return ChatMessage(
            message_id=msg_id,
            timestamp_offset=timestamp_offset,
            user_id=user_id,
            author_name=author_name,
            content=content,
            emotes=emotes,
            badges=badges,
        )


class LiveChatTailer:
    """Asynchronous client for tailing live Twitch/Kick chat streams."""

    def __init__(
        self,
        channel_name: str,
        burst_detector: Optional[RollingBurstDetector] = None,
        anonymous: bool = True,
        nick: Optional[str] = None,
        oauth: Optional[str] = None,
        stream_start_ms: Optional[float] = None,
    ):
        self.channel_name = channel_name.lstrip("#").lower()
        self.burst_detector = burst_detector
        self.anonymous = anonymous
        self.nick = nick or f"justinfan{int(time.time() % 100000)}"
        self.oauth = oauth or "SCHMOOPIIE"
        self.stream_start_ms = stream_start_ms or (time.time() * 1000.0)

        self._running = False
        self._callbacks: List[Callable[[ChatMessage], Any]] = []
        self._burst_callbacks: List[Callable[[Any], Any]] = []
        self._rolling_messages: List[ChatMessage] = []
        self._recent_timestamps: List[float] = []
        self._total_received = 0
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._task: Optional[asyncio.Task] = None

    @property
    def total_messages(self) -> int:
        return self._total_received

    @property
    def is_running(self) -> bool:
        return self._running

    def register_callback(self, cb: Callable[[ChatMessage], Any]) -> None:
        """Subscribes a synchronous or asynchronous callback to incoming chat messages."""
        self._callbacks.append(cb)

    def register_burst_callback(self, cb: Callable[[Any], Any]) -> None:
        """Subscribes a synchronous or asynchronous callback to detected token bursts."""
        self._burst_callbacks.append(cb)

    def get_message_velocity(self, window_sec: float = 10.0) -> float:
        """Computes current chat messages per second over sliding window."""
        now = time.time()
        self._recent_timestamps = [t for t in self._recent_timestamps if now - t <= window_sec]
        if not self._recent_timestamps:
            return 0.0
        return len(self._recent_timestamps) / max(window_sec, 1.0)

    async def start(self, host: str = "irc.chat.twitch.tv", port: int = 6697, use_ssl: bool = True) -> None:
        """Starts live connection task."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop(host, port, use_ssl))

    async def stop(self) -> None:
        """Stops live connection."""
        self._running = False
        if self._writer:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def ingest_raw_line(self, line: str) -> Optional[ChatMessage]:
        """Manually ingests an IRC line (useful for testing and simulated feeds)."""
        msg = TwitchIrcParser.parse_line(line)
        if not msg:
            return None

        # Auto PONG
        if msg.command == "PING":
            return None

        if msg.command == "PRIVMSG":
            chat_msg = TwitchIrcParser.to_chat_message(msg, self.stream_start_ms)
            if chat_msg:
                now = time.time()
                self._recent_timestamps.append(now)
                self._total_received += 1

                # Feed burst detector if available
                if self.burst_detector:
                    self._rolling_messages.append(chat_msg)
                    cutoff = chat_msg.timestamp_offset - (self.burst_detector.window_sec * 2)
                    self._rolling_messages = [m for m in self._rolling_messages if m.timestamp_offset >= cutoff]
                    if len(self._rolling_messages) >= self.burst_detector.min_occurrences:
                        bursts = self.burst_detector.detect_bursts(self._rolling_messages)
                        for b in bursts:
                            for bcb in self._burst_callbacks:
                                try:
                                    b_res = bcb(b)
                                    if asyncio.iscoroutine(b_res):
                                        await b_res
                                except Exception as b_err:
                                    logger.error(f"Error in burst callback: {b_err}")

                # Dispatch callbacks
                for cb in self._callbacks:
                    try:
                        res = cb(chat_msg)
                        if asyncio.iscoroutine(res):
                            await res
                    except Exception as err:
                        logger.error(f"Error in chat callback: {err}")
                return chat_msg
        return None

    async def _run_loop(self, host: str, port: int, use_ssl: bool) -> None:
        """Internal TCP / SSL connection loop with automatic reconnect."""
        while self._running:
            try:
                ssl_ctx = ssl.create_default_context() if use_ssl else None
                self._reader, self._writer = await asyncio.open_connection(
                    host, port, ssl=ssl_ctx
                )

                # Request Twitch IRCv3 capabilities
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

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Live chat connection error: {e}. Reconnecting in 3s...")
                await asyncio.sleep(3.0)
