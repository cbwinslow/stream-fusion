"""YouTube Live Chat Connector leveraging chat-downloader and InnerTube (Spec 22)."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import time
from typing import Any, Dict, List, Optional

from stream_fusion.connectors.base import BaseChatConnector
from stream_fusion.connectors.normalizer import MessageNormalizer
from stream_fusion.models.schemas import (
    ChatMessage,
    LivePlatform,
    PlatformCapabilities,
)
from stream_fusion.nlp.adaptive_slang import RollingBurstDetector

logger = logging.getLogger(__name__)


class YouTubeChatConnector(BaseChatConnector):
    """Asynchronous client for tailing live YouTube chat streams via chat-downloader.
    
    Supports live video IDs, YouTube live URLs (e.g. https://www.youtube.com/watch?v=...),
    and channel handles (e.g. @CreatorName). Operates without requiring a Google Cloud API key.
    """

    def __init__(
        self,
        channel_or_url: str,
        burst_detector: Optional[RollingBurstDetector] = None,
        stream_start_ms: Optional[float] = None,
        max_reconnect_attempts: int = 10,
        base_reconnect_delay: float = 2.0,
        message_receive_timeout: float = 0.5,
    ):
        super().__init__(
            channel_name=channel_or_url,
            platform=LivePlatform.YOUTUBE_LIVE,
            burst_detector=burst_detector,
            stream_start_ms=stream_start_ms,
            max_reconnect_attempts=max_reconnect_attempts,
            base_reconnect_delay=base_reconnect_delay,
        )
        self.target_url = self._format_youtube_target(channel_or_url)
        self.message_receive_timeout = message_receive_timeout
        self._executor: Optional[ThreadPoolExecutor] = None
        self._stop_event = asyncio.Event()

    @staticmethod
    def _format_youtube_target(target: str) -> str:
        """Formats input into a valid YouTube URL or identifier."""
        target = target.strip()
        if target.startswith("http://") or target.startswith("https://"):
            return target
        if target.startswith("@"):
            return f"https://www.youtube.com/{target}/live"
        if len(target) == 11 and "/" not in target:
            # 11-char YouTube video ID
            return f"https://www.youtube.com/watch?v={target}"
        if target.startswith("UC") and len(target) == 24:
            # Channel ID
            return f"https://www.youtube.com/channel/{target}/live"
        return f"https://www.youtube.com/@{target}/live"

    @property
    def capabilities(self) -> PlatformCapabilities:
        return PlatformCapabilities(
            platform=LivePlatform.YOUTUBE_LIVE,
            supports_emotes=True,
            supports_badges=True,
            supports_superchats=True,
            supports_bits=False,
            supports_subscriptions=True,
            requires_api_key=False,
            supports_anonymous=True,
        )

    async def ingest_raw_item(self, item: Dict[str, Any]) -> Optional[ChatMessage]:
        """Manually ingests a YouTube raw chat item (for testing and simulated live feeds)."""
        msg = MessageNormalizer.from_youtube_item(item, self.stream_start_ms)
        if msg:
            await self.ingest_normalized_message(msg)
            return msg
        return None

    def _sync_stream_chat(self, loop: asyncio.AbstractEventLoop, q: asyncio.Queue):
        """Worker thread executing synchronous chat-downloader generator."""
        try:
            from chat_downloader import ChatDownloader
            downloader = ChatDownloader()
            chat_iter = downloader.get_chat(
                url=self.target_url,
                chat_type="live",
                message_receive_timeout=self.message_receive_timeout,
            )
            for item in chat_iter:
                if not self._running:
                    break
                asyncio.run_coroutine_threadsafe(q.put(item), loop)
        except Exception as e:
            logger.warning(f"[YouTubeChat] Generator thread terminated: {e}")
        finally:
            asyncio.run_coroutine_threadsafe(q.put(None), loop)

    async def _connect_and_listen(self) -> None:
        """Runs the chat-downloader stream and dispatches items to normalizer."""
        self._stop_event.clear()
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yt-chat")
        future = loop.run_in_executor(self._executor, self._sync_stream_chat, loop, q)

        try:
            while self._running:
                try:
                    item = await asyncio.wait_for(q.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                if item is None:
                    # Generator ended or disconnected
                    break

                await self.ingest_raw_item(item)
        finally:
            self._running = False
            if self._executor:
                self._executor.shutdown(wait=False, cancel_futures=True)
                self._executor = None

    async def _disconnect(self) -> None:
        """Signals thread to stop."""
        self._stop_event.set()
        if self._executor:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
