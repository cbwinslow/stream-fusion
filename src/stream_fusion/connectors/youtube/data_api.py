"""YouTube Official Data API v3 Live Chat Polling Connector (Spec 22)."""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional
import uuid

from stream_fusion.connectors.base import BaseChatConnector
from stream_fusion.connectors.normalizer import MessageNormalizer
from stream_fusion.models.schemas import (
    ChatMessage,
    LivePlatform,
    PlatformCapabilities,
)
from stream_fusion.nlp.adaptive_slang import RollingBurstDetector

logger = logging.getLogger(__name__)


class YouTubeDataApiConnector(BaseChatConnector):
    """Authenticated connector polling YouTube Data API v3 liveChatMessages.list.
    
    Reference: developers.google.com/youtube/v3/live/docs/liveChatMessages/list
    """

    def __init__(
        self,
        live_chat_id: str,
        api_key: str,
        burst_detector: Optional[RollingBurstDetector] = None,
        stream_start_ms: Optional[float] = None,
        max_reconnect_attempts: int = 10,
        base_reconnect_delay: float = 2.0,
    ):
        super().__init__(
            channel_name=live_chat_id,
            platform=LivePlatform.YOUTUBE_LIVE,
            burst_detector=burst_detector,
            stream_start_ms=stream_start_ms,
            max_reconnect_attempts=max_reconnect_attempts,
            base_reconnect_delay=base_reconnect_delay,
        )
        self.live_chat_id = live_chat_id
        self.api_key = api_key
        self._next_page_token: Optional[str] = None
        self._polling_interval_ms: int = 2000

    @property
    def capabilities(self) -> PlatformCapabilities:
        return PlatformCapabilities(
            platform=LivePlatform.YOUTUBE_LIVE,
            supports_emotes=True,
            supports_badges=True,
            supports_superchats=True,
            supports_bits=False,
            supports_subscriptions=True,
            requires_api_key=True,
            supports_anonymous=False,
        )

    def normalize_api_item(self, item: Dict[str, Any]) -> Optional[ChatMessage]:
        """Converts YouTube Data API v3 LiveChatMessage resource into canonical ChatMessage."""
        msg_id = item.get("id") or str(uuid.uuid4())
        snippet = item.get("snippet", {})
        author_details = item.get("authorDetails", {})

        author_name = author_details.get("displayName", "Anonymous")
        user_id = str(author_details.get("channelId", author_name.lower()))

        # Badges
        badges = []
        if author_details.get("isChatOwner"):
            badges.append("Owner")
        if author_details.get("isChatModerator"):
            badges.append("Moderator")
        if author_details.get("isChatSponsor"):
            badges.append("Member")
        if author_details.get("isVerified"):
            badges.append("Verified")

        content = snippet.get("displayMessage", "")
        published_at = snippet.get("publishedAt")
        now = time.time()
        start_sec = (self.stream_start_ms / 1000.0) if self.stream_start_ms else now

        try:
            from datetime import datetime
            dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
            timestamp_offset = max(0.0, dt.timestamp() - start_sec)
        except Exception:
            timestamp_offset = max(0.0, now - start_sec)

        metadata: Dict[str, Any] = {
            "platform": "YOUTUBE_LIVE",
            "source": "DATA_API_V3",
        }

        # Superchat check
        super_chat = snippet.get("superChatDetails")
        if super_chat:
            metadata["monetization"] = {
                "type": "SUPER_CHAT",
                "amount": float(super_chat.get("amountMicros", 0)) / 1_000_000.0,
                "currency": super_chat.get("currency", "USD"),
                "text": super_chat.get("amountDisplayString", ""),
            }

        return ChatMessage(
            message_id=msg_id,
            timestamp_offset=timestamp_offset,
            user_id=user_id,
            author_name=author_name,
            content=content,
            emotes=[],
            badges=badges,
            metadata=metadata,
        )

    async def _connect_and_listen(self) -> None:
        """Polls liveChatMessages.list endpoint according to recommended polling intervals."""
        import requests

        base_url = "https://www.googleapis.com/youtube/v3/liveChat/messages"

        while self._running:
            params = {
                "liveChatId": self.live_chat_id,
                "part": "snippet,authorDetails",
                "key": self.api_key,
                "maxResults": 500,
            }
            if self._next_page_token:
                params["pageToken"] = self._next_page_token

            resp = requests.get(base_url, params=params, timeout=5.0)
            if resp.status_code != 200:
                raise ConnectionError(f"YouTube Data API error ({resp.status_code}): {resp.text}")

            data = resp.json()
            self._next_page_token = data.get("nextPageToken")
            self._polling_interval_ms = data.get("pollingIntervalMillis", 2000)

            items = data.get("items", [])
            for raw_item in items:
                msg = self.normalize_api_item(raw_item)
                if msg:
                    await self.ingest_normalized_message(msg)

            await asyncio.sleep(max(1.0, self._polling_interval_ms / 1000.0))

    async def _disconnect(self) -> None:
        """Teardown method."""
        self._next_page_token = None
