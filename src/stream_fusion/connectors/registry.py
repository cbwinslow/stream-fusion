"""Connector Registry factory for instantiating platform connectors (Spec 22)."""

import logging
from typing import Dict, Optional, Type

from stream_fusion.connectors.base import BaseChatConnector
from stream_fusion.connectors.kick.pusher_connector import KickChatConnector
from stream_fusion.connectors.twitch.irc_connector import TwitchChatConnector
from stream_fusion.connectors.youtube.chat_downloader import YouTubeChatConnector
from stream_fusion.connectors.youtube.data_api import YouTubeDataApiConnector
from stream_fusion.models.schemas import LivePlatform, LiveStreamConfig
from stream_fusion.nlp.adaptive_slang import RollingBurstDetector

logger = logging.getLogger(__name__)


class ConnectorRegistry:
    """Central factory for managing and instantiating platform chat and stream connectors."""

    _chat_connectors: Dict[LivePlatform, Type[BaseChatConnector]] = {
        LivePlatform.TWITCH: TwitchChatConnector,
        LivePlatform.KICK: KickChatConnector,
        LivePlatform.YOUTUBE_LIVE: YouTubeChatConnector,
    }

    @classmethod
    def register_chat_connector(
        cls,
        platform: LivePlatform,
        connector_cls: Type[BaseChatConnector],
    ) -> None:
        """Registers a custom platform chat connector class."""
        cls._chat_connectors[platform] = connector_cls

    @classmethod
    def create_chat_connector(
        cls,
        config: LiveStreamConfig,
        burst_detector: Optional[RollingBurstDetector] = None,
        stream_start_ms: Optional[float] = None,
    ) -> BaseChatConnector:
        """Instantiates the appropriate chat connector for the given LiveStreamConfig."""
        platform = config.platform

        if platform == LivePlatform.TWITCH:
            return TwitchChatConnector(
                channel_name=config.channel_name,
                burst_detector=burst_detector,
                anonymous=config.anonymous_chat,
                nick=config.irc_nick,
                oauth=config.irc_oauth,
                stream_start_ms=stream_start_ms,
            )

        if platform == LivePlatform.KICK:
            return KickChatConnector(
                channel_name=config.channel_name,
                chatroom_id=config.chatroom_id,
                burst_detector=burst_detector,
                stream_start_ms=stream_start_ms,
            )

        if platform == LivePlatform.YOUTUBE_LIVE:
            if config.youtube_api_key:
                # Use official Data API if key is present
                return YouTubeDataApiConnector(
                    live_chat_id=config.channel_name,
                    api_key=config.youtube_api_key,
                    burst_detector=burst_detector,
                    stream_start_ms=stream_start_ms,
                )
            target = config.stream_url or config.channel_name
            return YouTubeChatConnector(
                channel_or_url=target,
                burst_detector=burst_detector,
                stream_start_ms=stream_start_ms,
            )

        # Fallback to registered class
        connector_cls = cls._chat_connectors.get(platform)
        if connector_cls:
            return connector_cls(
                channel_name=config.channel_name,
                burst_detector=burst_detector,
                stream_start_ms=stream_start_ms,
            )

        raise ValueError(f"Unsupported or unregistered live platform: {platform}")
