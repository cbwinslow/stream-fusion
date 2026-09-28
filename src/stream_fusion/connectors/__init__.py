"""StreamFusion Live Multi-Platform Connectors subsystem (Spec 22)."""

from stream_fusion.connectors.base import BaseChatConnector
from stream_fusion.connectors.normalizer import MessageNormalizer
from stream_fusion.connectors.registry import ConnectorRegistry
from stream_fusion.connectors.kick.pusher_connector import KickChatConnector
from stream_fusion.connectors.kick.resolver import KickChannelResolver
from stream_fusion.connectors.kick.webhook_receiver import KickWebhookReceiver
from stream_fusion.connectors.twitch.irc_connector import TwitchChatConnector
from stream_fusion.connectors.youtube.chat_downloader import YouTubeChatConnector
from stream_fusion.connectors.youtube.data_api import YouTubeDataApiConnector

__all__ = [
    "BaseChatConnector",
    "MessageNormalizer",
    "ConnectorRegistry",
    "KickChatConnector",
    "KickChannelResolver",
    "KickWebhookReceiver",
    "TwitchChatConnector",
    "YouTubeChatConnector",
    "YouTubeDataApiConnector",
]
