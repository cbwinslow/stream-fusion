"""Live Stream Coordinator Orchestrating Real-Time Ingest, Chat, and Broadcasting (Spec 19)."""

import asyncio
from datetime import datetime, timezone
import logging
import time
from typing import Any, Dict, List, Optional
import uuid

from stream_fusion.chat.live_irc import LiveChatTailer
from stream_fusion.ingest.live_buffer import LiveMediaSegment
from stream_fusion.ingest.live_tailer import LiveStreamIngestor
from stream_fusion.models.schemas import (
    ChatMessage,
    LiveState,
    LiveStreamConfig,
    LiveStreamStatus,
    MemeBurstEvent,
)
from stream_fusion.monitoring.broadcaster import LiveEventBroadcaster
from stream_fusion.nlp.adaptive_slang import RollingBurstDetector
from stream_fusion.schema.envelope import StreamFusionEnvelope

from stream_fusion.connectors.registry import ConnectorRegistry

logger = logging.getLogger(__name__)


class LiveStreamCoordinator:
    """Master coordinator for real-time live ingestion and event broadcasting.
    
    Orchestrates LiveStreamIngestor, BaseChatConnector (Twitch, Kick, YouTube), RollingBurstDetector,
    and LiveEventBroadcaster into an unified live event pipeline.
    """

    def __init__(
        self,
        config: LiveStreamConfig,
        broadcaster: Optional[LiveEventBroadcaster] = None,
        burst_detector: Optional[RollingBurstDetector] = None,
        chat_connector: Optional[Any] = None,
    ):
        self.config = config
        self.stream_id = f"live-{config.channel_name}-{int(time.time())}"
        self.broadcaster = broadcaster or LiveEventBroadcaster(replay_capacity=config.max_replay_buffer_size)
        self.burst_detector = burst_detector or RollingBurstDetector(window_sec=10.0, z_threshold=3.0)
        self.ingestor = LiveStreamIngestor(config=config)
        self.chat_tailer = chat_connector or ConnectorRegistry.create_chat_connector(
            config=config,
            burst_detector=self.burst_detector,
        )

        self._state: LiveState = LiveState.STOPPED
        self._start_time: Optional[float] = None
        self._started_at_dt: Optional[datetime] = None
        self._last_error: Optional[str] = None

        # Wire up listeners
        self._wire_listeners()

    def _wire_listeners(self) -> None:
        """Connects chat and ingest callbacks to broadcaster."""
        # 1. On Chat Message
        def on_chat(msg: ChatMessage):
            env = StreamFusionEnvelope[ChatMessage](
                event_type="CHAT_MESSAGE",
                stream_id=self.stream_id,
                payload=msg,
            )
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.broadcaster.broadcast(env))
            except RuntimeError:
                pass

        self.chat_tailer.register_callback(on_chat)

        # 2. On Media Segment Ingested
        def on_segment(seg: LiveMediaSegment):
            env = StreamFusionEnvelope[Dict[str, Any]](
                event_type="MEDIA_SEGMENT_BUFFERED",
                stream_id=self.stream_id,
                payload=seg.model_dump(),
            )
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.broadcaster.broadcast(env))
            except RuntimeError:
                pass
        self.ingestor.register_segment_callback(on_segment)

        # 3. On Token / Slang Burst Detected
        def on_burst(candidate):
            env = StreamFusionEnvelope[Dict[str, Any]](
                event_type="CHAT_BURST",
                stream_id=self.stream_id,
                payload=candidate.model_dump() if hasattr(candidate, "model_dump") else {"candidate": str(candidate)},
            )
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.broadcaster.broadcast(env))
            except RuntimeError:
                pass

        self.chat_tailer.register_burst_callback(on_burst)

    @property
    def state(self) -> LiveState:
        return self._state

    def get_status(self) -> LiveStreamStatus:
        """Returns the current operational status and health metrics."""
        uptime = 0.0
        if self._start_time and self._state == LiveState.RUNNING:
            uptime = max(0.0, time.time() - self._start_time)

        chat_vel = self.chat_tailer.get_message_velocity()
        health = self.ingestor.get_health_metrics(chat_velocity=chat_vel)

        return LiveStreamStatus(
            stream_id=self.stream_id,
            state=self._state,
            channel_name=self.config.channel_name,
            platform=self.config.platform,
            uptime_sec=round(uptime, 2),
            total_bytes_ingested=self.ingestor.total_bytes,
            total_chat_messages=self.chat_tailer.total_messages,
            active_subscribers=self.broadcaster.active_subscribers_count,
            health=health,
            last_error=self._last_error,
            started_at=self._started_at_dt,
        )

    async def start(self) -> None:
        """Starts live ingest, chat tailing, and event broadcaster."""
        if self._state == LiveState.RUNNING:
            return

        self._state = LiveState.STARTING
        try:
            self._start_time = time.time()
            self._started_at_dt = datetime.now(timezone.utc)

            # Start broadcaster server if ports are configured
            await self.broadcaster.start_server(port=self.config.ws_port)

            # Start media ingestor
            self.ingestor.start()

            # Start chat tailer if not custom offline
            if self.config.stream_url is None:
                # Live Twitch/Kick IRC
                await self.chat_tailer.start()

            self._state = LiveState.RUNNING

            # Broadcast initial START envelope
            self.broadcaster.emit_event(
                event_type="LIVE_STREAM_STARTED",
                payload={"channel": self.config.channel_name, "stream_id": self.stream_id},
                stream_id=self.stream_id,
            )
            logger.info(f"LiveStreamCoordinator started for {self.stream_id}")
        except Exception as e:
            self._state = LiveState.ERROR
            self._last_error = str(e)
            logger.error(f"Failed to start LiveStreamCoordinator: {e}")
            raise

    async def stop(self) -> None:
        """Gracefully shuts down the coordinator and stops all sub-services."""
        self._state = LiveState.STOPPING
        try:
            # Emit STOP envelope
            self.broadcaster.emit_event(
                event_type="LIVE_STREAM_STOPPED",
                payload={"stream_id": self.stream_id, "final_messages": self.chat_tailer.total_messages},
                stream_id=self.stream_id,
            )

            await self.chat_tailer.stop()
            self.ingestor.stop()
            await self.broadcaster.stop_server()
            self._state = LiveState.STOPPED
            logger.info(f"LiveStreamCoordinator gracefully stopped for {self.stream_id}")
        except Exception as e:
            self._state = LiveState.ERROR
            self._last_error = str(e)
            logger.error(f"Error during shutdown: {e}")
