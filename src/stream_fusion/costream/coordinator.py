"""Multi-Stream Co-Stream Coordinator and Concurrent Session Supervisor (Spec 23)."""

import asyncio
from contextlib import AsyncExitStack
from datetime import datetime, timezone
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Set
import uuid

from stream_fusion.connectors.base import BaseChatConnector, ConnectorState
from stream_fusion.connectors.registry import ConnectorRegistry
from stream_fusion.costream.audience_comparator import CrossAudienceComparator
from stream_fusion.costream.exceptions import (
    ChannelConnectionError,
    CoStreamError,
    ResourceCeilingExceededError,
)
from stream_fusion.costream.sync_engine import CrossStreamSyncEngine
from stream_fusion.ingest.live_buffer import LiveMediaSegment
from stream_fusion.ingest.live_tailer import LiveStreamIngestor
from stream_fusion.models.schemas import (
    ChatMessage,
    CoStreamChannelConfig,
    CoStreamChannelTelemetry,
    CoStreamSessionConfig,
    CoStreamSessionStatus,
    CrossAudienceSentimentPoint,
    CrossStreamSyncResult,
    LivePlatform,
    LiveState,
    MemeBurstEvent,
)
from stream_fusion.monitoring.broadcaster import LiveEventBroadcaster
from stream_fusion.nlp.adaptive_slang import RollingBurstDetector
from stream_fusion.schema.envelope import StreamFusionEnvelope

logger = logging.getLogger(__name__)


class MultiStreamSupervisor:
    """Supervises a single channel's chat and media ingest within a co-stream session."""

    def __init__(
        self,
        channel_config: CoStreamChannelConfig,
        burst_detector: Optional[RollingBurstDetector] = None,
        max_message_queue_size: int = 2000,
        chat_connector: Optional[BaseChatConnector] = None,
    ):
        self.config = channel_config
        self.channel_id = channel_config.channel_id
        self.platform = channel_config.stream_config.platform
        self.burst_detector = burst_detector or RollingBurstDetector(window_sec=10.0, z_threshold=3.0)
        self.max_message_queue_size = max_message_queue_size

        self.chat_connector = chat_connector or ConnectorRegistry.create_chat_connector(
            config=channel_config.stream_config,
            burst_detector=self.burst_detector,
        )
        self.ingestor = LiveStreamIngestor(config=channel_config.stream_config)

        # Bounded message queue for resource safety
        self.message_queue: asyncio.Queue[ChatMessage] = asyncio.Queue(maxsize=max_message_queue_size)
        self.recent_messages: List[ChatMessage] = []
        self._max_recent_messages = 500

        # Telemetry
        self.state: ConnectorState = ConnectorState.STOPPED
        self.total_messages: int = 0
        self.reconnect_attempts: int = 0
        self.last_error: Optional[str] = None
        self._task: Optional[asyncio.Task] = None

    async def start(
        self,
        on_message_callback: Optional[Callable[[str, ChatMessage], Any]] = None,
        on_burst_callback: Optional[Callable[[str, Any], Any]] = None,
    ) -> None:
        """Starts connector and registers fanout handlers."""
        self.state = ConnectorState.STARTING

        # Wire message listener
        def _handle_msg(msg: ChatMessage):
            self.total_messages += 1
            # Maintain bounded recent message history
            if len(self.recent_messages) >= self._max_recent_messages:
                self.recent_messages.pop(0)
            self.recent_messages.append(msg)

            # Put into queue non-blockingly, dropping oldest if full to prevent OOM
            try:
                self.message_queue.put_nowait(msg)
            except asyncio.QueueFull:
                try:
                    self.message_queue.get_nowait()
                    self.message_queue.put_nowait(msg)
                except Exception:
                    pass

            if on_message_callback:
                try:
                    res = on_message_callback(self.channel_id, msg)
                    if asyncio.iscoroutine(res):
                        asyncio.create_task(res)
                except Exception as ex:
                    logger.debug("Error in message callback for %s: %s", self.channel_id, ex)

        self.chat_connector.register_callback(_handle_msg)

        if on_burst_callback:
            def _handle_burst(burst):
                try:
                    res = on_burst_callback(self.channel_id, burst)
                    if asyncio.iscoroutine(res):
                        asyncio.create_task(res)
                except Exception as ex:
                    logger.debug("Error in burst callback for %s: %s", self.channel_id, ex)
            self.chat_connector.register_burst_callback(_handle_burst)

        # Start connector in task
        try:
            await self.chat_connector.start()
            self.state = ConnectorState.RUNNING
        except Exception as e:
            self.state = ConnectorState.ERROR
            self.last_error = str(e)
            logger.error("Failed to start channel %s: %s", self.channel_id, e)
            raise ChannelConnectionError(f"Channel {self.channel_id} failed: {e}") from e

    async def stop(self) -> None:
        """Stops connector and cleans up resources gracefully."""
        self.state = ConnectorState.STOPPED
        try:
            await self.chat_connector.stop()
        except Exception as ex:
            logger.warning("Error stopping chat connector for %s: %s", self.channel_id, ex)
        try:
            await self.ingestor.stop()
        except Exception as ex:
            logger.warning("Error stopping ingestor for %s: %s", self.channel_id, ex)


class MultiStreamCoordinator:
    """Master orchestrator for concurrent multi-stream co-streaming sessions.
    
    Coordinates multiple channel supervisors across platforms, performs real-time clock
    synchronization, compares cross-platform audience reaction and sentiment, and manages
    bounded resource allocations with clean fault isolation.
    """

    def __init__(
        self,
        config: CoStreamSessionConfig,
        broadcaster: Optional[LiveEventBroadcaster] = None,
        sync_engine: Optional[CrossStreamSyncEngine] = None,
        audience_comparator: Optional[CrossAudienceComparator] = None,
    ):
        self.config = config
        self.session_id = config.session_id
        self.broadcaster = broadcaster or LiveEventBroadcaster(replay_capacity=500)
        self.sync_engine = sync_engine or CrossStreamSyncEngine(
            reference_channel_id=config.reference_channel_id
        )
        self.audience_comparator = audience_comparator or CrossAudienceComparator(
            bucket_window_sec=config.bucket_window_sec
        )

        self._supervisors: Dict[str, MultiStreamSupervisor] = {}
        self._start_time: Optional[float] = None
        self._is_active: bool = False
        self._background_tasks: List[asyncio.Task] = []
        self._aligned_messages_history: List[Tuple[float, str, ChatMessage]] = []
        self._max_history_messages: int = 5000

        # Build supervisors from config
        self._init_supervisors()

    def _init_supervisors(self) -> None:
        """Initializes channel supervisors based on session configuration."""
        ref_id = self.config.reference_channel_id
        for ch_cfg in self.config.channels:
            if ch_cfg.is_reference_stream and not ref_id:
                ref_id = ch_cfg.channel_id
            sup = MultiStreamSupervisor(channel_config=ch_cfg)
            self._supervisors[ch_cfg.channel_id] = sup

            # Apply manual offset if provided
            if ch_cfg.manual_latency_offset is not None:
                self.sync_engine.set_manual_offset(
                    ch_cfg.channel_id,
                    ch_cfg.manual_latency_offset,
                    confidence=1.0,
                )

        if ref_id:
            self.sync_engine.set_reference_channel(ref_id)
        elif self._supervisors:
            first_ch = next(iter(self._supervisors.keys()))
            self.sync_engine.set_reference_channel(first_ch)

    @property
    def is_active(self) -> bool:
        return self._is_active

    @property
    def supervisors(self) -> Dict[str, MultiStreamSupervisor]:
        return self._supervisors

    async def start(self) -> None:
        """Starts all channel supervisors concurrently with fault isolation."""
        if self._is_active:
            return

        self._start_time = time.time()
        self._is_active = True

        logger.info(
            "Starting MultiStreamCoordinator session %s with %d channels...",
            self.session_id,
            len(self._supervisors),
        )

        # Broadcast session start
        await self._broadcast_envelope(
            "COSTREAM_SESSION_STARTED",
            {
                "session_id": self.session_id,
                "title": self.config.session_title,
                "channels": list(self._supervisors.keys()),
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
        )

        # Start channels concurrently; individual channel failure does not halt the entire session
        start_tasks = []
        for ch_id, sup in self._supervisors.items():
            start_tasks.append(
                self._start_channel_isolated(ch_id, sup)
            )

        results = await asyncio.gather(*start_tasks, return_exceptions=True)
        for ch_id, res in zip(self._supervisors.keys(), results):
            if isinstance(res, Exception):
                logger.error("Channel %s encountered startup failure: %s", ch_id, res)

        # Start periodic sync and analysis loop if auto_sync enabled
        if self.config.auto_sync:
            self._background_tasks.append(
                asyncio.create_task(self._periodic_sync_and_analysis_loop())
            )

    async def _start_channel_isolated(self, ch_id: str, sup: MultiStreamSupervisor) -> None:
        """Starts a single channel supervisor safely with exception shielding."""
        try:
            await sup.start(
                on_message_callback=self._on_incoming_message,
                on_burst_callback=self._on_incoming_burst,
            )
        except Exception as ex:
            logger.warning("Fault isolation: channel %s start failure: %s", ch_id, ex)

    def _on_incoming_message(self, channel_id: str, msg: ChatMessage) -> None:
        """Callback invoked when any channel receives a chat message."""
        u_ts = self.sync_engine.transform_timestamp(channel_id, msg.timestamp_offset)
        self._aligned_messages_history.append((u_ts, channel_id, msg))

        # Enforce memory cap on message history
        if len(self._aligned_messages_history) > self._max_history_messages:
            self._aligned_messages_history.pop(0)

        # Broadcast unified envelope
        env = StreamFusionEnvelope[ChatMessage](
            event_type="COSTREAM_MESSAGE",
            stream_id=self.session_id,
            payload=msg,
        )
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.broadcaster.broadcast(env))
        except RuntimeError:
            pass

    def _on_incoming_burst(self, channel_id: str, burst: Any) -> None:
        """Callback invoked when a token burst occurs on a channel."""
        env = StreamFusionEnvelope[Dict[str, Any]](
            event_type="COSTREAM_BURST",
            stream_id=self.session_id,
            payload={
                "channel_id": channel_id,
                "burst": burst.model_dump() if hasattr(burst, "model_dump") else str(burst),
            },
        )
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.broadcaster.broadcast(env))
        except RuntimeError:
            pass

    async def _periodic_sync_and_analysis_loop(self) -> None:
        """Periodic background task that calibrates clock drift and checks for divergence."""
        interval = max(5.0, self.config.sync_interval_sec)
        while self._is_active:
            try:
                await asyncio.sleep(interval)
                if not self._is_active:
                    break

                # 1. Update Cross-Audience Sentiment
                ch_plat = {
                    ch_id: sup.platform.value
                    for ch_id, sup in self._supervisors.items()
                }
                timeline = self.audience_comparator.compute_aligned_sentiment_timeline(
                    self._aligned_messages_history[-500:],
                    channel_to_platform=ch_plat,
                )

                if timeline:
                    latest_point = timeline[-1]
                    if latest_point.divergence_detected:
                        await self._broadcast_envelope(
                            "COSTREAM_DIVERGENCE_ALERT",
                            latest_point.model_dump(),
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in co-stream background analysis loop: %s", e)

    async def _broadcast_envelope(self, event_type: str, payload: Any) -> None:
        """Helper to broadcast envelope to all listening clients."""
        env = StreamFusionEnvelope[Any](
            event_type=event_type,
            stream_id=self.session_id,
            payload=payload,
        )
        await self.broadcaster.broadcast(env)

    async def stop(self) -> None:
        """Gracefully shuts down all channels and cancels background tasks."""
        if not self._is_active:
            return

        logger.info("Stopping MultiStreamCoordinator session %s...", self.session_id)
        self._is_active = False

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()

        # Stop supervisors concurrently with shielding
        stop_tasks = [sup.stop() for sup in self._supervisors.values()]
        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)

        await self._broadcast_envelope(
            "COSTREAM_SESSION_STOPPED",
            {
                "session_id": self.session_id,
                "stopped_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def get_status(self) -> CoStreamSessionStatus:
        """Retrieves real-time status and health telemetry across all co-stream channels."""
        channels_telemetry: Dict[str, CoStreamChannelTelemetry] = {}
        total_msgs = 0

        for ch_id, sup in self._supervisors.items():
            vel = sup.chat_connector.get_message_velocity(window_sec=10.0) if hasattr(sup.chat_connector, "get_message_velocity") else 0.0
            offset = self.sync_engine.get_offset(ch_id)
            conf = self.sync_engine.get_confidence(ch_id)
            total_msgs += sup.total_messages

            channels_telemetry[ch_id] = CoStreamChannelTelemetry(
                channel_id=ch_id,
                platform=sup.platform,
                state=sup.state,
                total_messages_received=sup.total_messages,
                current_message_velocity=round(vel, 2),
                calibrated_latency_offset=offset,
                sync_confidence=conf,
                reconnect_attempts=sup.reconnect_attempts,
                last_error=sup.last_error,
            )

        uptime = time.time() - self._start_time if self._start_time else 0.0

        # Calculate current agreement index from recent history
        ch_plat = {
            ch_id: sup.platform.value
            for ch_id, sup in self._supervisors.items()
        }
        timeline = self.audience_comparator.compute_aligned_sentiment_timeline(
            self._aligned_messages_history[-200:],
            channel_to_platform=ch_plat,
        )
        current_agreement = timeline[-1].cross_platform_agreement if timeline else 1.0

        return CoStreamSessionStatus(
            session_id=self.session_id,
            is_active=self._is_active,
            uptime_sec=round(uptime, 1),
            channels=channels_telemetry,
            reference_channel_id=self.sync_engine.reference_channel_id,
            cross_platform_agreement_index=current_agreement,
            total_messages=total_msgs,
        )
