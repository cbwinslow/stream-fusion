"""Centralized Abstract Base Classes for Live Chat & Stream Connectors (Spec 22)."""

from abc import ABC, abstractmethod
import asyncio
from datetime import datetime, timezone
import logging
import random
import time
from typing import Any, Callable, Dict, List, Optional

from stream_fusion.models.schemas import (
    ChatMessage,
    ConnectorState,
    LivePlatform,
    PlatformCapabilities,
)
from stream_fusion.nlp.adaptive_slang import RollingBurstDetector

logger = logging.getLogger(__name__)


class BaseChatConnector(ABC):
    """Unified Abstract Base Class for live chat platform connectors.
    
    Provides centralized connection state tracking, exponential backoff with jitter,
    sliding-window message velocity calculation, burst detector integration, and callback fanout.
    Subclasses only need to implement platform-specific wire protocol connections and framing.
    """

    def __init__(
        self,
        channel_name: str,
        platform: LivePlatform,
        burst_detector: Optional[RollingBurstDetector] = None,
        stream_start_ms: Optional[float] = None,
        max_reconnect_attempts: int = 10,
        base_reconnect_delay: float = 2.0,
        max_reconnect_delay: float = 60.0,
    ):
        self.channel_name = channel_name.strip()
        self.platform = platform
        self.burst_detector = burst_detector
        self.stream_start_ms = stream_start_ms or (time.time() * 1000.0)
        self.max_reconnect_attempts = max_reconnect_attempts
        self.base_reconnect_delay = base_reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay

        self._state: ConnectorState = ConnectorState.STOPPED
        self._running: bool = False
        self._task: Optional[asyncio.Task] = None
        self._callbacks: List[Callable[[ChatMessage], Any]] = []
        self._burst_callbacks: List[Callable[[Any], Any]] = []

        # Centralized metrics & sliding-window tracking
        self._total_received: int = 0
        self._total_reconnects: int = 0
        self._reconnect_attempts: int = 0
        self._recent_timestamps: List[float] = []
        self._rolling_messages: List[ChatMessage] = []
        self._last_error: Optional[str] = None

    @property
    @abstractmethod
    def capabilities(self) -> PlatformCapabilities:
        """Returns the platform-specific feature capabilities."""
        pass

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def state(self) -> ConnectorState:
        return self._state

    @property
    def total_messages(self) -> int:
        return self._total_received

    @property
    def total_reconnects(self) -> int:
        return self._total_reconnects

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def register_callback(self, cb: Callable[[ChatMessage], Any]) -> None:
        """Subscribes a callback to receive normalized incoming ChatMessage objects."""
        self._callbacks.append(cb)

    def register_burst_callback(self, cb: Callable[[Any], Any]) -> None:
        """Subscribes a callback to receive detected token / meme bursts."""
        self._burst_callbacks.append(cb)

    def get_message_velocity(self, window_sec: float = 10.0) -> float:
        """Calculates current chat arrival rate in messages per second over a sliding window."""
        now = time.time()
        self._recent_timestamps = [t for t in self._recent_timestamps if now - t <= window_sec]
        if not self._recent_timestamps:
            return 0.0
        return len(self._recent_timestamps) / max(window_sec, 1.0)

    async def ingest_normalized_message(self, message: ChatMessage) -> None:
        """Centralized processing and dispatch for a normalized ChatMessage.
        
        Invoked by platform subclasses whenever a raw wire message is decoded.
        """
        now = time.time()
        self._recent_timestamps.append(now)
        self._total_received += 1

        # Centralized Burst Detection
        if self.burst_detector:
            self._rolling_messages.append(message)
            cutoff = message.timestamp_offset - (self.burst_detector.window_sec * 2)
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
                            logger.error(f"[{self.platform}] Error in burst callback: {b_err}")

        # Centralized Callback Fanout
        for cb in self._callbacks:
            try:
                res = cb(message)
                if asyncio.iscoroutine(res):
                    await res
            except Exception as err:
                logger.error(f"[{self.platform}] Error in chat callback: {err}")

    async def start(self) -> None:
        """Starts the connector background listening task."""
        if self._running:
            return
        self._running = True
        self._state = ConnectorState.STARTING
        self._reconnect_attempts = 0
        self._task = asyncio.create_task(self._supervision_loop())

    async def stop(self) -> None:
        """Gracefully disconnects and shuts down the background task."""
        self._running = False
        self._state = ConnectorState.STOPPED
        try:
            await self._disconnect()
        except Exception as e:
            logger.debug(f"[{self.platform}] Disconnect error during stop: {e}")

        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    async def _supervision_loop(self) -> None:
        """Supervision loop managing connection lifecycle with exponential backoff and jitter."""
        while self._running:
            try:
                self._state = ConnectorState.RUNNING
                self._reconnect_attempts = 0
                await self._connect_and_listen()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._last_error = str(e)
                if not self._running:
                    break

                self._total_reconnects += 1
                self._reconnect_attempts += 1
                self._state = ConnectorState.RECONNECTING

                if self._reconnect_attempts > self.max_reconnect_attempts:
                    logger.error(
                        f"[{self.platform}] Max reconnect attempts ({self.max_reconnect_attempts}) exceeded: {e}"
                    )
                    self._state = ConnectorState.ERROR
                    break

                # Exponential backoff with jitter
                delay = min(
                    self.max_reconnect_delay,
                    self.base_reconnect_delay * (2 ** (self._reconnect_attempts - 1)),
                )
                jitter = random.uniform(0.0, 0.5 * delay)
                sleep_sec = delay + jitter
                logger.warning(
                    f"[{self.platform}] Connection error: {e}. Reconnecting in {sleep_sec:.2f}s "
                    f"(attempt {self._reconnect_attempts}/{self.max_reconnect_attempts})..."
                )
                await asyncio.sleep(sleep_sec)

        if self._state != ConnectorState.ERROR:
            self._state = ConnectorState.STOPPED

    @abstractmethod
    async def _connect_and_listen(self) -> None:
        """Protocol-specific connection and streaming loop. Subclasses implement this."""
        pass

    @abstractmethod
    async def _disconnect(self) -> None:
        """Protocol-specific socket/stream teardown. Subclasses implement this."""
        pass
