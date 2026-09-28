"""Circular Segment Buffer for Bounded Live Media Storage (Spec 19)."""

import os
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class LiveMediaSegment(BaseModel):
    """Represents a bounded chunk/slice of live stream media."""
    segment_id: int
    start_sec: float
    end_sec: float
    duration_sec: float
    video_path: Optional[str] = None
    audio_path: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    byte_size: int = 0


class CircularSegmentBuffer:
    """Bounded circular FIFO buffer for live streaming audio/video segments.
    
    Automatically tracks total buffered duration and unlinks expired files from
    disk when capacity exceeds `buffer_duration_sec` or `max_bytes`.
    """

    def __init__(
        self,
        buffer_duration_sec: float = 120.0,
        max_bytes: int = 250 * 1024 * 1024,  # 250 MB
        auto_unlink: bool = True,
    ):
        self.buffer_duration_sec = buffer_duration_sec
        self.max_bytes = max_bytes
        self.auto_unlink = auto_unlink
        self._segments: List[LiveMediaSegment] = []
        self._lock = threading.Lock()
        self._total_bytes: int = 0

    @property
    def segment_count(self) -> int:
        with self._lock:
            return len(self._segments)

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return self._total_bytes

    def get_buffered_duration(self) -> float:
        """Returns the difference between the newest end_sec and oldest start_sec."""
        with self._lock:
            if not self._segments:
                return 0.0
            return max(0.0, self._segments[-1].end_sec - self._segments[0].start_sec)

    def add_segment(
        self,
        segment_id: int,
        start_sec: float,
        end_sec: float,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LiveMediaSegment:
        """Adds a new media segment and evicts expired segments beyond buffer_duration."""
        duration = max(0.0, end_sec - start_sec)
        byte_size = 0
        if video_path and os.path.isfile(video_path):
            byte_size += os.path.getsize(video_path)
        if audio_path and os.path.isfile(audio_path):
            byte_size += os.path.getsize(audio_path)

        seg = LiveMediaSegment(
            segment_id=segment_id,
            start_sec=start_sec,
            end_sec=end_sec,
            duration_sec=duration,
            video_path=video_path,
            audio_path=audio_path,
            metadata=metadata or {},
            byte_size=byte_size,
        )

        with self._lock:
            self._segments.append(seg)
            self._total_bytes += byte_size
            self._evict_expired_locked()

        return seg

    def _evict_expired_locked(self) -> None:
        """Evicts oldest segments until both duration and byte constraints are met."""
        if not self._segments:
            return

        latest_end = self._segments[-1].end_sec

        while len(self._segments) > 1:
            oldest = self._segments[0]
            current_span = latest_end - oldest.start_sec
            over_duration = current_span > self.buffer_duration_sec
            over_bytes = self._total_bytes > self.max_bytes

            if over_duration or over_bytes:
                evicted = self._segments.pop(0)
                self._total_bytes = max(0, self._total_bytes - evicted.byte_size)
                if self.auto_unlink:
                    self._unlink_files(evicted)
            else:
                break

    def _unlink_files(self, seg: LiveMediaSegment) -> None:
        """Safely unlinks segment files if they exist."""
        for path_str in [seg.video_path, seg.audio_path]:
            if path_str:
                try:
                    p = Path(path_str)
                    if p.is_file():
                        p.unlink(missing_ok=True)
                except OSError:
                    pass

    def get_segments_in_range(self, start_sec: float, end_sec: float) -> List[LiveMediaSegment]:
        """Returns copies of all segments overlapping [start_sec, end_sec]."""
        with self._lock:
            return [
                s for s in self._segments
                if not (s.end_sec < start_sec or s.start_sec > end_sec)
            ]

    def get_all_segments(self) -> List[LiveMediaSegment]:
        """Returns snapshot list of all buffered segments."""
        with self._lock:
            return list(self._segments)

    def clear(self) -> None:
        """Clears all buffered segments and unlinks associated files."""
        with self._lock:
            if self.auto_unlink:
                for seg in self._segments:
                    self._unlink_files(seg)
            self._segments.clear()
            self._total_bytes = 0
