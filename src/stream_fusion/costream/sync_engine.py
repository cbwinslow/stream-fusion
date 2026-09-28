"""Cross-Stream Temporal Synchronization and Clock Drift Alignment Engine (Spec 23)."""

from datetime import datetime, timezone
import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from stream_fusion.costream.exceptions import StreamSynchronizationError
from stream_fusion.models.schemas import (
    AudioSegment,
    ChatMessage,
    CrossStreamSyncResult,
)

logger = logging.getLogger(__name__)


class CrossStreamSyncEngine:
    """Synchronizes multiple live stream channels onto a canonical temporal reference clock.
    
    Computes pairwise latency offsets between a master reference channel and co-stream channels
    using discrete cross-correlation over acoustic/chat signals, common trigger event matching,
    or manual operator calibration.
    """

    def __init__(
        self,
        reference_channel_id: Optional[str] = None,
        max_search_lag_sec: float = 15.0,
        sample_step_sec: float = 0.5,
    ):
        self.reference_channel_id = reference_channel_id
        self.max_search_lag_sec = max_search_lag_sec
        self.sample_step_sec = sample_step_sec
        self._offsets: Dict[str, float] = {}
        self._confidence_scores: Dict[str, float] = {}

    def set_reference_channel(self, channel_id: str) -> None:
        """Sets or updates the primary reference channel."""
        self.reference_channel_id = channel_id
        self._offsets[channel_id] = 0.0
        self._confidence_scores[channel_id] = 1.0

    def set_manual_offset(
        self,
        channel_id: str,
        offset_sec: float,
        confidence: float = 1.0,
    ) -> None:
        """Manually specifies the latency offset for a channel relative to reference."""
        self._offsets[channel_id] = float(offset_sec)
        self._confidence_scores[channel_id] = max(0.0, min(1.0, float(confidence)))
        logger.info(
            "Manual offset set for channel %s: %.3fs (confidence: %.2f)",
            channel_id,
            offset_sec,
            confidence,
        )

    def get_offset(self, channel_id: str) -> float:
        """Gets calibrated latency offset in seconds for a channel (0.0 if reference or unknown)."""
        return self._offsets.get(channel_id, 0.0)

    def get_confidence(self, channel_id: str) -> float:
        """Gets synchronization confidence score [0.0, 1.0] for a channel."""
        return self._confidence_scores.get(channel_id, 0.0)

    def transform_timestamp(self, channel_id: str, local_ts_sec: float) -> float:
        """Transforms a channel-local timestamp into canonical unified epoch time.
        
        Formula: T_unified = local_ts_sec - offset_sec
        """
        offset = self.get_offset(channel_id)
        return local_ts_sec - offset

    def calibrate_offsets(
        self,
        channel_signals: Dict[str, Sequence[float]],
        reference_channel_id: Optional[str] = None,
        sample_step_sec: Optional[float] = None,
        max_lag_sec: Optional[float] = None,
    ) -> CrossStreamSyncResult:
        """Calculates latency offsets across channels via discrete cross-correlation.
        
        Args:
            channel_signals: Dict mapping channel_id to time series signal (e.g., chat velocity or RMS energy).
            reference_channel_id: Channel ID to use as anchor (defaults to self.reference_channel_id).
            sample_step_sec: Time duration per signal bin in seconds.
            max_lag_sec: Maximum latency search window (+/- seconds).
            
        Returns:
            CrossStreamSyncResult containing calibrated offsets and confidence scores.
        """
        ref_id = reference_channel_id or self.reference_channel_id
        if not ref_id:
            if not channel_signals:
                raise StreamSynchronizationError("Cannot calibrate offsets: No channel signals provided.")
            ref_id = next(iter(channel_signals.keys()))

        self.set_reference_channel(ref_id)
        step = sample_step_sec or self.sample_step_sec
        max_lag = max_lag_sec or self.max_search_lag_sec

        ref_signal = np.array(channel_signals.get(ref_id, []), dtype=np.float64)
        if len(ref_signal) == 0:
            raise StreamSynchronizationError(
                f"Reference channel '{ref_id}' has no signal data for calibration."
            )

        max_lag_samples = int(math.ceil(max_lag / step))

        for ch_id, raw_sig in channel_signals.items():
            if ch_id == ref_id:
                self._offsets[ch_id] = 0.0
                self._confidence_scores[ch_id] = 1.0
                continue

            target_sig = np.array(raw_sig, dtype=np.float64)
            if len(target_sig) == 0:
                self._offsets[ch_id] = 0.0
                self._confidence_scores[ch_id] = 0.0
                continue

            offset, confidence = self._compute_pairwise_cross_correlation(
                ref_signal,
                target_sig,
                max_lag_samples=max_lag_samples,
                step_sec=step,
            )
            self._offsets[ch_id] = round(offset, 4)
            self._confidence_scores[ch_id] = round(confidence, 4)

        return CrossStreamSyncResult(
            reference_channel_id=ref_id,
            channel_offsets=dict(self._offsets),
            confidence_scores=dict(self._confidence_scores),
            sync_method="CROSS_CORRELATION",
            timestamp=datetime.now(timezone.utc),
        )

    def _compute_pairwise_cross_correlation(
        self,
        ref_signal: np.ndarray,
        target_signal: np.ndarray,
        max_lag_samples: int,
        step_sec: float,
    ) -> Tuple[float, float]:
        """Calculates optimal lag maximizing normalized cross-correlation between two signals."""
        min_len = min(len(ref_signal), len(target_signal))
        if min_len < 3:
            return 0.0, 0.0

        r_cut = ref_signal[:min_len]
        t_cut = target_signal[:min_len]

        # Zero-mean normalization
        r_std = np.std(r_cut)
        t_std = np.std(t_cut)

        if r_std < 1e-6 or t_std < 1e-6:
            # Flat signal - cannot establish meaningful correlation
            return 0.0, 0.1

        r_norm = (r_cut - np.mean(r_cut)) / (r_std * len(r_cut))
        t_norm = (t_cut - np.mean(t_cut)) / t_std

        # Cross correlation via numpy
        corr = np.correlate(t_norm, r_norm, mode="full")
        lags = np.arange(-len(r_norm) + 1, len(t_norm))

        # Restrict to search window
        valid_indices = np.where((lags >= -max_lag_samples) & (lags <= max_lag_samples))[0]
        if len(valid_indices) == 0:
            return 0.0, 0.0

        sub_corr = corr[valid_indices]
        sub_lags = lags[valid_indices]

        best_idx = int(np.argmax(sub_corr))
        best_lag = sub_lags[best_idx]
        best_corr = float(sub_corr[best_idx])

        # Lag in seconds
        offset_sec = float(best_lag * step_sec)
        confidence = max(0.0, min(1.0, best_corr))

        return offset_sec, confidence

    def calibrate_from_trigger_events(
        self,
        channel_triggers: Dict[str, Sequence[float]],
        reference_channel_id: Optional[str] = None,
        max_association_sec: float = 8.0,
    ) -> CrossStreamSyncResult:
        """Calibrates offsets by matching sparse shared trigger event timestamps across streams.
        
        Args:
            channel_triggers: Dict mapping channel_id to sorted lists of detected trigger timestamps.
            reference_channel_id: Anchor channel ID.
            max_association_sec: Maximum time difference to associate two triggers as identical event.
        """
        ref_id = reference_channel_id or self.reference_channel_id
        if not ref_id or ref_id not in channel_triggers:
            ref_id = next(iter(channel_triggers.keys()))
        self.set_reference_channel(ref_id)

        ref_events = sorted(channel_triggers.get(ref_id, []))
        if not ref_events:
            raise StreamSynchronizationError(
                f"Reference channel '{ref_id}' has no trigger events for calibration."
            )

        for ch_id, events in channel_triggers.items():
            if ch_id == ref_id:
                self._offsets[ch_id] = 0.0
                self._confidence_scores[ch_id] = 1.0
                continue

            target_events = sorted(events)
            if not target_events:
                self._offsets[ch_id] = 0.0
                self._confidence_scores[ch_id] = 0.0
                continue

            # Compute nearest neighbor pairwise differences
            diffs: List[float] = []
            for t_ref in ref_events:
                # Find closest event in target_events
                closest = min(target_events, key=lambda t: abs(t - t_ref))
                diff = closest - t_ref
                if abs(diff) <= max_association_sec:
                    diffs.append(diff)

            if diffs:
                median_offset = float(np.median(diffs))
                # Confidence proportional to proportion of matched events and consistency (low MAD)
                match_ratio = len(diffs) / len(ref_events)
                mad = float(np.median(np.abs(np.array(diffs) - median_offset)))
                consistency = max(0.0, 1.0 - (mad / max_association_sec))
                confidence = float(np.clip(match_ratio * consistency, 0.0, 1.0))

                self._offsets[ch_id] = round(median_offset, 4)
                self._confidence_scores[ch_id] = round(confidence, 4)
            else:
                self._offsets[ch_id] = 0.0
                self._confidence_scores[ch_id] = 0.0

        return CrossStreamSyncResult(
            reference_channel_id=ref_id,
            channel_offsets=dict(self._offsets),
            confidence_scores=dict(self._confidence_scores),
            sync_method="TRIGGER_EVENT_MATCHING",
            timestamp=datetime.now(timezone.utc),
        )

    def align_messages(
        self,
        messages_by_channel: Dict[str, Sequence[ChatMessage]],
    ) -> List[Tuple[float, str, ChatMessage]]:
        """Transforms and aligns all messages into a unified chronologically sorted stream.
        
        Returns:
            List of tuples: (unified_timestamp_sec, channel_id, normalized_chat_message)
        """
        unified: List[Tuple[float, str, ChatMessage]] = []

        for ch_id, msgs in messages_by_channel.items():
            offset = self.get_offset(ch_id)
            for m in msgs:
                u_ts = m.timestamp_offset - offset
                unified.append((u_ts, ch_id, m))

        unified.sort(key=lambda item: item[0])
        return unified

    def align_audio_segments(
        self,
        segments_by_channel: Dict[str, Sequence[AudioSegment]],
    ) -> List[Tuple[float, float, str, AudioSegment]]:
        """Transforms audio segments onto the canonical timeline.
        
        Returns:
            List of tuples: (unified_start_sec, unified_end_sec, channel_id, audio_segment)
        """
        unified: List[Tuple[float, float, str, AudioSegment]] = []

        for ch_id, segs in segments_by_channel.items():
            offset = self.get_offset(ch_id)
            for s in segs:
                u_start = s.start_sec - offset
                u_end = s.end_sec - offset
                unified.append((u_start, u_end, ch_id, s))

        unified.sort(key=lambda item: item[0])
        return unified
