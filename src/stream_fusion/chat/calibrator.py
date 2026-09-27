"""Dynamic broadcast latency calibrator via signal cross-correlation."""

from typing import List, Optional
import numpy as np

from stream_fusion.models.schemas import AudioSegment, ChatMessage


class LatencyCalibrator:
    """Computes exact broadcast delay offset between video/audio triggers and chat bursts."""

    def __init__(self, min_lag_sec: float = 2.0, max_lag_sec: float = 9.0, default_lag_sec: float = 4.5):
        self.min_lag_sec = min_lag_sec
        self.max_lag_sec = max_lag_sec
        self.default_lag_sec = default_lag_sec

    def compute_optimal_latency(
        self,
        audio_segments: List[AudioSegment],
        chat_messages: List[ChatMessage],
        stream_duration_sec: float,
        resolution_sec: float = 0.5,
    ) -> float:
        """Calculates optimal delay offset tau using discrete cross-correlation."""
        if not audio_segments or len(chat_messages) < 10 or stream_duration_sec <= 10.0:
            return self.default_lag_sec

        num_bins = int(stream_duration_sec / resolution_sec) + 1
        audio_signal = np.zeros(num_bins, dtype=np.float32)
        chat_signal = np.zeros(num_bins, dtype=np.float32)

        # 1. Build audio speech trigger signal
        for seg in audio_segments:
            start_bin = int(seg.start_sec / resolution_sec)
            end_bin = int(seg.end_sec / resolution_sec)
            if start_bin < num_bins:
                # Give higher weight to streamer speech
                weight = 2.0 if seg.speaker_label == "STREAMER" else 1.0
                audio_signal[start_bin : min(end_bin + 1, num_bins)] += weight

        # 2. Build chat density signal
        for msg in chat_messages:
            b = int(msg.timestamp_offset / resolution_sec)
            if b < num_bins:
                chat_signal[b] += 1.0

        # Normalize signals (zero-mean, unit variance)
        audio_std = np.std(audio_signal)
        chat_std = np.std(chat_signal)
        if audio_std < 1e-6 or chat_std < 1e-6:
            return self.default_lag_sec

        norm_audio = (audio_signal - np.mean(audio_signal)) / audio_std
        norm_chat = (chat_signal - np.mean(chat_signal)) / chat_std

        # 3. Cross-correlation over candidate lags
        min_lag_bins = int(self.min_lag_sec / resolution_sec)
        max_lag_bins = int(self.max_lag_sec / resolution_sec)

        best_corr = -1.0
        best_lag_sec = self.default_lag_sec

        for lag_bin in range(min_lag_bins, max_lag_bins + 1):
            # Chat is delayed: chat_signal[t + lag] correlates with audio_signal[t]
            if lag_bin < len(norm_chat):
                valid_len = len(norm_audio) - lag_bin
                if valid_len > 10:
                    corr = float(np.sum(norm_audio[:valid_len] * norm_chat[lag_bin : lag_bin + valid_len]))
                    if corr > best_corr:
                        best_corr = corr
                        best_lag_sec = lag_bin * resolution_sec

        return best_lag_sec

    def compute_rolling_latency_curve(
        self,
        audio_segments: List[AudioSegment],
        chat_messages: List[ChatMessage],
        stream_duration_sec: float,
        window_size_sec: float = 300.0,
        step_sec: float = 150.0,
    ) -> List[dict]:
        """Computes time-varying latency delay offset over time to track broadcast buffer drift."""
        if stream_duration_sec <= window_size_sec:
            base_lag = self.compute_optimal_latency(audio_segments, chat_messages, stream_duration_sec)
            return [{"window_center_sec": stream_duration_sec / 2.0, "latency_offset_sec": base_lag}]

        num_windows = int((stream_duration_sec - window_size_sec) // step_sec) + 1
        curve = []
        for i in range(num_windows):
            w_start = i * step_sec
            w_end = min(stream_duration_sec, w_start + window_size_sec)
            sub_audio = [
                s for s in audio_segments
                if max(w_start, s.start_sec) < min(w_end, s.end_sec)
            ]
            sub_chat = [
                m for m in chat_messages
                if w_start <= m.timestamp_offset <= w_end
            ]
            lag = self.compute_optimal_latency(
                sub_audio, sub_chat, stream_duration_sec=(w_end - w_start)
            )
            curve.append(
                {
                    "window_center_sec": (w_start + w_end) / 2.0,
                    "latency_offset_sec": lag,
                }
            )
        return curve
