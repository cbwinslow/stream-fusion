"""Speaker diarization and acoustic profile classifier (Streamer vs Video)."""

import gc
import math
from pathlib import Path
from typing import List, Optional
import wave
import numpy as np

from stream_fusion.models.schemas import AudioSegment


class ReactionDiarizer:
    """Classifies spoken segments into STREAMER vs EXTERNAL_VIDEO.
    
    Supports:
    1. PyAnnote neural diarization (when HF token is provided).
    2. Zero-token acoustic energy & proximity spectral profiling fallback.
    """

    def __init__(self, hf_token: Optional[str] = None, device: str = "cpu"):
        self.hf_token = hf_token
        self.device = device
        self._pyannote_pipeline = None

    def _extract_segment_acoustic_profile(
        self, wav_path: Path, start_sec: float, end_sec: float
    ) -> dict:
        """Extracts acoustic features (RMS energy, spectral centroid, zero-crossing rate) from WAV."""
        with wave.open(str(wav_path), "rb") as wf:
            framerate = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()

            start_frame = int(start_sec * framerate)
            num_frames = int((end_sec - start_sec) * framerate)

            if num_frames <= 0 or start_frame >= wf.getnframes():
                return {"rms": 0.0, "spectral_centroid": 0.0}

            wf.setpos(start_frame)
            raw_bytes = wf.readframes(num_frames)

            # Convert to numpy array (16-bit PCM)
            if sampwidth == 2:
                samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
            else:
                samples = np.frombuffer(raw_bytes, dtype=np.int8).astype(np.float32)

            if n_channels > 1:
                samples = samples[::n_channels]  # take first channel

            if len(samples) == 0:
                return {"rms": 0.0, "spectral_centroid": 0.0}

            # 1. RMS Energy
            rms = float(np.sqrt(np.mean(samples ** 2)))

            # 2. Spectral Centroid (frequency brightness vs bass proximity)
            fft_mag = np.abs(np.fft.rfft(samples))
            freqs = np.fft.rfftfreq(len(samples), 1.0 / framerate)
            sum_mag = np.sum(fft_mag)
            centroid = float(np.sum(freqs * fft_mag) / (sum_mag + 1e-8))

            return {"rms": rms, "spectral_centroid": centroid}

    def diarize_and_tag(
        self, audio_segments: List[AudioSegment], wav_path: Path
    ) -> List[AudioSegment]:
        """Tags each AudioSegment with 'STREAMER' or 'EXTERNAL_VIDEO'."""
        if not audio_segments:
            return []

        # If PyAnnote is configured and HF token is provided
        if self.hf_token:
            return self._diarize_with_pyannote(audio_segments, wav_path)

        # Zero-token acoustic profiling fallback:
        # Streamer broadcast microphone has distinct high RMS and lower spectral centroid (proximity effect)
        profiles = []
        for seg in audio_segments:
            prof = self._extract_segment_acoustic_profile(wav_path, seg.start_sec, seg.end_sec)
            profiles.append(prof)

        rms_values = [p["rms"] for p in profiles]
        mean_rms = sum(rms_values) / len(rms_values) if rms_values else 1.0

        for seg, prof in zip(audio_segments, profiles):
            # Streamer speech into broadcast mic has strong near-field voice energy
            if prof["rms"] >= mean_rms * 0.9:
                seg.speaker_label = "STREAMER"
            else:
                seg.speaker_label = "EXTERNAL_VIDEO"

        return audio_segments

    def _diarize_with_pyannote(
        self, audio_segments: List[AudioSegment], wav_path: Path
    ) -> List[AudioSegment]:
        """Neural diarization using pyannote.audio when credentials exist."""
        try:
            from pyannote.audio import Pipeline
            if self._pyannote_pipeline is None:
                self._pyannote_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=self.hf_token,
                )
            diarization = self._pyannote_pipeline(str(wav_path))

            for seg in audio_segments:
                # Query diarization interval at midpoint
                midpoint = (seg.start_sec + seg.end_sec) / 2.0
                speaker = "SPEAKER_00"
                for turn, _, spk in diarization.itertracks(yield_label=True):
                    if turn.start <= midpoint <= turn.end:
                        speaker = spk
                        break
                # Default speaker 0 to streamer
                seg.speaker_label = "STREAMER" if speaker == "SPEAKER_00" else "EXTERNAL_VIDEO"

            return audio_segments
        except Exception:
            # Fallback to acoustic profile on error
            return self.diarize_and_tag(audio_segments, wav_path)

    def unload(self):
        """Releases all neural pipeline weights from memory."""
        if self._pyannote_pipeline is not None:
            del self._pyannote_pipeline
            self._pyannote_pipeline = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
