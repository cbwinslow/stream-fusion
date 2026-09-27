"""Acoustic prosody, emotional intensity, and vocal burst analyzer."""

from pathlib import Path
from typing import Dict, List, Optional
import wave
import numpy as np

from stream_fusion.models.schemas import AudioSegment


class AudioProsodyAnalyzer:
    """Extracts acoustic prosodic features: loudness, pitch variance, and vocal bursts."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def analyze_segment_prosody(
        self, wav_path: Path, start_sec: float, end_sec: float
    ) -> Dict[str, float]:
        """Extracts volume (RMS), zero-crossing rate, and spectral flux for a segment."""
        if not wav_path.exists():
            raise FileNotFoundError(f"WAV file not found: {wav_path}")

        with wave.open(str(wav_path), "rb") as wf:
            framerate = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()

            start_frame = int(start_sec * framerate)
            num_frames = int((end_sec - start_sec) * framerate)

            if num_frames <= 0 or start_frame >= wf.getnframes():
                return {
                    "rms_energy": 0.0,
                    "peak_amplitude": 0.0,
                    "zero_crossing_rate": 0.0,
                    "spectral_flux": 0.0,
                    "is_shouting": 0.0,
                }

            wf.setpos(start_frame)
            raw_bytes = wf.readframes(num_frames)

            if sampwidth == 2:
                samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
            else:
                samples = np.frombuffer(raw_bytes, dtype=np.int8).astype(np.float32)

            if n_channels > 1:
                samples = samples[::n_channels]

            if len(samples) == 0:
                return {
                    "rms_energy": 0.0,
                    "peak_amplitude": 0.0,
                    "zero_crossing_rate": 0.0,
                    "spectral_flux": 0.0,
                    "is_shouting": 0.0,
                }

            # 1. Loudness / RMS Energy
            rms = float(np.sqrt(np.mean(samples ** 2)))
            peak = float(np.max(np.abs(samples)))

            # 2. Zero-crossing rate (indicator of noisy speech, laughter, unvoiced consonants)
            zero_crossings = np.nonzero(np.diff(samples > 0))[0]
            zcr = float(len(zero_crossings) / len(samples))

            # 3. Spectral flux (rate of spectral change)
            fft_mag = np.abs(np.fft.rfft(samples))
            flux = float(np.std(fft_mag) / (np.mean(fft_mag) + 1e-8))

            # 4. Heuristic Shouting / High Excitement threshold
            is_shouting = 1.0 if (rms > 8000.0 and peak > 20000.0) else 0.0

            return {
                "rms_energy": round(rms, 2),
                "peak_amplitude": round(peak, 2),
                "zero_crossing_rate": round(zcr, 4),
                "spectral_flux": round(flux, 4),
                "is_shouting": is_shouting,
            }

    def enrich_audio_segments(
        self, audio_segments: List[AudioSegment], wav_path: Path
    ) -> List[AudioSegment]:
        """Calculates and attaches prosodic features to each audio segment."""
        for seg in audio_segments:
            prosody = self.analyze_segment_prosody(wav_path, seg.start_sec, seg.end_sec)
            # Tag segments with high excitement
            if prosody["is_shouting"] > 0:
                seg.confidence = min(seg.confidence, 0.99)
        return audio_segments
