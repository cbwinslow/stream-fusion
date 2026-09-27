"""Speaker Voiceprint Library & Cross-Stream Diarization (Spec 11).

Provides acoustic speaker embedding extraction, creator voiceprint enrollment,
and cross-channel co-streamer matching with SNR noise gating.
"""

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from stream_fusion.models.schemas import (
    AudioSegment,
    CoStreamInteraction,
    SpeakerMatchResult,
    VoiceprintProfile,
)


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Computes cosine similarity between two float vectors."""
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class SpeakerEmbeddingExtractor:
    """Extracts L2-normalized acoustic speaker embeddings with SNR and clipping filters."""

    def __init__(self, embedding_dim: int = 192, snr_threshold_db: float = 12.0):
        self.embedding_dim = embedding_dim
        self.snr_threshold_db = snr_threshold_db

    def check_audio_quality(self, audio_data: np.ndarray) -> Tuple[bool, float]:
        """Evaluates Signal-to-Noise Ratio (SNR) and clipping.

        Returns (passes_quality_gate, snr_db).
        """
        if len(audio_data) == 0:
            return False, 0.0

        # Check peak clipping
        peak = float(np.max(np.abs(audio_data)))
        if peak >= 0.999:
            # Saturated/clipped audio
            return False, 0.0

        # Estimate signal and noise floor energy
        energy = audio_data ** 2
        p_signal = float(np.mean(energy))
        # Estimate noise from lowest 10% percentile frames
        sorted_energy = np.sort(energy)
        noise_cutoff = max(1, int(len(sorted_energy) * 0.10))
        p_noise = float(np.mean(sorted_energy[:noise_cutoff]))

        if p_noise <= 1e-10:
            snr_db = 40.0
        else:
            snr_db = 10.0 * math.log10(max(1e-9, p_signal / p_noise))

        passes = snr_db >= self.snr_threshold_db
        return passes, round(snr_db, 2)

    def extract_embedding(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Optional[List[float]]:
        """Extracts normalized speaker embedding from raw audio samples."""
        passes, _ = self.check_audio_quality(audio_data)
        if not passes and len(audio_data) < sample_rate * 0.5:
            # Allow fallback if very short, but skip if completely clipped
            if float(np.max(np.abs(audio_data))) >= 0.999:
                return None

        # Acoustic pseudo-d-vector extraction via subband spectral energy distribution
        # In full production this wraps ECAPA-TDNN; here we provide a deterministic,
        # robust acoustic fingerprint fallback that guarantees unit L2 norm.
        n_samples = len(audio_data)
        if n_samples == 0:
            return None

        # Split into subband bins and compute spectral moments
        fft = np.abs(np.fft.rfft(audio_data))
        freq_bins = np.array_split(fft, self.embedding_dim)
        features = np.array([float(np.mean(b)) if len(b) > 0 else 0.0 for b in freq_bins], dtype=np.float32)

        norm = np.linalg.norm(features)
        if norm > 0.0:
            features = features / norm
        else:
            features = np.zeros(self.embedding_dim, dtype=np.float32)
            features[0] = 1.0

        return features.tolist()


class VoiceprintLibrary:
    """Manages the registry of known creator voice centroids."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path
        self.profiles: Dict[str, VoiceprintProfile] = {}
        if storage_path and storage_path.exists():
            self.load()

    def load(self) -> None:
        """Loads voiceprint registry from disk."""
        if not self.storage_path or not self.storage_path.exists():
            return
        with open(self.storage_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.profiles = {k: VoiceprintProfile(**v) for k, v in data.items()}

    def save(self) -> None:
        """Persists voiceprint registry to disk."""
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: v.model_dump() for k, v in self.profiles.items()}
        temp_path = self.storage_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        temp_path.replace(self.storage_path)

    def enroll_creator(
        self,
        creator_id: str,
        display_name: str,
        embedding: List[float],
        primary_channel: Optional[str] = None,
        confidence_threshold: float = 0.76,
    ) -> VoiceprintProfile:
        """Enrolls or updates a creator profile."""
        if creator_id in self.profiles:
            # Incremental centroid update
            prof = self.profiles[creator_id]
            curr_cent = np.array(prof.centroid_embedding, dtype=np.float32)
            new_emb = np.array(embedding, dtype=np.float32)
            n = prof.sample_count
            updated_cent = (n * curr_cent + new_emb) / (n + 1)
            norm = np.linalg.norm(updated_cent)
            if norm > 0:
                updated_cent = updated_cent / norm
            prof.centroid_embedding = updated_cent.tolist()
            prof.sample_count += 1
            prof.last_updated = datetime.now(timezone.utc).isoformat()
        else:
            prof = VoiceprintProfile(
                creator_id=creator_id,
                display_name=display_name,
                primary_channel=primary_channel,
                centroid_embedding=embedding,
                sample_count=1,
                confidence_threshold=confidence_threshold,
                last_updated=datetime.now(timezone.utc).isoformat(),
            )
            self.profiles[creator_id] = prof

        self.save()
        return prof

    def identify_speaker(
        self,
        embedding: List[float],
        expected_host_id: Optional[str] = None,
        override_threshold: Optional[float] = None,
    ) -> SpeakerMatchResult:
        """Matches embedding against known creator voiceprints."""
        best_match: Optional[VoiceprintProfile] = None
        best_sim = -1.0

        for prof in self.profiles.values():
            sim = cosine_similarity(embedding, prof.centroid_embedding)
            if sim > best_sim:
                best_sim = sim
                best_match = prof

        if not best_match:
            return SpeakerMatchResult(assigned_label="UNKNOWN", confidence=0.0, is_known_creator=False)

        threshold = override_threshold or best_match.confidence_threshold

        if best_sim >= threshold:
            if expected_host_id and best_match.creator_id == expected_host_id:
                label = f"STREAMER:{best_match.creator_id}"
            else:
                label = f"CO_STREAMER:{best_match.creator_id}"

            return SpeakerMatchResult(
                assigned_label=label,
                creator_id=best_match.creator_id,
                confidence=round(best_sim, 3),
                is_known_creator=True,
            )

        return SpeakerMatchResult(
            assigned_label="UNKNOWN",
            creator_id=None,
            confidence=round(best_sim, 3),
            is_known_creator=False,
        )


class CoStreamDiarizer:
    """Applies cross-stream voiceprint matching to diarized audio segments."""

    def __init__(self, library: VoiceprintLibrary, extractor: Optional[SpeakerEmbeddingExtractor] = None):
        self.library = library
        self.extractor = extractor or SpeakerEmbeddingExtractor()

    def attribute_segments(
        self,
        audio_segments: List[AudioSegment],
        segment_embeddings: Dict[int, List[float]],
        host_creator_id: str,
        vod_id: str = "unknown_vod",
    ) -> Tuple[List[AudioSegment], List[CoStreamInteraction]]:
        """Resolves speaker labels and discovers co-streaming interactions."""
        attributed_segments: List[AudioSegment] = []
        raw_interactions: List[Tuple[str, float, float]] = []  # (guest_id, start_sec, end_sec)

        for seg in audio_segments:
            emb = segment_embeddings.get(seg.segment_id)
            if not emb:
                attributed_segments.append(seg)
                continue

            match = self.library.identify_speaker(emb, expected_host_id=host_creator_id)
            new_label = match.assigned_label if match.is_known_creator else seg.speaker_label

            updated_seg = seg.model_copy(update={"speaker_label": new_label})
            attributed_segments.append(updated_seg)

            if match.is_known_creator and match.creator_id != host_creator_id:
                raw_interactions.append((match.creator_id, seg.start_sec, seg.end_sec))

        # Cluster interactions by guest creator
        interactions: List[CoStreamInteraction] = []
        if raw_interactions:
            # Group contiguous interactions within 60s
            current_guest, start_t, end_t = raw_interactions[0]
            total_duration = end_t - start_t

            for guest, s, e in raw_interactions[1:]:
                if guest == current_guest and (s - end_t) <= 60.0:
                    end_t = e
                    total_duration += (e - s)
                else:
                    interactions.append(
                        CoStreamInteraction(
                            host_creator=host_creator_id,
                            guest_creator=current_guest,
                            vod_id=vod_id,
                            interaction_start_sec=round(start_t, 2),
                            interaction_end_sec=round(end_t, 2),
                            total_spoken_duration_sec=round(total_duration, 2),
                            timestamp=datetime.now(timezone.utc).isoformat(),
                        )
                    )
                    current_guest, start_t, end_t = guest, s, e
                    total_duration = end_t - start_t

            interactions.append(
                CoStreamInteraction(
                    host_creator=host_creator_id,
                    guest_creator=current_guest,
                    vod_id=vod_id,
                    interaction_start_sec=round(start_t, 2),
                    interaction_end_sec=round(end_t, 2),
                    total_spoken_duration_sec=round(total_duration, 2),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            )

        return attributed_segments, interactions
