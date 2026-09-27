"""Unit tests for Speaker Voiceprint Library & Co-Stream Diarization (Spec 11)."""

from pathlib import Path
import numpy as np
import pytest

from stream_fusion.audio.voiceprint import (
    CoStreamDiarizer,
    SpeakerEmbeddingExtractor,
    VoiceprintLibrary,
    cosine_similarity,
)
from stream_fusion.models.schemas import AudioSegment


def test_cosine_similarity():
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    v3 = [0.0, 1.0, 0.0]
    assert cosine_similarity(v1, v2) == pytest.approx(1.0)
    assert cosine_similarity(v1, v3) == pytest.approx(0.0)


def test_snr_and_audio_quality():
    extractor = SpeakerEmbeddingExtractor(embedding_dim=64, snr_threshold_db=12.0)

    # Clean signal + low noise floor
    t = np.linspace(0, 1.0, 16000)
    clean_audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    passes, snr = extractor.check_audio_quality(clean_audio)
    assert passes
    assert snr >= 12.0

    # Clipped audio (> 0.999)
    clipped_audio = np.ones(1000)
    passes, _ = extractor.check_audio_quality(clipped_audio)
    assert not passes


def test_embedding_extractor_l2_norm():
    extractor = SpeakerEmbeddingExtractor(embedding_dim=64)
    audio = np.random.uniform(-0.5, 0.5, 16000).astype(np.float32)
    emb = extractor.extract_embedding(audio)
    assert emb is not None
    assert len(emb) == 64
    norm = np.linalg.norm(emb)
    assert norm == pytest.approx(1.0, rel=1e-3)


def test_voiceprint_library_enrollment_and_matching(tmp_path: Path):
    reg_path = tmp_path / "voiceprints.json"
    library = VoiceprintLibrary(storage_path=reg_path)

    # 1. Enroll TheBurntPeanut
    peanut_emb = [1.0] + [0.0] * 63
    library.enroll_creator(
        creator_id="theburntpeanut",
        display_name="TheBurntPeanut",
        embedding=peanut_emb,
        primary_channel="twitch.tv/theburntpeanut",
    )

    # 2. Enroll HutchMF
    hutch_emb = [0.0, 1.0] + [0.0] * 62
    library.enroll_creator(
        creator_id="hutchmf",
        display_name="HutchMF",
        embedding=hutch_emb,
        primary_channel="twitch.tv/hutchmf",
    )

    # Verify disk persistence
    assert reg_path.exists()
    reloaded = VoiceprintLibrary(storage_path=reg_path)
    assert "theburntpeanut" in reloaded.profiles
    assert "hutchmf" in reloaded.profiles

    # 3. Test Host Identification
    match_host = reloaded.identify_speaker(peanut_emb, expected_host_id="theburntpeanut")
    assert match_host.is_known_creator
    assert match_host.assigned_label == "STREAMER:theburntpeanut"
    assert match_host.creator_id == "theburntpeanut"

    # 4. Test Co-Streamer Identification
    match_guest = reloaded.identify_speaker(hutch_emb, expected_host_id="theburntpeanut")
    assert match_guest.is_known_creator
    assert match_guest.assigned_label == "CO_STREAMER:hutchmf"
    assert match_guest.creator_id == "hutchmf"

    # 5. Unknown Voice
    unknown_emb = [0.0, 0.0, 1.0] + [0.0] * 61
    match_unknown = reloaded.identify_speaker(unknown_emb, expected_host_id="theburntpeanut")
    assert not match_unknown.is_known_creator
    assert match_unknown.assigned_label == "UNKNOWN"


def test_costream_diarizer_workflow(tmp_path: Path):
    reg_path = tmp_path / "voiceprints.json"
    library = VoiceprintLibrary(storage_path=reg_path)

    peanut_emb = [1.0] + [0.0] * 63
    hutch_emb = [0.0, 1.0] + [0.0] * 62
    library.enroll_creator("theburntpeanut", "TheBurntPeanut", peanut_emb)
    library.enroll_creator("hutchmf", "HutchMF", hutch_emb)

    diarizer = CoStreamDiarizer(library=library)

    # Audio segments from a DayZ co-stream on TheBurntPeanut's channel
    segments = [
        AudioSegment(segment_id=1, start_sec=10.0, end_sec=15.0, speaker_label="SPEAKER_00", transcript="Where are we meeting Hutch?"),
        AudioSegment(segment_id=2, start_sec=16.0, end_sec=22.0, speaker_label="SPEAKER_01", transcript="I am over by the military base."),
        AudioSegment(segment_id=3, start_sec=25.0, end_sec=30.0, speaker_label="SPEAKER_01", transcript="Watch out, there are zombies behind you."),
    ]

    embeddings = {
        1: peanut_emb,
        2: hutch_emb,
        3: hutch_emb,
    }

    attributed, interactions = diarizer.attribute_segments(
        segments,
        embeddings,
        host_creator_id="theburntpeanut",
        vod_id="vod_dayz_101",
    )

    assert len(attributed) == 3
    assert attributed[0].speaker_label == "STREAMER:theburntpeanut"
    assert attributed[1].speaker_label == "CO_STREAMER:hutchmf"
    assert attributed[2].speaker_label == "CO_STREAMER:hutchmf"

    assert len(interactions) == 1
    interaction = interactions[0]
    assert interaction.host_creator == "theburntpeanut"
    assert interaction.guest_creator == "hutchmf"
    assert interaction.interaction_start_sec == 16.0
    assert interaction.interaction_end_sec == 30.0
    assert interaction.total_spoken_duration_sec > 0
