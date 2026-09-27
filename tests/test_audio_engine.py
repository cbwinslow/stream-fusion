"""Tests for AudioTranscriber and ReactionDiarizer."""

from pathlib import Path
import pytest
from stream_fusion.models.schemas import AudioSegment
from stream_fusion.audio.diarizer import ReactionDiarizer
from stream_fusion.audio.transcriber import AudioTranscriber


def test_acoustic_profile_diarization():
    wav_path = Path(__file__).parent / "fixtures" / "sample_reaction_audio.wav"
    assert wav_path.exists(), "Audio fixture missing"

    diarizer = ReactionDiarizer(hf_token=None)

    # Simulated segments: Seg 1 (0.5 to 3.5s - Streamer voice) vs Seg 2 (4.5 to 7.5s - Video audio)
    segments = [
        AudioSegment(
            segment_id=1,
            start_sec=0.5,
            end_sec=3.5,
            speaker_label="UNKNOWN",
            transcript="I cannot believe this is happening",
        ),
        AudioSegment(
            segment_id=2,
            start_sec=4.5,
            end_sec=7.5,
            speaker_label="UNKNOWN",
            transcript="Background video soundtrack playing",
        ),
    ]

    tagged_segments = diarizer.diarize_and_tag(segments, wav_path)

    assert len(tagged_segments) == 2
    # Verify Seg 1 (higher RMS, lower spectral centroid) tagged as STREAMER
    assert tagged_segments[0].speaker_label == "STREAMER"
    # Verify Seg 2 tagged as EXTERNAL_VIDEO
    assert tagged_segments[1].speaker_label == "EXTERNAL_VIDEO"

    # Test unload / cleanup
    diarizer.unload()


def test_audio_transcriber_initialization_and_device():
    transcriber = AudioTranscriber(model_size="tiny", device="cpu", compute_type="int8")
    assert transcriber.device == "cpu"
    assert transcriber.compute_type == "int8"
    assert transcriber._model is None  # lazy loading

    # Verify unloading works safely even if model was not loaded
    transcriber.unload()
    assert transcriber._model is None
