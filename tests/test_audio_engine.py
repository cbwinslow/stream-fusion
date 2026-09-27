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


def test_pyannote_failure_fallback_without_recursion():
    wav_path = Path(__file__).parent / "fixtures" / "sample_reaction_audio.wav"
    # An invalid token or pyannote import failure should gracefully fall back to acoustic profiling
    diarizer = ReactionDiarizer(hf_token="invalid_hf_token_12345")
    segments = [
        AudioSegment(
            segment_id=1,
            start_sec=0.5,
            end_sec=3.5,
            speaker_label="UNKNOWN",
            transcript="I cannot believe this is happening",
        ),
    ]
    tagged = diarizer.diarize_and_tag(segments, wav_path)
    assert len(tagged) == 1
    assert tagged[0].speaker_label in ["STREAMER", "EXTERNAL_VIDEO"]
    diarizer.unload()


def test_diarizer_edge_cases(tmp_path: Path):
    diarizer = ReactionDiarizer()
    assert diarizer.diarize_and_tag([], Path("dummy.wav")) == []

    # Out of range timestamp profiling
    wav_path = Path(__file__).parent / "fixtures" / "sample_reaction_audio.wav"
    prof = diarizer._extract_segment_acoustic_profile(wav_path, start_sec=999.0, end_sec=1005.0)
    assert prof["rms"] == 0.0
    assert prof["spectral_centroid"] == 0.0


def test_transcriber_missing_file():
    transcriber = AudioTranscriber(model_size="tiny", device="cpu")
    with pytest.raises(FileNotFoundError):
        transcriber.transcribe(Path("missing_audio_track.wav"))

