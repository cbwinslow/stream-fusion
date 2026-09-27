"""Tests for AudioProsodyAnalyzer."""

from pathlib import Path
import pytest
from stream_fusion.models.schemas import AudioSegment
from stream_fusion.audio.prosody import AudioProsodyAnalyzer


def test_analyze_segment_prosody():
    wav_path = Path(__file__).parent / "fixtures" / "sample_reaction_audio.wav"
    assert wav_path.exists()

    analyzer = AudioProsodyAnalyzer()
    # Segment 1 (0-3s) has high amplitude sine wave
    features_loud = analyzer.analyze_segment_prosody(wav_path, start_sec=0.5, end_sec=3.0)
    assert features_loud["rms_energy"] > 10000.0
    assert features_loud["peak_amplitude"] > 20000.0

    # Segment 2 (4-7s) has low amplitude sine wave
    features_quiet = analyzer.analyze_segment_prosody(wav_path, start_sec=4.5, end_sec=7.0)
    assert features_quiet["rms_energy"] < features_loud["rms_energy"]
    assert features_quiet["peak_amplitude"] < features_loud["peak_amplitude"]


def test_prosody_missing_file():
    analyzer = AudioProsodyAnalyzer()
    with pytest.raises(FileNotFoundError):
        analyzer.analyze_segment_prosody(Path("non_existent.wav"), 0.0, 1.0)
