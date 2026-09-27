"""Unit tests for ASS karaoke subtitle generation."""

from pathlib import Path
from stream_fusion.models.schemas import WordTiming
from stream_fusion.export.subtitles import format_ass_timestamp, generate_karaoke_ass


def test_format_ass_timestamp():
    assert format_ass_timestamp(0.0) == "0:00:00.00"
    assert format_ass_timestamp(4.5) == "0:00:04.50"
    assert format_ass_timestamp(65.25) == "0:01:05.25"
    assert format_ass_timestamp(3661.12) == "1:01:01.12"


def test_generate_karaoke_ass(tmp_path: Path):
    words = [
        WordTiming(word="Wait", start=10.0, end=10.4),
        WordTiming(word="hold", start=10.4, end=10.7),
        WordTiming(word="on", start=10.7, end=10.9),
        WordTiming(word="a", start=10.9, end=11.1),
        WordTiming(word="second.", start=11.1, end=11.6),
        WordTiming(word="Bro", start=12.0, end=12.3),
        WordTiming(word="are", start=12.3, end=12.5),
        WordTiming(word="you", start=12.5, end=12.7),
        WordTiming(word="serious?", start=12.7, end=13.2),
    ]

    out_ass = tmp_path / "test_karaoke.ass"
    res = generate_karaoke_ass(
        words=words,
        output_path=out_ass,
        base_offset_sec=10.0,
        max_words_per_line=5,
    )

    assert res.exists()
    content = res.read_text(encoding="utf-8")

    # Verify script headers
    assert "[Script Info]" in content
    assert "PlayResX: 1080" in content
    assert "PlayResY: 1920" in content
    assert "[V4+ Styles]" in content
    assert "Style: Karaoke" in content

    # Verify dialogue and \k timings
    assert "[Events]" in content
    assert "Dialogue: 0," in content
    assert r"{\k" in content
    assert "Wait" in content
    assert "second." in content
    assert "serious?" in content


def test_generate_karaoke_ass_empty(tmp_path: Path):
    out_ass = tmp_path / "empty.ass"
    res = generate_karaoke_ass([], out_ass)
    assert res.exists()
    content = res.read_text(encoding="utf-8")
    assert "[Script Info]" in content
    assert "Dialogue:" not in content
