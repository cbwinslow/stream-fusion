"""Unit tests for the StreamFusion Typer CLI."""

from pathlib import Path
from typer.testing import CliRunner

from stream_fusion.cli import app

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Multimodal Livestream VOD & Chat Grounding and Analysis CLI" in result.output
    assert "process" in result.output
    assert "clip" in result.output
    assert "demo" in result.output


def test_cli_process_help():
    result = runner.invoke(app, ["process", "--help"])
    assert result.exit_code == 0
    assert "--latency-offset" in result.output
    assert "--auto-latency" in result.output
    assert "--chunk-duration" in result.output


def test_cli_clip_help():
    result = runner.invoke(app, ["clip", "--help"])
    assert result.exit_code == 0
    assert "--start" in result.output
    assert "--end" in result.output
    assert "--out" in result.output


def test_cli_demo_execution(tmp_path: Path):
    out_html = tmp_path / "demo_report.html"
    result = runner.invoke(app, ["demo", "--out", str(out_html)])
    assert result.exit_code == 0
    assert "StreamFusion Demonstration Mode" in result.output
    assert out_html.exists()
    assert out_html.stat().st_size > 500


def test_cli_chat_nlp():
    fixture_chat = Path(__file__).parent / "fixtures" / "sample_twitch_chat.json"
    result = runner.invoke(app, ["chat-nlp", str(fixture_chat), "--min-authors", "2"])
    assert result.exit_code == 0
    assert "Chat NLP" in result.output
    assert "Community Emotional Intent Distribution" in result.output


def test_cli_sponsor():
    fixture_chat = Path(__file__).parent / "fixtures" / "sample_twitch_chat.json"
    result = runner.invoke(app, ["sponsor", "TestBrand", "--chat", str(fixture_chat), "--start", "0", "--end", "20"])
    assert result.exit_code == 0
    assert "Sponsor Impact Analysis" in result.output
    assert "Brand Attention Score" in result.output

