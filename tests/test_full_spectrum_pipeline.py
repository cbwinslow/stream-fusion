"""Unit and Integration Test Suite for Spec 24: Full-Spectrum Pipeline Unification."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from typer.testing import CliRunner

from stream_fusion.cli import app
from stream_fusion.config import StreamFusionConfig
from stream_fusion.models.schemas import (
    AudioSegment,
    ChatMessage,
    FullSpectrumConfig,
    FullSpectrumManifest,
    FusionSlice,
    GroundedClaimResult,
    ShortProductionPackage,
    SponsorImpactReport,
    SponsorSegment,
    StageExecutionStatus,
    StreamAnalysisResult,
    StreamerClaim,
    VisualKeyframe,
)
from stream_fusion.orchestration.full_spectrum import FullSpectrumPipeline
from stream_fusion.orchestration.manifest import FullSpectrumManifestBuilder
from stream_fusion.schema.agent_rpc import AgentRpcDispatcher


# --- Test 1: FullSpectrumManifestBuilder ---

def test_manifest_builder(tmp_path: Path):
    builder = FullSpectrumManifestBuilder(stream_id="test_stream", output_directory=tmp_path)
    builder.record_stage("phase_1_demuxing", StageExecutionStatus.SUCCESS, duration_sec=1.23, output_summary="Demuxed audio")
    builder.record_stage("phase_2_audio_diarization", StageExecutionStatus.DEGRADED, duration_sec=2.45, error_message="Soft warning")

    # Mock analysis result
    analysis = StreamAnalysisResult(
        stream_id="test_stream",
        duration_sec=30.0,
        total_chat_messages=10,
        slices=[
            FusionSlice(
                bucket_index=0,
                start_sec=0.0,
                end_sec=5.0,
                agreement_score=0.8,
            )
        ],
        metadata={"latency_offset_sec": 4.5},
    )

    builder.populate_from_analysis(
        analysis,
        effective_media_sec=30.0,
        total_audio_segments=1,
        total_keyframes=1,
    )
    saved_path = builder.save()

    assert saved_path.exists()
    with open(saved_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["stream_id"] == "test_stream"
    assert data["effective_media_duration_sec"] == 30.0
    assert data["total_audio_segments"] == 1
    assert data["total_keyframes"] == 1
    assert data["calibrated_broadcast_delay_sec"] == 4.5
    assert "phase_1_demuxing" in data["stages"]
    assert data["stages"]["phase_1_demuxing"]["status"] == "SUCCESS"
    assert data["stages"]["phase_2_audio_diarization"]["status"] == "DEGRADED"


# --- Test 2: FullSpectrumPipeline with Mocked Subsystems ---

def test_full_spectrum_pipeline_mocked_execution(tmp_path: Path):
    fixture_video = Path(__file__).parent / "fixtures" / "sample_test_vod.mp4"
    fixture_chat = Path(__file__).parent / "fixtures" / "sample_twitch_chat.json"

    # Configure pipeline with mock-friendly settings
    cfg = FullSpectrumConfig(
        isolate_gpu_workers=False,
        bounded_buffer=False,
        enable_adaptive_slang=True,
        enable_web_grounding=True,
        enable_stance_tracking=True,
        enable_sponsor_quantifier=True,
        enable_short_production=True,
        short_candidate_count=2,
        dry_run_shorts=True,
        stance_history_db=str(tmp_path / "stance.json"),
        adaptive_lexicon_db=str(tmp_path / "slang.json"),
    )

    base_cfg = StreamFusionConfig()
    base_cfg.audio.whisper_model = "tiny"
    base_cfg.audio.device = "cpu"
    base_cfg.vision.device = "cpu"

    pipeline = FullSpectrumPipeline(config=cfg, base_config=base_cfg)

    # Mock demuxer and neural components for instant test execution
    pipeline.demuxer.extract_audio_16k_mono = MagicMock(return_value=tmp_path / "mock.wav")
    pipeline.demuxer.extract_frames_at_interval = MagicMock(return_value=[tmp_path / "frame1.jpg"])

    mock_audio = [
        AudioSegment(segment_id=1, start_sec=1.0, end_sec=5.0, speaker_label="STREAMER", transcript="This new Blizzard game is awesome W W W based"),
        AudioSegment(segment_id=2, start_sec=6.0, end_sec=9.0, speaker_label="STREAMER", transcript="Sponsored by Starforge Systems gaming PC"),
    ]
    mock_frames = [
        VisualKeyframe(frame_index=1, timestamp_sec=2.0, scene_type="REACT_VIDEO", screen_summary="Blizzard game trailer playing on stream"),
    ]

    with patch("stream_fusion.orchestration.full_spectrum.AudioTranscriber") as mock_trans:
        mock_trans.return_value.transcribe.return_value = mock_audio
        with patch("stream_fusion.orchestration.full_spectrum.ReactionDiarizer") as mock_diar:
            mock_diar.return_value.diarize_and_tag.return_value = mock_audio
            with patch("stream_fusion.orchestration.full_spectrum.VisionProcessor") as mock_vp:
                mock_vp.return_value.process_frame.return_value = mock_frames[0]

                analysis, manifest = pipeline.run(
                    media_input=fixture_video,
                    chat_input=fixture_chat,
                    output_dir=tmp_path / "full_spectrum_out",
                    duration_sec=10.0,
                )

    assert analysis.stream_id == "sample_test_vod"
    assert manifest.stream_id == "sample_test_vod"
    assert manifest.total_pipeline_duration_sec > 0.0

    # Verify all 9 stages ran and succeeded
    expected_stages = [
        "phase_1_demuxing",
        "phase_2_audio_diarization",
        "phase_3_vision_ocr",
        "phase_4_chat_and_slang",
        "phase_5_fusion_matrix",
        "phase_6_sponsor_quantifier",
        "phase_7_knowledge_and_stance",
        "phase_8_multi_agent_shorts",
        "phase_9_packaging",
    ]
    for stg in expected_stages:
        assert stg in manifest.stages
        assert manifest.stages[stg].status in (StageExecutionStatus.SUCCESS, StageExecutionStatus.DEGRADED)

    # Check that manifest file was created on disk
    manifest_file = tmp_path / "full_spectrum_out" / "sample_test_vod_full_spectrum_manifest.json"
    assert manifest_file.exists()


# --- Test 3: Fault Isolation / Degraded Stage Resilience ---

def test_full_spectrum_degraded_stage_resilience(tmp_path: Path):
    fixture_video = Path(__file__).parent / "fixtures" / "sample_test_vod.mp4"

    cfg = FullSpectrumConfig(
        enable_adaptive_slang=True,
        enable_web_grounding=True,
        enable_sponsor_quantifier=True,
        enable_short_production=True,
        dry_run_shorts=True,
    )
    pipeline = FullSpectrumPipeline(config=cfg)

    # Mock demuxer and neural components
    pipeline.demuxer.extract_audio_16k_mono = MagicMock(return_value=tmp_path / "mock.wav")
    pipeline.demuxer.extract_frames_at_interval = MagicMock(return_value=[tmp_path / "frame1.jpg"])

    mock_audio = [AudioSegment(segment_id=1, start_sec=1.0, end_sec=5.0, speaker_label="STREAMER", transcript="Hello")]
    mock_frames = [VisualKeyframe(frame_index=1, timestamp_sec=2.0, scene_type="GAMEPLAY", screen_summary="Screen")]

    # Deliberately make sponsor quantifier and short orchestrator fail to test graceful degradation
    pipeline.sponsor_quantifier.analyze_sponsors = MagicMock(side_effect=RuntimeError("Sponsor module failure"))
    pipeline.short_orchestrator.produce_shorts = MagicMock(side_effect=RuntimeError("Short studio render failure"))

    with patch("stream_fusion.orchestration.full_spectrum.AudioTranscriber") as mock_trans:
        mock_trans.return_value.transcribe.return_value = mock_audio
        with patch("stream_fusion.orchestration.full_spectrum.ReactionDiarizer") as mock_diar:
            mock_diar.return_value.diarize_and_tag.return_value = mock_audio
            with patch("stream_fusion.orchestration.full_spectrum.VisionProcessor") as mock_vp:
                mock_vp.return_value.process_frame.return_value = mock_frames[0]

                # Pipeline should complete without raising, marking degraded stages
                analysis, manifest = pipeline.run(
                    media_input=fixture_video,
                    output_dir=tmp_path / "resilient_out",
                    duration_sec=10.0,
                )

    assert manifest.stages["phase_6_sponsor_quantifier"].status == StageExecutionStatus.DEGRADED
    assert "Sponsor module failure" in (manifest.stages["phase_6_sponsor_quantifier"].error_message or "")
    assert manifest.stages["phase_8_multi_agent_shorts"].status == StageExecutionStatus.DEGRADED
    assert manifest.stages["phase_5_fusion_matrix"].status == StageExecutionStatus.SUCCESS


# --- Test 4: AgentRpcDispatcher Full-Spectrum RPC Methods ---

def test_agent_rpc_full_spectrum_handlers(tmp_path: Path):
    dispatcher = AgentRpcDispatcher()

    # Create dummy manifest file
    manifest_data = {
        "manifest_id": "fsm-test1234",
        "stream_id": "test_stream",
        "total_pipeline_duration_sec": 42.5,
        "stages": {},
    }
    manifest_file = tmp_path / "test_manifest.json"
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    # Query manifest via RPC
    req = {
        "jsonrpc": "2.0",
        "method": "streamfusion.getFullSpectrumManifest",
        "params": {"manifest_path": str(manifest_file)},
        "id": 999,
    }
    resp_str = dispatcher.handle_line(json.dumps(req))
    resp = json.loads(resp_str)

    assert "result" in resp
    assert resp["result"]["manifest_id"] == "fsm-test1234"
    assert resp["result"]["total_pipeline_duration_sec"] == 42.5


# --- Test 5: CLI full-spectrum & run-all Help Commands ---

def test_cli_full_spectrum_help():
    runner = CliRunner()
    result = runner.invoke(app, ["full-spectrum", "--help"])
    assert result.exit_code == 0
    assert "unified 9-phase" in result.output.lower() or "synergy" in result.output.lower()
    assert "--shorts" in result.output
    assert "--dry-run-shorts" in result.output
    assert "--ground-claims" in result.output

    res_run_all = runner.invoke(app, ["run-all", "--help"])
    assert res_run_all.exit_code == 0
    assert "Convenience alias" in res_run_all.output
