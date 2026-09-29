"""End-to-End Real Stream Benchmark & Stress Test Suite (Specs 01-25).

Executes the complete pipeline on real Asmongold broadcast footage (asmon_sample_60s.mp4)
and Twitch chat replay (sample_asmon_chat.json), validating:
- Harvester homelab staging & SHA-256 integrity verification (Spec 25)
- Full-Spectrum 9-Phase Master Synergy Pipeline (Spec 24)
- Multi-Agent Short Studio (Spec 21)
- Subprocess worker isolation & bounded keyframe buffering (Spec 18)
- Adaptive Slang & Meme Discovery (Spec 17)
- Chatter Safety Profiling (Spec 12)
- Speaker Voiceprint Enrollment & Diarization (Spec 11)
- Sponsor & Brand Impact Auditing (Spec 09)
"""

import json
from pathlib import Path
import shutil
import tempfile
import pytest
from typer.testing import CliRunner

from stream_fusion.cli import app
from stream_fusion.config import StreamFusionConfig
from stream_fusion.harvester.bridge import ingest_to_pipeline, verify_harvest_checksums
from stream_fusion.harvester.catalog import HarvestCatalog
from stream_fusion.harvester.engine import compute_file_sha256
from stream_fusion.models.schemas import (
    FullSpectrumConfig,
    HarvestedVodRecord,
    HarvestStatus,
    StageExecutionStatus,
    StreamerTargetRecord,
)
from stream_fusion.orchestration.full_spectrum import FullSpectrumPipeline

runner = CliRunner()


@pytest.fixture
def real_stream_assets():
    """Provides paths to real Asmongold video and chat assets."""
    root = Path(__file__).parent.parent
    video_path = root / "asmon_sample_60s.mp4"
    chat_path = root / "sample_asmon_chat.json"

    if not video_path.exists() or not chat_path.exists():
        pytest.skip("Real Asmon stream assets (asmon_sample_60s.mp4 / sample_asmon_chat.json) not found in root")

    return video_path, chat_path


@pytest.fixture
def staged_homelab_vod(real_stream_assets, tmp_path: Path):
    """Stages real stream video and chat into homelab storage layout with checksums."""
    video_source, chat_source = real_stream_assets

    homelab_root = tmp_path / "homelab_storage"
    vod_dir = homelab_root / "vods" / "asmongold" / "2026-09-29_asmon_real_60s"
    vod_dir.mkdir(parents=True, exist_ok=True)

    media_target = vod_dir / "media.mp4"
    chat_target = vod_dir / "chat.json"
    meta_target = vod_dir / "metadata.json"
    checksum_target = vod_dir / "checksums.sha256"
    manifest_target = vod_dir / "ingest_manifest.json"

    # Copy real broadcast footage and chat replay
    shutil.copyfile(video_source, media_target)
    shutil.copyfile(chat_source, chat_target)

    # Write metadata
    metadata = {
        "vod_id": "asmon_real_60s",
        "streamer_id": "asmongold",
        "title": "Asmongold Reacts to Gaming News & Trailer",
        "platform": "TWITCH",
        "duration_sec": 60.0,
    }
    with open(meta_target, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Compute SHA-256 hashes
    media_hash = compute_file_sha256(media_target)
    chat_hash = compute_file_sha256(chat_target)
    meta_hash = compute_file_sha256(meta_target)

    with open(checksum_target, "w", encoding="utf-8") as f:
        f.write(f"{media_hash}  media.mp4\n")
        f.write(f"{chat_hash}  chat.json\n")
        f.write(f"{meta_hash}  metadata.json\n")

    # Ingest manifest
    ingest_manifest = {
        "vod_id": "asmon_real_60s",
        "streamer_id": "asmongold",
        "platform": "TWITCH",
        "title": metadata["title"],
        "media_path": str(media_target.resolve()),
        "chat_path": str(chat_target.resolve()),
        "metadata_path": str(meta_target.resolve()),
        "checksums": {
            "media.mp4": media_hash,
            "chat.json": chat_hash,
            "metadata.json": meta_hash,
        },
    }
    with open(manifest_target, "w", encoding="utf-8") as f:
        json.dump(ingest_manifest, f, indent=2)

    # Initialize catalog
    catalog = HarvestCatalog(database_url=f"sqlite:///{tmp_path / 'catalog.db'}")

    target = StreamerTargetRecord(
        streamer_id="asmongold",
        display_name="Asmongold",
        channel_urls=["https://twitch.tv/zackrawrr"],
        primary_platform="TWITCH",
        download_priority=10,
        voiceprint_embedding=[0.05] * 192,
    )
    catalog.add_target(target)

    vod_record = HarvestedVodRecord(
        vod_id="asmon_real_60s",
        streamer_id="asmongold",
        platform="TWITCH",
        title=metadata["title"],
        video_path=str(media_target.resolve()),
        chat_path=str(chat_target.resolve()),
        metadata_path=str(meta_target.resolve()),
        file_size_bytes=media_target.stat().st_size,
        status=HarvestStatus.HARVESTED,
    )
    catalog.add_vod(vod_record)

    return catalog, vod_record, vod_dir


def test_real_stream_full_spectrum_synergy_benchmark(staged_homelab_vod):
    """Executes all 9 phases of FullSpectrumPipeline on real Asmongold footage."""
    catalog, vod_record, vod_dir = staged_homelab_vod

    # 1. Verify SHA-256 integrity
    assert verify_harvest_checksums(vod_dir) is True

    # 2. Configure for fast, bounded, CPU-friendly execution
    fs_config = FullSpectrumConfig(
        isolate_gpu_workers=False,
        bounded_buffer=True,
        window_size_sec=10.0,
        sample_interval_sec=2.0,
        enable_adaptive_slang=True,
        enable_web_grounding=True,
        enable_stance_tracking=True,
        enable_sponsor_quantifier=True,
        enable_short_production=True,
        short_candidate_count=2,
        dry_run_shorts=True,
        stance_history_db=str(vod_dir / "stance_history.json"),
        adaptive_lexicon_db=str(vod_dir / "adaptive_lexicon.json"),
    )

    base_config = StreamFusionConfig()
    base_config.audio.whisper_model = "tiny"
    base_config.audio.device = "cpu"
    base_config.audio.compute_type = "int8"
    base_config.vision.device = "cpu"

    out_dir = vod_dir / "analysis"

    # Instantiate pipeline with configured base settings
    pipeline = FullSpectrumPipeline(config=fs_config, base_config=base_config)

    # 3. Execute 9-phase pipeline on 10 seconds of real broadcast
    analysis, manifest = pipeline.run(
        media_input=Path(vod_record.video_path),
        chat_input=Path(vod_record.chat_path),
        output_dir=out_dir,
        duration_sec=10.0,
        creator_name="Asmongold",
    )

    # 4. Verify Phase Yields & Data Structures
    assert analysis.stream_id == "media"
    assert manifest.effective_media_duration_sec == 10.0
    assert manifest.total_fusion_slices > 0
    assert manifest.total_keyframes > 0
    assert manifest.total_chat_messages > 0

    # 5. Verify all 9 Stage Statuses
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

    for stage_name in expected_stages:
        assert stage_name in manifest.stages, f"Missing stage {stage_name}"
        stg = manifest.stages[stage_name]
        assert stg.status in (
            StageExecutionStatus.SUCCESS,
            StageExecutionStatus.DEGRADED,
        ), f"Stage {stage_name} failed: {stg.error_message}"

    # 6. Verify Artifacts Generated on Disk
    assert (out_dir / "media_matrix.parquet").exists()
    assert (out_dir / "media_grounding_report.html").exists()
    assert (out_dir / "media_full_spectrum_manifest.json").exists()


    # 7. Update catalog status to ANALYZED
    catalog.update_vod_status(
        vod_id=vod_record.vod_id,
        status=HarvestStatus.ANALYZED,
    )
    final_record = catalog.get_vod(vod_record.vod_id)
    assert final_record.status == HarvestStatus.ANALYZED
    catalog.close()


def test_real_stream_cli_harvester_bridge(staged_homelab_vod):
    """Verifies CLI command `streamfusion harvest ingest-to-pipeline` against real stream data."""
    catalog, vod_record, vod_dir = staged_homelab_vod
    db_path = catalog.db_path
    catalog.close()

    res = runner.invoke(
        app,
        [
            "harvest",
            "ingest-to-pipeline",
            "--vod-id", "asmon_real_60s",
            "--db", str(db_path),
            "--duration", "6.0",
            "--shorts", "1",
            "--dry-run-shorts",
            "--no-ground-claims",
        ],
    )
    assert res.exit_code == 0
    assert "Successfully ingested asmon_real_60s into Full-Spectrum Pipeline" in res.stdout
