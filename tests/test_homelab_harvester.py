"""Unit and Integration Tests for Spec 25: Homelab Harvester & Targeted Streamer Ingestion."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
from typing import Any, Dict
import pytest
from typer.testing import CliRunner

from stream_fusion.audio.voiceprint import VoiceprintLibrary
from stream_fusion.chat.profiler import ChatterProfileStore
from stream_fusion.cli import app
from stream_fusion.harvester.bridge import ingest_to_pipeline, verify_harvest_checksums
from stream_fusion.harvester.catalog import HarvestCatalog
from stream_fusion.harvester.crawler import ChannelVodCrawler
from stream_fusion.harvester.engine import HomelabHarvester, compute_file_sha256
from stream_fusion.harvester.roster import RosterLoader
from stream_fusion.harvester.synergy import (
    audit_harvested_sponsors,
    feed_adaptive_slang,
    ingest_chatter_safety,
    seed_voiceprints,
)
from stream_fusion.models.schemas import (
    AudioSegment,
    BrandProfile,
    ChatMessage,
    HarvestedVodRecord,
    HarvestStatus,
    StreamerTargetRecord,
)

from stream_fusion.nlp.adaptive_slang import AdaptiveLexiconStore, AdaptiveSlangEngine
from stream_fusion.schema.agent_rpc import AgentRpcDispatcher

runner = CliRunner()


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def memory_catalog():
    cat = HarvestCatalog(database_url=":memory:")
    yield cat
    cat.close()


# =========================================================================
# 1. RosterLoader Tests (YAML, JSON, CSV)
# =========================================================================

def test_roster_yaml_roundtrip(temp_dir: Path):
    yaml_file = temp_dir / "roster.yaml"
    targets = [
        StreamerTargetRecord(
            streamer_id="asmon",
            display_name="Asmongold",
            channel_urls=["https://twitch.tv/asmon", "https://youtube.com/@zackrawrr"],
            primary_platform="TWITCH",
            download_priority=10,
            tags=["reaction", "gaming"],
        ),
        StreamerTargetRecord(
            streamer_id="kai",
            display_name="Kai Cenat",
            channel_urls=["https://twitch.tv/kaicenat"],
            primary_platform="TWITCH",
            download_priority=8,
            tags=["irl"],
        ),
    ]

    RosterLoader.save_to_yaml(yaml_file, targets)
    loaded = RosterLoader.load_from_yaml(yaml_file)

    assert len(loaded) == 2
    assert loaded[0].streamer_id == "asmon"
    assert loaded[0].display_name == "Asmongold"
    assert len(loaded[0].channel_urls) == 2
    assert loaded[0].download_priority == 10
    assert loaded[1].streamer_id == "kai"


def test_roster_json_roundtrip(temp_dir: Path):
    json_file = temp_dir / "roster.json"
    targets = [
        StreamerTargetRecord(
            streamer_id="shroud",
            display_name="Shroud",
            channel_urls=["https://twitch.tv/shroud"],
            primary_platform="TWITCH",
            quality_preset="1080p",
            download_priority=7,
        )
    ]

    RosterLoader.save(json_file, targets)
    loaded = RosterLoader.load(json_file)

    assert len(loaded) == 1
    assert loaded[0].streamer_id == "shroud"
    assert loaded[0].quality_preset == "1080p"


def test_roster_csv_roundtrip(temp_dir: Path):
    csv_file = temp_dir / "roster.csv"
    targets = [
        StreamerTargetRecord(
            streamer_id="pokimane",
            display_name="Pokimane",
            channel_urls=["https://twitch.tv/pokimane", "https://youtube.com/@pokimane"],
            primary_platform="TWITCH",
            download_priority=6,
            tags=["podcast", "variety"],
        )
    ]

    RosterLoader.save(csv_file, targets)
    loaded = RosterLoader.load(csv_file)

    assert len(loaded) == 1
    assert loaded[0].streamer_id == "pokimane"
    assert len(loaded[0].channel_urls) == 2
    assert loaded[0].tags == ["podcast", "variety"]
    assert loaded[0].download_priority == 6


# =========================================================================
# 2. HarvestCatalog Dual-Backend & CRUD Tests
# =========================================================================

def test_catalog_target_crud(memory_catalog: HarvestCatalog):
    target = StreamerTargetRecord(
        streamer_id="xqc",
        display_name="xQc",
        channel_urls=["https://kick.com/xqc"],
        primary_platform="KICK",
        download_priority=9,
        enabled=True,
    )
    memory_catalog.add_target(target)

    fetched = memory_catalog.get_target("xqc")
    assert fetched is not None
    assert fetched.display_name == "xQc"
    assert fetched.primary_platform == "KICK"
    assert fetched.download_priority == 9

    # Update target
    target.download_priority = 10
    memory_catalog.add_target(target)
    updated = memory_catalog.get_target("xqc")
    assert updated.download_priority == 10

    # List targets
    all_targets = memory_catalog.list_targets()
    assert len(all_targets) == 1

    # Delete target
    deleted = memory_catalog.delete_target("xqc")
    assert deleted is True
    assert memory_catalog.get_target("xqc") is None


def test_catalog_vod_crud_and_status_transitions(memory_catalog: HarvestCatalog):
    target = StreamerTargetRecord(
        streamer_id="tarik",
        display_name="Tarik",
        channel_urls=["https://twitch.tv/tarik"],
        primary_platform="TWITCH",
        download_priority=8,
    )
    memory_catalog.add_target(target)

    vod = HarvestedVodRecord(
        vod_id="twitch_v999001",
        streamer_id="tarik",
        platform="TWITCH",
        title="VALORANT Champions Watch Party",
        published_at=datetime.now(timezone.utc),
        duration_sec=3600.0,
        status=HarvestStatus.DISCOVERED,
    )
    memory_catalog.add_vod(vod)

    # Fetch VOD
    fetched = memory_catalog.get_vod("twitch_v999001")
    assert fetched is not None
    assert fetched.title == "VALORANT Champions Watch Party"
    assert fetched.status == HarvestStatus.DISCOVERED

    # Update to QUEUED
    memory_catalog.update_vod_status("twitch_v999001", HarvestStatus.QUEUED)
    queued = memory_catalog.get_queued_vods()
    assert len(queued) == 1
    assert queued[0].vod_id == "twitch_v999001"

    # Status counts
    counts = memory_catalog.count_vods_by_status()
    assert counts.get(HarvestStatus.QUEUED.value) == 1

    # Transition to DOWNLOADING then HARVESTED
    memory_catalog.update_vod_status(
        "twitch_v999001",
        HarvestStatus.HARVESTED,
        video_path="/path/to/media.mp4",
        file_size_bytes=1024000,
        download_speed_mbps=45.2,
    )
    harvested = memory_catalog.get_vod("twitch_v999001")
    assert harvested.status == HarvestStatus.HARVESTED
    assert harvested.video_path == "/path/to/media.mp4"
    assert harvested.file_size_bytes == 1024000
    assert harvested.download_speed_mbps == 45.2


# =========================================================================
# 3. ChannelVodCrawler Tests (Discovery & Deduplication)
# =========================================================================

def test_crawler_discovery_and_deduplication(memory_catalog: HarvestCatalog):
    target = StreamerTargetRecord(
        streamer_id="asmongold",
        display_name="Asmongold",
        channel_urls=["https://twitch.tv/zackrawrr"],
        primary_platform="TWITCH",
        lookback_days=7,
        max_recent_vods=3,
    )
    memory_catalog.add_target(target)

    now = datetime.now(timezone.utc)
    mock_metadata = {
        "_type": "playlist",
        "entries": [
            {
                "id": "v101",
                "title": "Reacting to Drama",
                "duration": 7200,
                "timestamp": (now - timedelta(days=1)).timestamp(),
                "webpage_url": "https://twitch.tv/videos/v101",
            },
            {
                "id": "v102",
                "title": "Old VOD Outside Lookback",
                "duration": 5400,
                "timestamp": (now - timedelta(days=20)).timestamp(),
                "webpage_url": "https://twitch.tv/videos/v102",
            },
        ],
    }

    def mock_extractor(url: str, max_vods: int) -> Dict[str, Any]:
        return mock_metadata

    crawler = ChannelVodCrawler(extractor_override=mock_extractor)

    # First crawl: v101 should be queued, v102 skipped due to lookback
    discovered = crawler.discover_target_vods(target, catalog=memory_catalog, auto_queue=True)
    assert len(discovered) == 1
    assert discovered[0].vod_id == "twitch_v101"
    assert discovered[0].status == HarvestStatus.QUEUED

    # Second crawl: v101 should be deduplicated (not queued twice)
    discovered_second = crawler.discover_target_vods(target, catalog=memory_catalog, auto_queue=True)
    assert len(discovered_second) == 0


# =========================================================================
# 4. HomelabHarvester Engine & Worker Pool Tests
# =========================================================================

def test_harvester_directory_layout_and_checksums(temp_dir: Path, memory_catalog: HarvestCatalog):
    target = StreamerTargetRecord(
        streamer_id="teststreamer",
        display_name="Test Streamer",
        primary_platform="TWITCH",
    )
    memory_catalog.add_target(target)

    pub_date = datetime(2026, 9, 29, 12, 0, 0, tzinfo=timezone.utc)
    vod = HarvestedVodRecord(
        vod_id="twitch_v55555",
        streamer_id="teststreamer",
        platform="TWITCH",
        title="Test Stream Ingest",
        published_at=pub_date,
        status=HarvestStatus.QUEUED,
    )
    memory_catalog.add_vod(vod)

    harvester = HomelabHarvester(
        catalog=memory_catalog,
        homelab_root=temp_dir,
        simulate=True,
    )

    harvested_record = harvester.harvest_vod("twitch_v55555")
    assert harvested_record.status == HarvestStatus.HARVESTED

    expected_dir = temp_dir / "vods" / "teststreamer" / "2026-09-29_twitch_v55555"
    assert expected_dir.exists()
    assert (expected_dir / "media.mp4").exists()
    assert (expected_dir / "chat.json").exists()
    assert (expected_dir / "metadata.json").exists()
    assert (expected_dir / "thumbnail.jpg").exists()
    assert (expected_dir / "checksums.sha256").exists()
    assert (expected_dir / "ingest_manifest.json").exists()

    # Verify checksums integrity
    assert verify_harvest_checksums(expected_dir) is True


def test_harvester_concurrency_and_per_streamer_limit(temp_dir: Path, memory_catalog: HarvestCatalog):
    t1 = StreamerTargetRecord(streamer_id="s1", display_name="Streamer 1", download_priority=10)
    t2 = StreamerTargetRecord(streamer_id="s2", display_name="Streamer 2", download_priority=5)
    memory_catalog.add_target(t1)
    memory_catalog.add_target(t2)

    # Add 2 VODs for s1, 1 VOD for s2
    memory_catalog.add_vod(HarvestedVodRecord(vod_id="v_s1_1", streamer_id="s1", platform="TWITCH", status=HarvestStatus.QUEUED))
    memory_catalog.add_vod(HarvestedVodRecord(vod_id="v_s1_2", streamer_id="s1", platform="TWITCH", status=HarvestStatus.QUEUED))
    memory_catalog.add_vod(HarvestedVodRecord(vod_id="v_s2_1", streamer_id="s2", platform="TWITCH", status=HarvestStatus.QUEUED))

    harvester = HomelabHarvester(
        catalog=memory_catalog,
        homelab_root=temp_dir,
        max_concurrent_workers=2,
        max_per_streamer=1,
        simulate=True,
    )

    # In first batch: with max_per_streamer=1 and max_workers=2,
    # it should pick 1 for s1 and 1 for s2 (not 2 for s1)
    processed = harvester.process_queue()
    assert len(processed) == 2
    processed_streamers = {v.streamer_id for v in processed}
    assert processed_streamers == {"s1", "s2"}


def test_harvester_disk_space_guard(temp_dir: Path, memory_catalog: HarvestCatalog):
    target = StreamerTargetRecord(streamer_id="disk_test", display_name="Disk Guard Streamer")
    memory_catalog.add_target(target)
    memory_catalog.add_vod(HarvestedVodRecord(vod_id="v_disk_1", streamer_id="disk_test", platform="TWITCH", status=HarvestStatus.QUEUED))

    # Setting min_free_disk_gb to unrealistic high value (1,000,000 GB)
    harvester = HomelabHarvester(
        catalog=memory_catalog,
        homelab_root=temp_dir,
        min_free_disk_gb=1_000_000.0,
        simulate=False,
    )

    with pytest.raises(RuntimeError, match="Insufficient disk space"):
        harvester.harvest_vod("v_disk_1")

    # Status should be ERROR
    record = memory_catalog.get_vod("v_disk_1")
    assert record.status == HarvestStatus.ERROR
    assert "Insufficient disk space" in record.error_message


# =========================================================================
# 5. Synergy Integrations Tests (Voiceprint, Chatter, Slang, Sponsor)
# =========================================================================

def test_synergy_voiceprint_seeding(temp_dir: Path):
    vp_path = temp_dir / "voiceprints.json"
    library = VoiceprintLibrary(storage_path=vp_path)

    dummy_emb = [0.1] * 192
    roster = [
        StreamerTargetRecord(
            streamer_id="hasanabi",
            display_name="HasanAbi",
            channel_urls=["https://twitch.tv/hasanabi"],
            voiceprint_embedding=dummy_emb,
        ),
        StreamerTargetRecord(
            streamer_id="no_vp_streamer",
            display_name="No Voiceprint",
            channel_urls=["https://twitch.tv/novp"],
            voiceprint_embedding=None,
        ),
    ]

    enrolled_count = seed_voiceprints(roster, library)
    assert enrolled_count == 1
    assert "hasanabi" in library.profiles
    assert library.profiles["hasanabi"].display_name == "HasanAbi"


def test_synergy_chatter_safety_and_slang(temp_dir: Path):
    chat_file = temp_dir / "chat.json"
    messages = [
        {
            "message_id": f"msg-{i}",
            "timestamp_offset": float(i),
            "user_id": f"chatter_{i % 3}",
            "author_name": f"Viewer_{i % 3}",
            "content": "POGGERS this game is crazy LUL" if i % 2 == 0 else "KEKW unbelievable",
            "emotes": [],
            "badges": [],
            "metadata": {},
        }
        for i in range(15)
    ]
    with open(chat_file, "w", encoding="utf-8") as f:
        json.dump(messages, f)

    # Chatter store integration
    store = ChatterProfileStore(db_path=":memory:")
    count = ingest_chatter_safety(chat_file, store, streamer_id="test_streamer")
    assert count == 15
    profile = store.get_chatter_profile("chatter_0")
    assert profile is not None
    assert profile.total_messages == 5

    # Adaptive slang integration
    lexicon_db = temp_dir / "adaptive_lexicon.json"
    engine = AdaptiveSlangEngine(lexicon_store=AdaptiveLexiconStore(db_path=lexicon_db))
    slang_count = feed_adaptive_slang(chat_file, engine, streamer_id="test_streamer")
    assert slang_count >= 0  # Processed without error


def test_synergy_sponsor_auditing():
    vod = HarvestedVodRecord(
        vod_id="v_sponsor_test",
        streamer_id="asmon",
        platform="TWITCH",
    )
    brand = BrandProfile(
        brand_id="starforge",
        brand_name="Starforge Systems",
        aliases=["Starforge"],
        product_keywords=["starforge", "pc build"],
    )
    audio = [
        AudioSegment(
            segment_id=0,
            start_sec=10.0,
            end_sec=25.0,
            speaker_label="STREAMER",
            transcript="Check out Starforge Systems for amazing PC builds with code ASMON",
        )
    ]
    chat = [
        ChatMessage(
            message_id="c1",
            timestamp_offset=12.0,
            user_id="u1",
            author_name="viewer1",
            content="Starforge PC looks sick!",
        )
    ]

    reports = audit_harvested_sponsors(vod, [brand], audio_segments=audio, chat_messages=chat)
    assert len(reports) == 1
    assert reports[0].brand_id == "starforge"



# =========================================================================
# 6. Pipeline Bridge Tests
# =========================================================================

def test_pipeline_bridge_handoff(temp_dir: Path, memory_catalog: HarvestCatalog):
    target = StreamerTargetRecord(streamer_id="bridge_streamer", display_name="Bridge Creator")
    memory_catalog.add_target(target)

    vod_dir = temp_dir / "vods" / "bridge_streamer" / "2026-09-29_bridge_v1"
    vod_dir.mkdir(parents=True, exist_ok=True)
    media_file = vod_dir / "media.mp4"
    chat_file = vod_dir / "chat.json"

    with open(media_file, "wb") as f:
        f.write(b"SIMULATED_VIDEO_DATA")

    with open(chat_file, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "message_id": "m1",
                    "timestamp_offset": 1.0,
                    "user_id": "u1",
                    "author_name": "Fan1",
                    "content": "Awesome stream!",
                }
            ],
            f,
        )

    # Compute checksum
    f_hash = compute_file_sha256(media_file)
    with open(vod_dir / "checksums.sha256", "w", encoding="utf-8") as f:
        f.write(f"{f_hash}  media.mp4\n")

    vod = HarvestedVodRecord(
        vod_id="bridge_v1",
        streamer_id="bridge_streamer",
        platform="TWITCH",
        video_path=str(media_file.resolve()),
        chat_path=str(chat_file.resolve()),
        status=HarvestStatus.HARVESTED,
    )
    memory_catalog.add_vod(vod)

    # Mock FullSpectrumPipeline so we don't need real ffmpeg/Whisper
    from unittest.mock import MagicMock, patch
    mock_analysis = MagicMock()
    mock_analysis.stream_id = "bridge_v1"
    mock_manifest = MagicMock()
    mock_manifest.total_fusion_slices = 10
    mock_manifest.shorts_produced_count = 1
    mock_manifest.model_dump.return_value = {"stream_id": "bridge_v1"}

    with patch("stream_fusion.harvester.bridge.FullSpectrumPipeline") as MockPipeline:
        instance = MockPipeline.return_value
        instance.run.return_value = (mock_analysis, mock_manifest)

        analysis, manifest = ingest_to_pipeline(
            vod_id="bridge_v1",
            catalog=memory_catalog,
            verify_checksums=True,
            run_shorts=True,
            dry_run_shorts=True,
        )

        assert analysis.stream_id == "bridge_v1"
        assert manifest.total_fusion_slices == 10
        # Catalog should now reflect ANALYZED status
        updated_vod = memory_catalog.get_vod("bridge_v1")
        assert updated_vod.status == HarvestStatus.ANALYZED
        assert updated_vod.analyzed_at is not None


# =========================================================================
# 7. CLI & JSON-RPC Integration Tests
# =========================================================================

def test_cli_roster_and_harvester(temp_dir: Path):
    db_path = temp_dir / "test_cli_catalog.db"

    # 1. Roster Add
    res = runner.invoke(
        app,
        [
            "harvest",
            "roster-add",
            "--id", "tarik",
            "--name", "Tarik",
            "--url", "https://twitch.tv/tarik",
            "--platform", "TWITCH",
            "--priority", "9",
            "--db", str(db_path),
        ],
    )
    assert res.exit_code == 0
    assert "Added streamer target" in res.stdout

    # 2. Roster List
    res_list = runner.invoke(app, ["harvest", "roster-list", "--db", str(db_path)])
    assert res_list.exit_code == 0
    assert "Tarik" in res_list.stdout

    # 3. Status
    res_status = runner.invoke(app, ["harvest", "status", "--db", str(db_path), "--homelab-root", str(temp_dir)])
    assert res_status.exit_code == 0
    assert "Homelab Harvester Status" in res_status.stdout

    # 4. Run harvester in simulation mode
    res_run = runner.invoke(
        app,
        [
            "harvest",
            "run",
            "--db", str(db_path),
            "--homelab-root", str(temp_dir),
            "--simulate",
        ],
    )
    assert res_run.exit_code == 0
    assert "Harvester finished" in res_run.stdout


def test_json_rpc_harvest_handlers(temp_dir: Path):
    db_path = temp_dir / "rpc_catalog.db"
    cat = HarvestCatalog(database_url=f"sqlite:///{db_path}")
    cat.add_target(
        StreamerTargetRecord(
            streamer_id="rpc_streamer",
            display_name="RPC Streamer",
            channel_urls=["https://twitch.tv/rpc"],
        )
    )
    cat.add_vod(
        HarvestedVodRecord(
            vod_id="rpc_vod_1",
            streamer_id="rpc_streamer",
            platform="TWITCH",
            status=HarvestStatus.QUEUED,
        )
    )
    cat.close()

    dispatcher = AgentRpcDispatcher()

    # 1. Get status
    status_req = {
        "jsonrpc": "2.0",
        "id": "1",
        "method": "streamfusion.harvest.getHarvesterStatus",
        "params": {
            "database_url": f"sqlite:///{db_path}",
            "homelab_root": str(temp_dir),
        },
    }
    resp = dispatcher.handle_request(status_req)
    assert "error" not in resp
    assert resp["result"]["queue_depth"] == 1

    # 2. List VODs
    list_req = {
        "jsonrpc": "2.0",
        "id": "2",
        "method": "streamfusion.harvest.listHarvestedVods",
        "params": {
            "database_url": f"sqlite:///{db_path}",
            "status": "QUEUED",
        },
    }
    list_resp = dispatcher.handle_request(list_req)
    assert "error" not in list_resp
    assert list_resp["result"]["count"] == 1
    assert list_resp["result"]["vods"][0]["vod_id"] == "rpc_vod_1"

    # 3. Start Harvester (simulate)
    start_req = {
        "jsonrpc": "2.0",
        "id": "3",
        "method": "streamfusion.harvest.startHarvester",
        "params": {
            "database_url": f"sqlite:///{db_path}",
            "homelab_root": str(temp_dir),
            "simulate": True,
        },
    }
    start_resp = dispatcher.handle_request(start_req)
    assert "error" not in start_resp
    assert start_resp["result"]["harvested_count"] == 1
