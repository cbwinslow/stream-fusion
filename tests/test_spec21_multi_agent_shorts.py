"""Comprehensive Test Suite for Spec 21: Autonomous Multi-Agent Short Production & Auto-Publisher."""

import json
from pathlib import Path
import tempfile
import pytest
from typer.testing import CliRunner

from stream_fusion.models.schemas import (
    AuditStatus,
    AudioSegment,
    ChatMessage,
    ContentAuditReport,
    CropLayout,
    EditorialCutPlan,
    FusionSlice,
    NarrativeArc,
    PlatformCopyBundle,
    ShortCandidate,
    ShortProductionPackage,
    SponsorSegment,
    StreamAnalysisResult,
    StreamerClaim,
    SubtitleStylePreset,
    ViralityScoreCard,
    VisualKeyframe,
    WordTiming,
)
from stream_fusion.production.checker import PolicyAgent
from stream_fusion.production.copywriter import CopywriterAgent
from stream_fusion.production.director import DirectorAgent
from stream_fusion.production.editor import EditorAgent
from stream_fusion.production.orchestrator import ShortProductionOrchestrator
from stream_fusion.production.publisher import (
    PublishDispatcher,
    PublisherAgent,
    TikTokPublisher,
    TwitterPublisher,
    YouTubePublisher,
)
from stream_fusion.schema.agent_rpc import AgentRpcDispatcher
from stream_fusion.schema.adapters import SqliteJsonStore
from stream_fusion.schema.envelope import StreamEventType, StreamFusionEnvelope
from stream_fusion.schema.registry import SchemaRegistry
from stream_fusion.cli import app


@pytest.fixture
def mock_stream_data():
    """Builds synthetic multi-modal stream data with highlights, sponsors, claims, and chat."""
    audio_segments = [
        AudioSegment(
            segment_id=0,
            start_sec=10.0,
            end_sec=18.0,
            speaker_label="STREAMER",
            transcript="I cannot believe this game right now this is totally wild!",
            words=[
                WordTiming(word="I", start=10.0, end=10.5),
                WordTiming(word="cannot", start=10.6, end=11.2),
                WordTiming(word="believe", start=11.3, end=12.0),
                WordTiming(word="this", start=12.1, end=12.5),
                WordTiming(word="game", start=12.6, end=13.0),
            ],
        ),
        AudioSegment(
            segment_id=1,
            start_sec=18.5,
            end_sec=32.0,
            speaker_label="STREAMER",
            transcript="No way! He actually hit that 360 no scope across the entire map!",
            words=[
                WordTiming(word="No", start=18.5, end=19.0),
                WordTiming(word="way!", start=19.1, end=19.8),
            ],
        ),
        AudioSegment(
            segment_id=2,
            start_sec=32.5,
            end_sec=45.0,
            speaker_label="STREAMER",
            transcript="Check out our sponsor Apex Energy Drink, use code CLUTCH for 20% off!",
            words=[],
        ),
    ]

    chat_messages = [
        ChatMessage(
            message_id=f"m_{i}",
            timestamp_offset=20.0 + (i * 0.2),
            user_id=f"u_{i}",
            author_name=f"chatter_{i}",
            content="OMEGALUL insane shot poggers absolute clutch!" if i % 2 == 0 else "KEKW no way bro!",
        )
        for i in range(25)
    ]
    # Add baseline chat
    chat_messages.extend([
        ChatMessage(
            message_id=f"m_base_{i}",
            timestamp_offset=float(i * 3.0),
            user_id=f"u_b_{i}",
            author_name=f"base_{i}",
            content="just chilling hello",
        )
        for i in range(30)
    ])

    keyframes = [
        VisualKeyframe(
            frame_index=1,
            timestamp_sec=22.0,
            scene_type="GAMEPLAY",
            screen_summary="First-person shooter clutch shot",
            ocr_text_blocks=["VICTORY ROYAL"],
        ),
        VisualKeyframe(
            frame_index=2,
            timestamp_sec=25.0,
            scene_type="REACT_VIDEO",
            screen_summary="Facecam laughing streamer",
        ),
    ]

    fusion_slices = [
        FusionSlice(
            bucket_index=0,
            start_sec=20.0,
            end_sec=24.0,
            visual_description="Clutch headshot",
            streamer_transcript="No way! He actually hit that!",
            agreement_score=0.92,
            is_spike_moment=True,
            chat_velocity_per_sec=5.0,
        ),
        FusionSlice(
            bucket_index=1,
            start_sec=36.0,
            end_sec=40.0,
            visual_description="Sponsor can on table",
            streamer_transcript="Check out Apex Energy Drink",
            agreement_score=0.85,
            chat_velocity_per_sec=1.0,
        ),
    ]

    claims = [
        StreamerClaim(
            claim_id="c_001",
            creator_id="streamer_test",
            vod_id="stream_clutch_test",
            timestamp_sec=20.0,
            end_sec=26.0,
            subject_entity="Player",
            predicate="HIT",
            object_value="360 no scope across the map",
            stance="APPROVAL",
            raw_quote="He actually hit that 360 no scope across the entire map",
            topic_category="GAMING",
        )
    ]

    sponsors = [
        SponsorSegment(
            segment_id=1,
            brand_id="Apex Energy Drink",
            start_sec=32.0,
            end_sec=50.0,
            matched_audio_transcripts=["Apex Energy Drink"],
        )
    ]

    analysis = StreamAnalysisResult(
        stream_id="stream_clutch_test",
        duration_sec=90.0,
        total_chat_messages=len(chat_messages),
        slices=fusion_slices,
        highlights=[
            {"timestamp_sec": 22.0, "score": 0.95},
            {"timestamp_sec": 38.0, "score": 0.60},
        ],
    )

    return {
        "analysis": analysis,
        "audio_segments": audio_segments,
        "chat_messages": chat_messages,
        "keyframes": keyframes,
        "fusion_slices": fusion_slices,
        "claims": claims,
        "sponsors": sponsors,
    }


# ---------------------------------------------------------------------------
# Test 1: Director Agent
# ---------------------------------------------------------------------------
def test_director_candidate_curation(mock_stream_data):
    director = DirectorAgent(min_duration_sec=15.0, max_duration_sec=45.0)
    candidates = director.select_candidates(
        analysis=mock_stream_data["analysis"],
        fusion_slices=mock_stream_data["fusion_slices"],
        audio_segments=mock_stream_data["audio_segments"],
        chat_messages=mock_stream_data["chat_messages"],
        top_k=2,
        min_highlight_score=0.5,
    )

    assert len(candidates) >= 1
    top_cand = candidates[0]
    assert 15.0 <= top_cand.duration_sec <= 45.0
    assert top_cand.highlight_score >= 0.5
    assert top_cand.chat_burst_zscore >= 1.0
    assert top_cand.hook_text != ""
    assert isinstance(top_cand.narrative_arc, NarrativeArc)


# ---------------------------------------------------------------------------
# Test 2: Editor Agent
# ---------------------------------------------------------------------------
def test_editor_cut_plan_generation(mock_stream_data):
    editor = EditorAgent()
    cand = ShortCandidate(
        start_sec=16.0,
        end_sec=36.0,
        duration_sec=20.0,
        peak_timestamp_sec=22.0,
        highlight_score=0.95,
        chat_burst_zscore=4.2,
        primary_emotion="HYSTERICAL_LAUGHTER",
        narrative_arc=NarrativeArc.INSTANT_CLIMAX_REACTION,
    )

    plan = editor.create_cut_plan(candidate=cand, keyframes=mock_stream_data["keyframes"])
    assert plan.crop_layout == CropLayout.STACKED_CAM_CONTENT
    assert plan.facecam_box is not None
    assert plan.content_box is not None
    assert plan.subtitle_preset == SubtitleStylePreset.KARAOKE_POP
    assert plan.highlight_word_color == "&H0000FFFF"  # Yellow for laughter
    assert plan.audio_duck_music_db == -14.0


# ---------------------------------------------------------------------------
# Test 3: Policy & Fact-Checking Agent
# ---------------------------------------------------------------------------
def test_policy_agent_safety_and_ftc_sponsor(mock_stream_data):
    policy = PolicyAgent()

    # Candidate 1: Non-sponsored clutch moment
    cand_clutch = ShortCandidate(
        start_sec=18.0,
        end_sec=30.0,
        duration_sec=12.0,
        peak_timestamp_sec=22.0,
        highlight_score=0.95,
        chat_burst_zscore=4.0,
    )
    report_clutch = policy.audit_candidate(
        candidate=cand_clutch,
        audio_segments=mock_stream_data["audio_segments"],
        claims=mock_stream_data["claims"],
        sponsor_segments=mock_stream_data["sponsors"],
    )
    assert report_clutch.audit_status == AuditStatus.PASSED
    assert report_clutch.has_sponsored_content is False
    assert report_clutch.ftc_disclosure_required is False
    assert report_clutch.claim_verified is True

    # Candidate 2: Sponsored highlight moment (Apex Energy Drink)
    cand_sponsor = ShortCandidate(
        start_sec=32.0,
        end_sec=44.0,
        duration_sec=12.0,
        peak_timestamp_sec=38.0,
        highlight_score=0.60,
        chat_burst_zscore=1.0,
    )
    report_sponsor = policy.audit_candidate(
        candidate=cand_sponsor,
        audio_segments=mock_stream_data["audio_segments"],
        claims=mock_stream_data["claims"],
        sponsor_segments=mock_stream_data["sponsors"],
    )
    assert report_sponsor.has_sponsored_content is True
    assert report_sponsor.ftc_disclosure_required is True
    assert "Apex Energy Drink" in report_sponsor.sponsor_brand_names
    assert report_sponsor.disclosure_tag == "#ad #sponsored"


# ---------------------------------------------------------------------------
# Test 4: Copywriter Agent
# ---------------------------------------------------------------------------
def test_copywriter_copy_and_virality():
    copywriter = CopywriterAgent()
    cand = ShortCandidate(
        start_sec=18.0,
        end_sec=42.0,
        duration_sec=24.0,
        peak_timestamp_sec=22.0,
        highlight_score=0.95,
        chat_burst_zscore=4.5,
        primary_emotion="HYSTERICAL_LAUGHTER",
        narrative_arc=NarrativeArc.INSTANT_CLIMAX_REACTION,
        hook_text="No way! He actually hit that 360 no scope!",
        summary="Streamer reacts to impossible cross-map snipe shot",
        dominant_slang=["poggers", "clutch", "kekw"],
    )
    audit = ContentAuditReport(audit_status=AuditStatus.PASSED, ftc_disclosure_required=False)

    copy_bundle, virality = copywriter.generate_copy(candidate=cand, audit=audit, creator_handle="Shroud")

    # YouTube assertions
    assert len(copy_bundle.youtube_title) <= 100
    assert "#Shorts" in copy_bundle.youtube_title
    assert "Shroud" in copy_bundle.youtube_description

    # TikTok assertions
    assert len(copy_bundle.tiktok_caption) <= 2200
    assert "#fyp" in copy_bundle.tiktok_caption
    assert any("poggers" in tag.lower() for tag in copy_bundle.tiktok_hashtags)

    # Twitter thread assertions
    assert len(copy_bundle.twitter_thread) == 3
    for tweet in copy_bundle.twitter_thread:
        assert len(tweet) <= 280

    # Virality scorecard assertions
    assert 0.0 <= virality.overall_virality_score <= 100.0
    assert virality.overall_virality_score >= 70.0  # High score due to high burst + emotion
    assert virality.predicted_completion_rate > 60.0


# ---------------------------------------------------------------------------
# Test 5: Publisher Agent & Platform Dispatcher
# ---------------------------------------------------------------------------
def test_publisher_packaging_and_dispatch():
    publisher = PublisherAgent()
    cand = ShortCandidate(
        start_sec=10.0,
        end_sec=30.0,
        duration_sec=20.0,
        peak_timestamp_sec=20.0,
        highlight_score=0.9,
        chat_burst_zscore=3.5,
    )
    plan = EditorialCutPlan()
    audit = ContentAuditReport()
    copy = PlatformCopyBundle(
        youtube_title="Insane Moment! #Shorts",
        youtube_description="Check it out",
        tiktok_caption="No way #viral",
        hook_headline="Wait till the end",
    )
    vir = ViralityScoreCard(
        overall_virality_score=88.5,
        hook_strength=85.0,
        pacing_score=90.0,
        chat_resonance=92.0,
        meme_potential=80.0,
        predicted_completion_rate=84.0,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        pkg, envelope = publisher.package(
            candidate=cand,
            plan=plan,
            audit=audit,
            copy=copy,
            virality=vir,
            output_dir=out_dir,
        )

        assert pkg.candidate.candidate_id == cand.candidate_id
        assert envelope.event_type == StreamEventType.SHORT_PRODUCED
        assert (out_dir / f"package_{cand.candidate_id}.json").exists()
        assert (out_dir / f"envelope_{cand.candidate_id}.json").exists()

        # Test Platform Dispatcher
        dispatcher = PublishDispatcher()
        pub_results = dispatcher.publish(package=pkg, platforms=["youtube", "tiktok", "twitter"], dry_run=True)
        assert len(pub_results) == 3
        platforms = [r.platform for r in pub_results]
        assert "youtube" in platforms
        assert "tiktok" in platforms
        assert "twitter" in platforms
        for r in pub_results:
            assert r.status == "MOCK_PUBLISHED"
            assert r.post_url is not None


# ---------------------------------------------------------------------------
# Test 6: Short Production Orchestrator
# ---------------------------------------------------------------------------
def test_orchestrator_end_to_end(mock_stream_data):
    orchestrator = ShortProductionOrchestrator()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        results = orchestrator.produce_shorts(
            analysis=mock_stream_data["analysis"],
            output_dir=out_dir,
            fusion_slices=mock_stream_data["fusion_slices"],
            keyframes=mock_stream_data["keyframes"],
            audio_segments=mock_stream_data["audio_segments"],
            chat_messages=mock_stream_data["chat_messages"],
            claims=mock_stream_data["claims"],
            sponsor_segments=mock_stream_data["sponsors"],
            top_k=2,
            dry_run=True,
        )

        assert len(results) >= 1
        pkg, env = results[0]
        assert isinstance(pkg, ShortProductionPackage)
        assert env.event_type == StreamEventType.SHORT_PRODUCED
        assert (out_dir / f"package_{pkg.candidate.candidate_id}.json").exists()


# ---------------------------------------------------------------------------
# Test 7: Schema Registry & JSON-RPC 2.0 Dispatcher
# ---------------------------------------------------------------------------
def test_schema_registry_and_agent_rpc(mock_stream_data):
    # Verify Schema Registry includes Spec 21 models
    registry = SchemaRegistry()
    registered_names = [s["schema_name"] for s in registry.list_schemas()]
    assert "ShortCandidate" in registered_names
    assert "EditorialCutPlan" in registered_names
    assert "ContentAuditReport" in registered_names
    assert "PlatformCopyBundle" in registered_names
    assert "ViralityScoreCard" in registered_names
    assert "ShortProductionPackage" in registered_names
    assert "PublishResult" in registered_names

    # Verify JSON Schema generation works
    json_schema = registry.export_json_schema("ShortProductionPackage")
    assert "properties" in json_schema
    assert "candidate" in json_schema["properties"]
    assert "virality" in json_schema["properties"]

    # Verify Agent RPC Dispatcher methods
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_events.db"
        store = SqliteJsonStore(db_path=db_path)
        dispatcher = AgentRpcDispatcher(store=store, registry=registry)

        # 1. Produce Shorts via RPC
        produce_req = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "streamfusion.produceShorts",
            "params": {
                "analysis": mock_stream_data["analysis"].model_dump(mode="json"),
                "output_dir": tmpdir,
                "top_k": 1,
                "dry_run": True,
            },
        })
        resp1 = json.loads(dispatcher.handle_line(produce_req))
        assert "result" in resp1
        shorts = resp1["result"]
        assert len(shorts) == 1
        pkg_id = shorts[0]["package_id"]

        # 2. List Shorts via RPC
        list_req = json.dumps({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "streamfusion.listShorts",
            "params": {"shorts_dir": tmpdir},
        })
        resp2 = json.loads(dispatcher.handle_line(list_req))
        assert "result" in resp2
        assert len(resp2["result"]) >= 1

        # 3. Publish Short via RPC
        publish_req = json.dumps({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "streamfusion.publishShort",
            "params": {
                "package": shorts[0],
                "platforms": ["youtube", "tiktok"],
                "dry_run": True,
            },
        })
        resp3 = json.loads(dispatcher.handle_line(publish_req))
        assert "result" in resp3
        pub_results = resp3["result"]
        assert len(pub_results) == 2
        assert pub_results[0]["status"] == "MOCK_PUBLISHED"


# ---------------------------------------------------------------------------
# Test 8: CLI Subcommands (generate, publish, inspect, list)
# ---------------------------------------------------------------------------
def test_cli_shorts_commands(mock_stream_data):
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        analysis_file = tmp_path / "fusion_analysis.json"
        analysis_file.write_text(
            mock_stream_data["analysis"].model_dump_json(indent=2),
            encoding="utf-8",
        )
        shorts_out = tmp_path / "shorts_out"

        # 1. Generate shorts
        res_gen = runner.invoke(
            app,
            [
                "shorts",
                "generate",
                "-a", str(analysis_file),
                "-o", str(shorts_out),
                "-k", "1",
                "--dry-run",
            ],
        )
        assert res_gen.exit_code == 0
        assert "Autonomous Short Production Results" in res_gen.stdout

        pkg_files = list(shorts_out.glob("package_*.json"))
        assert len(pkg_files) == 1
        pkg_file = pkg_files[0]

        # 2. Inspect short
        res_inspect = runner.invoke(
            app,
            ["shorts", "inspect", str(pkg_file)],
        )
        assert res_inspect.exit_code == 0
        assert "Short Package Inspection" in res_inspect.stdout
        assert "Virality Scorecard" in res_inspect.stdout

        # 3. List shorts
        res_list = runner.invoke(
            app,
            ["shorts", "list", "-o", str(shorts_out)],
        )
        assert res_list.exit_code == 0
        assert "Staged Vertical Short Packages" in res_list.stdout

        # 4. Publish short
        res_pub = runner.invoke(
            app,
            ["shorts", "publish", str(pkg_file), "-p", "youtube", "--dry-run"],
        )
        assert res_pub.exit_code == 0
        assert "Publication Dispatch" in res_pub.stdout
        assert "MOCK_PUBLISHED" in res_pub.stdout
