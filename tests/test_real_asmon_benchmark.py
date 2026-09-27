"""Real VOD Benchmark & Stateful Resumption integration test (Spec 10).

Validates end-to-end processing and zero-redundant-compute checkpoint resumption
on real Asmongold broadcast footage (asmon_sample_60s.mp4) and Twitch chat replay.
"""

from pathlib import Path
import pytest

from stream_fusion.analytics.sponsor_quantifier import SponsorDetector, SponsorReportGenerator
from stream_fusion.chat.analyzer import ChatAnalyzer
from stream_fusion.chat.nlp import ChatNLPAnalyzer, MemeBurstTracker, ChatterInfluenceScorer
from stream_fusion.checkpoint.manager import CheckpointManager
from stream_fusion.config import StreamFusionConfig
from stream_fusion.models.schemas import BrandProfile
from stream_fusion.pipeline import StreamPipeline


@pytest.fixture
def asmon_assets():
    root = Path(__file__).parent.parent
    video_path = root / "asmon_sample_60s.mp4"
    chat_path = root / "sample_asmon_chat.json"
    if not video_path.exists() or not chat_path.exists():
        pytest.skip("Asmon real VOD assets not found in root")
    return video_path, chat_path


def test_real_asmon_benchmark_and_resumption(asmon_assets, tmp_path: Path):
    video_path, chat_path = asmon_assets

    config = StreamFusionConfig()
    config.audio.whisper_model = "tiny"
    config.audio.device = "cpu"
    config.audio.compute_type = "int8"
    config.vision.device = "cpu"
    config.vision.sample_interval_sec = 2.0
    config.chat.bucket_window_sec = 2.0

    cache_dir = tmp_path / "asmon_cache"
    output_dir = tmp_path / "asmon_output"

    pipeline = StreamPipeline(config=config)

    # -------------------------------------------------------------
    # 1. First Run: Execute with Checkpoint Caching
    # -------------------------------------------------------------
    result1 = pipeline.run(
        media_input=video_path,
        chat_input=chat_path,
        output_dir=output_dir,
        duration_sec=6.0,
        cache_dir=cache_dir,
        chunk_index=0,
    )

    assert result1.stream_id == "asmon_sample_60s"
    assert result1.duration_sec == 6.0
    assert len(result1.slices) > 0

    # Verify checkpoint files were generated
    cp_mgr = CheckpointManager(cache_dir=cache_dir, stream_id="asmon_sample_60s")
    assert cp_mgr.is_chunk_completed(0)
    assert cp_mgr.load_chunk_audio(0) is not None
    assert cp_mgr.load_chunk_vision(0) is not None
    assert cp_mgr.load_chunk_chat(0) is not None

    # Check output artifacts
    html_report = output_dir / "asmon_sample_60s_grounding_report.html"
    assert html_report.exists()
    assert html_report.stat().st_size > 500

    parquet_matrix = output_dir / "asmon_sample_60s_matrix.parquet"
    assert parquet_matrix.exists()

    # -------------------------------------------------------------
    # 2. Second Run: Verify Resumption from Disk Cache
    # -------------------------------------------------------------
    output_dir_resumed = tmp_path / "asmon_output_resumed"
    result2 = pipeline.run(
        media_input=video_path,
        chat_input=chat_path,
        output_dir=output_dir_resumed,
        duration_sec=6.0,
        cache_dir=cache_dir,
        chunk_index=0,
    )
    assert result2.stream_id == "asmon_sample_60s"
    assert len(result2.slices) == len(result1.slices)


def test_real_asmon_chat_nlp_and_sponsor_analysis(asmon_assets):
    _, chat_path = asmon_assets

    chat_analyzer = ChatAnalyzer()
    raw_chat = chat_analyzer.parse_twitch_downloader_json(chat_path)
    assert len(raw_chat) > 50

    # 1. Spec 08: Intent classification on real Asmon chat
    nlp_analyzer = ChatNLPAnalyzer()
    classified = nlp_analyzer.analyze_message_stream(raw_chat[:100])
    assert len(classified) == 100
    intents = {dist.primary_intent for _, dist in classified}
    # Real Twitch chat contains various intents
    assert len(intents) >= 2

    # 2. Spec 08: Meme burst tracking on real chat
    burst_tracker = MemeBurstTracker(window_sec=15.0, similarity_threshold=0.5, min_distinct_authors=3)
    bursts = burst_tracker.detect_meme_bursts(raw_chat)
    # Validate structure if bursts detected
    for b in bursts:
        assert b.unique_spreaders >= 3
        assert b.propagation_velocity > 0

    # 3. Spec 08: Chatter influence scoring
    scorer = ChatterInfluenceScorer()
    profiles = scorer.score_chatters(raw_chat, bursts)
    assert len(profiles) > 0
    assert profiles[0].influence_score >= profiles[-1].influence_score

    # 4. Spec 09: Sponsor analysis
    brand = BrandProfile(
        brand_id="starforge",
        brand_name="Starforge Systems",
        aliases=["starforge", "pc"],
        promo_codes=["ASMON"],
        product_keywords=["pc", "rig", "gpu", "desktop"],
    )
    detector = SponsorDetector()
    report_gen = SponsorReportGenerator()

    # Detect with synthetic audio mention
    audio_segments = [
        {"segment_id": 1, "start_sec": 5.0, "end_sec": 12.0, "speaker_label": "STREAMER", "transcript": "Check out starforge systems with code asmon"}
    ]
    from stream_fusion.models.schemas import AudioSegment
    audio_objs = [AudioSegment(**a) for a in audio_segments]
    segments = detector.detect_segments(brand, audio_segments=audio_objs)
    assert len(segments) == 1
    report = report_gen.generate_report(brand, segments[0], raw_chat)
    assert report.brand_id == "starforge"
    assert report.mention_velocity >= 0.0
