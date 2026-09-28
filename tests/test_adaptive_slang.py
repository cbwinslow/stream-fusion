"""Unit and Integration Tests for Spec 17: Self-Expanding Adaptive Slang & Meme Engine."""

from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import pytest
from typer.testing import CliRunner

from stream_fusion.models.schemas import ChatMessage, AudioSegment
from stream_fusion.chat.nlp import ChatNLPAnalyzer
from stream_fusion.nlp.adaptive_slang import (
    SlangCandidate,
    AdaptiveTermEntry,
    RollingBurstDetector,
    ContextualAutoTagger,
    NovelClusterDetector,
    AdaptiveLexiconStore,
    AdaptiveSlangEngine,
)
from stream_fusion.schema.agent_rpc import AgentRpcDispatcher
from stream_fusion.cli import app


@pytest.fixture
def runner():
    return CliRunner()


def test_token_normalization_and_burst_detection():
    """Verify normalization of repeated characters and statistical burst Z-score evaluation."""
    detector = RollingBurstDetector(window_sec=5.0, z_threshold=2.5, min_occurrences=4, min_distinct_authors=3)

    tokens = detector.extract_tokens("catJAAAAMMMMM! That is sooooo INSANEEEE!!! <3 123")
    assert "catjam" in tokens
    assert "insane" in tokens
    assert "<3" in tokens
    assert "123" not in tokens

    # Generate synthetic message stream: background chatter + sudden burst of 'catjam'
    messages = []
    # Background messages
    for i in range(10):
        messages.append(
            ChatMessage(
                message_id=f"bg_{i}",
                timestamp_offset=float(i),
                user_id=f"user_{i}",
                author_name=f"chatter_{i}",
                content="hello streamer nice gameplay",
            )
        )

    # Burst of 'catjam' at t = 12.0s to 14.0s
    for j in range(6):
        messages.append(
            ChatMessage(
                message_id=f"burst_{j}",
                timestamp_offset=12.0 + j * 0.3,
                user_id=f"burst_user_{j}",
                author_name=f"hype_author_{j}",
                content="catJAM catJAM holy PogChamp",
            )
        )

    candidates = detector.detect_bursts(messages)
    assert len(candidates) >= 1
    terms = [c.term for c in candidates]
    assert "catjam" in terms

    catjam_cand = next(c for c in candidates if c.term == "catjam")
    assert catjam_cand.z_score >= 2.5
    assert catjam_cand.unique_authors >= 3
    assert catjam_cand.burst_velocity > 0


def test_contextual_auto_tagging():
    """Verify that co-occurring chat emotion and streamer audio prosody auto-tags the candidate."""
    tagger = ContextualAutoTagger()

    candidate = SlangCandidate(
        term="glazing",
        burst_velocity=2.5,
        z_score=4.1,
        total_occurrences=5,
        unique_authors=4,
        window_start_sec=10.0,
        window_end_sec=15.0,
    )

    messages = [
        ChatMessage(
            message_id="m1",
            timestamp_offset=11.0,
            user_id="u1",
            author_name="a1",
            content="bro is glazing him so hard yikes cringe",
        ),
        ChatMessage(
            message_id="m2",
            timestamp_offset=12.0,
            user_id="u2",
            author_name="a2",
            content="the glazing is disgusting ew",
        ),
    ]

    audio_segments = [
        AudioSegment(
            segment_id=1,
            start_sec=10.5,
            end_sec=14.0,
            speaker_label="STREAMER",
            transcript="Stop glazing! haha",
            confidence=0.9,
        )
    ]

    tagged = tagger.tag_candidate(candidate, messages, audio_segments=audio_segments)
    assert tagged.inferred_intent == "DISGUST_CRINGE"
    assert tagged.inferred_valence < 0.0
    assert tagged.acoustic_energy_boost >= 1.5
    assert tagged.confidence > 0.65


def test_novel_cluster_detector():
    """Verify novel cluster grouping using character n-gram centroids."""
    detector = NovelClusterDetector(n_gram_size=3, similarity_threshold=0.30)

    candidates = [
        SlangCandidate(
            term="backseating",
            burst_velocity=1.5,
            z_score=3.2,
            total_occurrences=4,
            unique_authors=3,
            window_start_sec=5.0,
            window_end_sec=10.0,
            inferred_intent="NEUTRAL",
            confidence=0.5,
        ),
        SlangCandidate(
            term="backseat",
            burst_velocity=1.8,
            z_score=3.5,
            total_occurrences=5,
            unique_authors=4,
            window_start_sec=5.0,
            window_end_sec=10.0,
            inferred_intent="NEUTRAL",
            confidence=0.5,
        ),
        SlangCandidate(
            term="unrelated_term_xyz",
            burst_velocity=1.0,
            z_score=3.0,
            total_occurrences=4,
            unique_authors=3,
            window_start_sec=5.0,
            window_end_sec=10.0,
            inferred_intent="NEUTRAL",
            confidence=0.5,
        ),
    ]

    clusters = detector.discover_clusters(candidates)
    assert len(clusters) >= 2
    # 'backseating' and 'backseat' should cluster together
    backseat_cluster = next((c for c in clusters if "backseat" in c.centroid_terms), None)
    assert backseat_cluster is not None
    assert "backseating" in backseat_cluster.centroid_terms


def test_adaptive_lexicon_store_and_decay(tmp_path: Path):
    """Verify persistence, promotion, temporal confidence decay, and pruning."""
    lex_file = tmp_path / "test_lexicon.json"
    store = AdaptiveLexiconStore(db_path=lex_file)

    now = datetime(2026, 9, 28, 12, 0, 0, tzinfo=timezone.utc)
    now_iso = now.isoformat()

    candidate = SlangCandidate(
        term="pepepoint",
        burst_velocity=3.0,
        z_score=5.0,
        total_occurrences=20,
        unique_authors=10,
        window_start_sec=0.0,
        window_end_sec=10.0,
        inferred_intent="AMUSEMENT",
        inferred_valence=0.8,
        confidence=0.88,
    )

    entry = store.add_or_update(candidate, timestamp_iso=now_iso)
    assert entry.status == "PROMOTED"
    assert entry.term == "pepepoint"
    store.save()
    assert lex_file.exists()

    # Active lexicon export
    active_map = store.get_active_lexicon()
    assert "AMUSEMENT" in active_map
    assert "pepepoint" in active_map["AMUSEMENT"]

    # Integrate directly with ChatNLPAnalyzer
    analyzer = ChatNLPAnalyzer(custom_lexicon=active_map)
    msg = ChatMessage(
        message_id="msg_test",
        timestamp_offset=1.0,
        user_id="u1",
        author_name="chatter",
        content="look at him pepepoint",
    )
    dist = analyzer.classify_message(msg)
    assert dist.primary_intent == "AMUSEMENT"
    assert dist.valence > 0.5

    # Simulate temporal decay after 28 days (2 half-lives: 0.88 * 0.5 * 0.5 = 0.22)
    future_time = now + timedelta(days=28)
    decayed_terms = store.apply_temporal_decay(as_of=future_time)
    assert "pepepoint" in decayed_terms
    assert store.entries["pepepoint"].status == "DECAYED"

    # Simulate temporal decay after 45 days (< 0.20 threshold)
    far_future_time = now + timedelta(days=45)
    cur_conf = store.entries["pepepoint"].compute_decayed_confidence(as_of=far_future_time)
    assert cur_conf < 0.20

    # Prune
    # Mocking datetime.now for prune_decayed by temporarily overriding last_seen
    store.entries["pepepoint"].last_seen_timestamp = (
        now - timedelta(days=50)
    ).isoformat()
    pruned_count = store.prune_decayed(threshold=0.20)
    assert pruned_count == 1
    assert "pepepoint" not in store.entries


def test_agent_rpc_query_slang(tmp_path: Path):
    """Test JSON-RPC 2.0 streamfusion.querySlang method."""
    lex_file = tmp_path / "rpc_lexicon.json"
    store = AdaptiveLexiconStore(db_path=lex_file)

    store.add_or_update(
        SlangCandidate(
            term="bedge",
            burst_velocity=2.0,
            z_score=4.0,
            total_occurrences=10,
            unique_authors=5,
            window_start_sec=1.0,
            window_end_sec=5.0,
            inferred_intent="NEUTRAL",
            inferred_valence=-0.1,
            confidence=0.75,
        )
    )
    store.save()

    dispatcher = AgentRpcDispatcher()
    resp = dispatcher.handle_request({
        "jsonrpc": "2.0",
        "method": "streamfusion.querySlang",
        "params": {"lexicon_path": str(lex_file)},
        "id": 42,
    })

    assert "result" in resp
    results = resp["result"]
    assert len(results) == 1
    assert results[0]["term"] == "bedge"


def test_slang_cli(runner, tmp_path: Path):
    """Test CLI commands: slang scan, slang list, slang prune."""
    chat_file = tmp_path / "sample_chat.json"
    lex_file = tmp_path / "cli_lexicon.json"

    # Create synthetic chat JSON
    msgs = []
    for i in range(8):
        msgs.append({
            "id": f"m_{i}",
            "created_at": i,
            "commenter": {"_id": f"u_{i}", "display_name": f"user_{i}"},
            "message": {"body": "gigabased gigabased PogChamp LULW"},
        })
    chat_file.write_text(json.dumps({"comments": msgs}), encoding="utf-8")

    # 1. slang scan
    res_scan = runner.invoke(
        app,
        ["slang", "scan", "--chat", str(chat_file), "--lexicon", str(lex_file), "--z-score", "1.5"],
    )
    assert res_scan.exit_code == 0
    assert "Discovered Emerging Slang Candidates" in res_scan.stdout

    # 2. slang list
    res_list = runner.invoke(app, ["slang", "list", "--lexicon", str(lex_file)])
    assert res_list.exit_code == 0
    assert "Adaptive Slang Lexicon" in res_list.stdout

    # 3. slang prune
    res_prune = runner.invoke(app, ["slang", "prune", "--lexicon", str(lex_file)])
    assert res_prune.exit_code == 0
    assert "Pruned" in res_prune.stdout
