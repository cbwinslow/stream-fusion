"""Unit and Integration Tests for Spec 28: Semantic Vector Search & Multimodal RAG Engine."""

import json
from pathlib import Path
import numpy as np
import pytest
from starlette.testclient import TestClient
from typer.testing import CliRunner

from stream_fusion.cli import app
from stream_fusion.harvester.catalog import HarvestCatalog
from stream_fusion.knowledge.search import (
    BM25Index,
    HybridSearchEngine,
    LocalEmbedder,
    LocalVectorStorage,
    MultimodalRagSynthesizer,
    format_timestamp,
)
from stream_fusion.models.schemas import (
    DashboardConfig,
    HarvestedVodRecord,
    IndexVodRequest,
    RagSynthesisRequest,
    SearchQueryRequest,
)
from stream_fusion.web.app import create_app


# ---------------------------------------------------------------------------
# 1. LocalEmbedder Tests
# ---------------------------------------------------------------------------

def test_local_embedder_deterministic_and_similarity():
    """Verify local embedder produces deterministic unit vectors and semantic clustering."""
    embedder = LocalEmbedder(dim=128)

    text_a = "Deadlock matchmaking system is completely broken and unbalanced"
    text_b = "Deadlock ranked MMR match system is unplayable"
    text_c = "Delicious Italian pasta recipe with creamy garlic parmesan sauce"

    vec_a1 = embedder.embed_text(text_a)
    vec_a2 = embedder.embed_text(text_a)
    vec_b = embedder.embed_text(text_b)
    vec_c = embedder.embed_text(text_c)

    # Dimensionality & Normalization
    assert vec_a1.shape == (128,)
    assert np.isclose(np.linalg.norm(vec_a1), 1.0, atol=1e-4)

    # Determinism
    assert np.allclose(vec_a1, vec_a2)

    # Semantic similarity: text_a and text_b share gamer/game semantics
    sim_ab = LocalEmbedder.cosine_similarity(vec_a1, vec_b)
    sim_ac = LocalEmbedder.cosine_similarity(vec_a1, vec_c)
    assert sim_ab > sim_ac, f"Expected {sim_ab} > {sim_ac}"

    # Batch embedding
    batch = embedder.embed_batch([text_a, text_b, text_c])
    assert batch.shape == (3, 128)
    sims = LocalEmbedder.batch_cosine_similarity(vec_a1, batch)
    assert np.isclose(sims[0], 1.0, atol=1e-4)
    assert sims[1] > sims[2]


# ---------------------------------------------------------------------------
# 2. BM25Index Tests
# ---------------------------------------------------------------------------

def test_bm25_index_slang_and_scoring():
    """Verify BM25 sparse index preserves slang, handles, and computes Okapi ranking."""
    bm25 = BM25Index(k1=1.5, b=0.75)

    bm25.add_document("doc1", "Asmongold reacted to Blizzard WoW fresh servers announcement")
    bm25.add_document("doc2", "Shroud played Deadlock with crazy recoil control and KEKW aim")
    bm25.add_document("doc3", "xQc slammed the desk after losing a 1v1 in Overwatch")

    assert bm25.corpus_size == 3
    assert "asmongold" in bm25.df
    assert "kekw" in bm25.df

    # Query matching
    scores_deadlock = bm25.score("Deadlock recoil")
    assert "doc2" in scores_deadlock
    assert "doc1" not in scores_deadlock

    scores_wow = bm25.score("Blizzard WoW")
    assert "doc1" in scores_wow
    assert "doc3" not in scores_wow

    # Deletion test
    bm25.remove_document("doc2")
    assert bm25.corpus_size == 2
    assert "doc2" not in bm25.doc_lengths
    scores_after = bm25.score("Deadlock recoil")
    assert "doc2" not in scores_after


# ---------------------------------------------------------------------------
# 3. LocalVectorStorage SQLite Tests
# ---------------------------------------------------------------------------

def test_local_vector_storage_crud(tmp_path: Path):
    """Verify SQLite persistent storage operations, batching, and metadata filtering."""
    db_file = tmp_path / "test_vectors.db"
    storage = LocalVectorStorage(db_path=db_file)

    emb1 = np.ones(128, dtype=np.float32) / np.sqrt(128)
    emb2 = np.zeros(128, dtype=np.float32)
    emb2[0] = 1.0

    storage.insert_chunk(
        chunk_id="c1",
        vod_id="vod_asmon",
        streamer_id="asmongold",
        timestamp_sec=120.0,
        timestamp_end_sec=135.0,
        content_type="speech",
        text="World of Warcraft is the best MMO",
        metadata={"priority": "high"},
        embedding=emb1,
    )

    storage.insert_chunk(
        chunk_id="c2",
        vod_id="vod_shroud",
        streamer_id="shroud",
        timestamp_sec=45.0,
        timestamp_end_sec=50.0,
        content_type="ocr",
        text="VICTORY ROYALE",
        metadata={"visual": "screen"},
        embedding=emb2,
    )

    chunk = storage.get_chunk("c1")
    assert chunk is not None
    assert chunk["streamer_id"] == "asmongold"
    assert chunk["metadata"]["priority"] == "high"
    assert np.allclose(chunk["embedding"], emb1)

    # Filter by streamer
    asmon_chunks = storage.query_candidate_chunks(streamer_ids=["asmongold"])
    assert len(asmon_chunks) == 1
    assert asmon_chunks[0]["chunk_id"] == "c1"

    # Filter by content_type
    ocr_chunks = storage.query_candidate_chunks(content_types=["ocr"])
    assert len(ocr_chunks) == 1
    assert ocr_chunks[0]["chunk_id"] == "c2"

    # Stats
    stats = storage.get_stats()
    assert stats.total_chunks == 2
    assert stats.total_vods == 2
    assert stats.total_streamers == 2
    assert stats.chunks_by_type.get("speech") == 1
    assert stats.chunks_by_type.get("ocr") == 1

    # Delete VOD
    deleted = storage.delete_vod("vod_asmon")
    assert deleted == 1
    assert storage.get_chunk("c1") is None

    storage.close()


# ---------------------------------------------------------------------------
# 4. HybridSearchEngine Tests
# ---------------------------------------------------------------------------

def test_hybrid_search_engine():
    """Verify dense + sparse scoring, time formatting, and deep-link generation."""
    storage = LocalVectorStorage(":memory:")
    embedder = LocalEmbedder(dim=128)
    bm25 = BM25Index()
    engine = HybridSearchEngine(storage=storage, embedder=embedder, bm25=bm25)

    engine.index_chunk(
        chunk_id="ch_1",
        vod_id="asmon_vod_101",
        streamer_id="asmongold",
        timestamp_sec=85.0,
        content_type="speech",
        text="Deadlock hero balance is awful right now, Yamato needs a nerf",
        metadata={"raw_quote": "Yamato needs a nerf"},
    )
    engine.index_chunk(
        chunk_id="ch_2",
        vod_id="shroud_vod_202",
        streamer_id="shroud",
        timestamp_sec=210.0,
        content_type="speech",
        text="Deadlock matchmaking feels great when playing with high MMR players",
        metadata={"raw_quote": "Matchmaking feels great"},
    )
    engine.index_chunk(
        chunk_id="ch_3",
        vod_id="asmon_vod_101",
        streamer_id="asmongold",
        timestamp_sec=300.0,
        content_type="claim",
        text="Blizzard confirmed fresh servers for 20th anniversary (Verdict: TRUE)",
        metadata={"verdict": "TRUE"},
    )

    # Search query
    req = SearchQueryRequest(query="Deadlock matchmaking and Yamato", top_k=5, hybrid_weight=0.5)
    res = engine.search(req)

    assert res.total_results >= 2
    top = res.results[0]
    assert top.chunk_id in ("ch_1", "ch_2")
    assert top.video_url.startswith("/api/media/")
    assert top.timestamp_formatted in ("01:25", "03:30")
    assert top.combined_score > 0.0

    # Streamer filter
    req_shroud = SearchQueryRequest(query="Deadlock", streamer_ids=["shroud"])
    res_shroud = engine.search(req_shroud)
    assert all(r.streamer_id == "shroud" for r in res_shroud.results)

    # Modality filter
    req_claim = SearchQueryRequest(query="Blizzard anniversary", content_types=["claim"])
    res_claim = engine.search(req_claim)
    assert len(res_claim.results) == 1
    assert res_claim.results[0].content_type == "claim"


# ---------------------------------------------------------------------------
# 5. Multimodal VOD Ingestion Chunker Tests
# ---------------------------------------------------------------------------

def test_vod_ingestion_chunker():
    """Verify automated chunking of Whisper segments, chat bursts, OCR frames, and claims."""
    storage = LocalVectorStorage(":memory:")
    engine = HybridSearchEngine(storage=storage)

    mock_manifest = {
        "stream_id": "vod_sample_42",
        "streamer_id": "asmongold",
        "stage_outputs": {
            "audio_segments": [
                {"start_sec": 0.0, "end_sec": 5.0, "text": "What is up guys it is Asmongold here."},
                {"start_sec": 5.0, "end_sec": 12.0, "text": "Today we are looking at the new World of Warcraft updates."},
                {"start_sec": 12.0, "end_sec": 20.0, "text": "Honestly the WoW Fresh realms are going to be huge."},
            ],
            "meme_bursts": [
                {"peak_timestamp_sec": 14.5, "dominant_emote": "KEKW", "volume_multiplier": 4.5}
            ],
            "keyframes": [
                {"timestamp_sec": 15.0, "ocr_texts": ["Battle.net", "World of Warcraft"], "scene_description": "Streamer reaction cam"}
            ],
            "claims": [
                {
                    "claim_id": "cl_99",
                    "timestamp_sec": 16.0,
                    "subject_entity": "World of Warcraft",
                    "raw_quote": "WoW Fresh realms are going to be huge",
                    "verdict": "TRUE",
                }
            ],
            "chat_messages": [
                {"timestamp_offset": 14.0, "author_name": "chatter1", "content": "KEKW FRESH HYPERS"},
                {"timestamp_offset": 15.0, "author_name": "chatter2", "content": "TRUE LULW"},
            ],
        },
    }

    resp = engine.index_vod_outputs(
        vod_id="vod_sample_42",
        streamer_id="asmongold",
        manifest_or_dir=mock_manifest,
        chunk_size_sec=15.0,
        overlap_sec=3.0,
    )

    assert resp.status == "INDEXED"
    assert resp.chunks_indexed >= 4
    assert resp.speech_chunks >= 1
    assert resp.chat_chunks == 1
    assert resp.ocr_chunks == 1
    assert resp.claim_chunks == 1

    # Search for chat burst
    res_chat = engine.search(SearchQueryRequest(query="KEKW burst volume"))
    assert len(res_chat.results) >= 1
    assert any(r.content_type == "chat" for r in res_chat.results)

    # Check chat context synchronized
    speech_res = engine.search(SearchQueryRequest(query="World of Warcraft Fresh realms"))
    assert len(speech_res.results) >= 1
    first = speech_res.results[0]
    assert first.chat_context is not None
    assert len(first.chat_context) >= 1
    assert any("chatter" in c["author"] for c in first.chat_context)


# ---------------------------------------------------------------------------
# 6. MultimodalRagSynthesizer Tests
# ---------------------------------------------------------------------------

def test_multimodal_rag_synthesizer():
    """Verify cross-stream RAG synthesis generates multi-streamer viewpoints and citations."""
    storage = LocalVectorStorage(":memory:")
    engine = HybridSearchEngine(storage=storage)

    # Index Asmongold evidence
    engine.index_chunk(
        chunk_id="asmon_deadlock",
        vod_id="asmon_vod_1",
        streamer_id="asmongold",
        timestamp_sec=142.5,
        content_type="speech",
        text="Deadlock matchmaking puts 500 hour players against beginners, Valve needs to fix this immediately.",
        metadata={"raw_quote": "Deadlock matchmaking puts 500 hour players against beginners"},
    )
    # Index Shroud evidence
    engine.index_chunk(
        chunk_id="shroud_deadlock",
        vod_id="shroud_vod_2",
        streamer_id="shroud",
        timestamp_sec=88.0,
        content_type="speech",
        text="I think Deadlock matchmaking is actually fine for a closed alpha, the recoil mechanics are super crisp.",
        metadata={"raw_quote": "I think Deadlock matchmaking is actually fine for a closed alpha"},
    )
    # Index Claim
    engine.index_chunk(
        chunk_id="claim_deadlock",
        vod_id="asmon_vod_1",
        streamer_id="asmongold",
        timestamp_sec=145.0,
        content_type="claim",
        text="Valve updated matchmaking algorithm on Thursday (Verdict: TRUE)",
        metadata={"verdict": "TRUE", "raw_quote": "Valve updated matchmaking algorithm"},
    )

    synthesizer = MultimodalRagSynthesizer(engine)
    req = RagSynthesisRequest(query="What did Asmongold and Shroud say about Deadlock's matchmaking?", top_k=6)
    res = synthesizer.synthesize(req)

    assert "asmongold" in res.streamers_covered
    assert "shroud" in res.streamers_covered
    assert len(res.citations) >= 2

    # Verify citation fields
    cit = res.citations[0]
    assert cit.video_url.startswith("/api/media/")
    assert cit.quote != ""
    assert cit.relevance_score > 0.0

    # Answer text should contain formatted markdown breakdown
    assert "Asmongold" in res.answer
    assert "Shroud" in res.answer
    assert "Direct Timestamp Links" in res.answer


# ---------------------------------------------------------------------------
# 7. Web API Endpoints Tests
# ---------------------------------------------------------------------------

def test_web_search_and_rag_endpoints(tmp_path: Path):
    """Test FastAPI REST endpoints for semantic search, RAG synthesis, stats, and indexing."""
    catalog = HarvestCatalog(f"sqlite:///{tmp_path}/test_catalog.db")
    storage = LocalVectorStorage(tmp_path / "web_vectors.db")
    engine = HybridSearchEngine(storage=storage)

    # Seed catalog VOD
    video_file = tmp_path / "sample.mp4"
    video_file.write_bytes(b"mock video data")
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps({
        "stream_id": "vod_web_001",
        "streamer_id": "asmongold",
        "stage_outputs": {
            "audio_segments": [{"start_sec": 10.0, "end_sec": 20.0, "text": "Blizzard announced fresh servers today."}],
            "claims": [{"claim_id": "c1", "timestamp_sec": 15.0, "raw_quote": "Fresh servers are real", "verdict": "TRUE"}],
        },
    }), encoding="utf-8")

    # Seed streamer target first for foreign key integrity
    from stream_fusion.models.schemas import StreamerTargetRecord
    catalog.upsert_streamer_target(StreamerTargetRecord(
        streamer_id="asmongold",
        display_name="Asmongold",
        channel_urls=["https://twitch.tv/asmongold"],
        primary_platform="twitch",
        download_priority=8,
        enabled=True,
    ))

    catalog.upsert_harvested_vod(HarvestedVodRecord(
        vod_id="vod_web_001",
        streamer_id="asmongold",
        platform="twitch",
        title="Testing WoW Fresh",
        video_path=str(video_file),
        metadata_path=str(manifest_file),
        duration_sec=60.0,
    ))

    config = DashboardConfig(homelab_root=str(tmp_path))
    app_instance = create_app(config=config, catalog=catalog, search_engine=engine)
    client = TestClient(app_instance)

    # 1. Index VOD endpoint
    idx_res = client.post("/api/index/vod/vod_web_001", json={"chunk_size_sec": 15.0})
    assert idx_res.status_code == 200
    idx_data = idx_res.json()
    assert idx_data["status"] == "INDEXED"
    assert idx_data["chunks_indexed"] >= 2

    # 2. Stats endpoint
    stats_res = client.get("/api/search/stats")
    assert stats_res.status_code == 200
    stats_data = stats_res.json()
    assert stats_data["total_chunks"] >= 2
    assert stats_data["total_vods"] == 1
    assert stats_data["total_streamers"] == 1

    # 3. Semantic search endpoint
    search_res = client.post("/api/search/semantic", json={
        "query": "Blizzard fresh servers",
        "top_k": 5,
        "hybrid_weight": 0.5,
    })
    assert search_res.status_code == 200
    s_data = search_res.json()
    assert s_data["total_results"] >= 1
    assert s_data["results"][0]["streamer_id"] == "asmongold"

    # 4. RAG synthesis endpoint
    rag_res = client.post("/api/search/rag", json={
        "query": "Did Asmongold say Blizzard announced fresh servers?",
        "top_k": 5,
    })
    assert rag_res.status_code == 200
    r_data = rag_res.json()
    assert "Asmongold" in r_data["answer"]
    assert len(r_data["citations"]) >= 1

    # 5. Media streaming endpoint
    media_res = client.get("/api/media/vod_web_001/video")
    assert media_res.status_code == 200
    assert media_res.content == b"mock video data"

    # 6. Delete index endpoint
    del_res = client.delete("/api/index/vod/vod_web_001")
    assert del_res.status_code == 200
    del_data = del_res.json()
    assert del_data["status"] == "DELETED"
    assert del_data["chunks_deleted"] >= 2


# ---------------------------------------------------------------------------
# 8. CLI Search Commands Tests
# ---------------------------------------------------------------------------

def test_cli_search_commands(tmp_path: Path):
    """Verify CLI search query, rag, and stats commands."""
    runner = CliRunner()

    # Pre-populate index in storage
    storage = LocalVectorStorage(tmp_path / "knowledge" / "vector_index.db")
    engine = HybridSearchEngine(storage=storage, homelab_root=tmp_path)
    engine.index_chunk(
        chunk_id="cli_01",
        vod_id="vod_cli",
        streamer_id="asmongold",
        timestamp_sec=55.0,
        content_type="speech",
        text="Dark and Darker playtest is live this weekend guys",
    )

    # CLI Stats
    res_stats = runner.invoke(app, ["search", "stats", "--homelab-root", str(tmp_path)])
    assert res_stats.exit_code == 0
    assert "Spec 28 Vector Index Statistics" in res_stats.stdout

    # CLI Query
    res_q = runner.invoke(app, ["search", "query", "Dark and Darker", "--homelab-root", str(tmp_path)])
    assert res_q.exit_code == 0
    assert "Retrieved Multimodal Evidence" in res_q.stdout
    assert "asmongold" in res_q.stdout

    # CLI RAG
    res_rag = runner.invoke(app, ["search", "rag", "What did Asmongold say about Dark and Darker?", "--homelab-root", str(tmp_path)])
    assert res_rag.exit_code == 0
    assert "RAG Synthesis" in res_rag.stdout


# ---------------------------------------------------------------------------
# 9. JsonVectorStorage & create_vector_storage Factory Tests
# ---------------------------------------------------------------------------

def test_json_vector_storage(tmp_path: Path):
    """Verify transparent JSON flatfile vector storage operations and persistence."""
    from stream_fusion.knowledge.search import JsonVectorStorage

    json_file = tmp_path / "custom_vectors.json"
    storage = JsonVectorStorage(file_path=json_file, dim=128)

    emb = np.zeros(128, dtype=np.float32)
    emb[0] = 1.0

    storage.insert_chunk(
        chunk_id="j_chunk_1",
        vod_id="vod_json_1",
        streamer_id="shroud",
        timestamp_sec=30.0,
        timestamp_end_sec=45.0,
        content_type="speech",
        text="Clean FPS gameplay without any frame drops",
        metadata={"res": "1440p"},
        embedding=emb,
    )

    assert json_file.exists()

    # Reload from disk into fresh storage instance
    storage_reloaded = JsonVectorStorage(file_path=json_file, dim=128)
    chunk = storage_reloaded.get_chunk("j_chunk_1")
    assert chunk is not None
    assert chunk["streamer_id"] == "shroud"
    assert chunk["metadata"]["res"] == "1440p"
    assert np.allclose(chunk["embedding"], emb)

    # Candidate query
    cands = storage_reloaded.query_candidate_chunks(streamer_ids=["shroud"])
    assert len(cands) == 1
    assert cands[0]["chunk_id"] == "j_chunk_1"

    # Nearest neighbor query
    nn = storage_reloaded.query_nearest_neighbors(query_vec=emb, top_k=5)
    assert len(nn) == 1
    assert np.isclose(nn[0][1], 1.0, atol=1e-4)

    # Stats
    stats = storage_reloaded.get_stats()
    assert stats.total_chunks == 1
    assert stats.total_streamers == 1
    assert stats.index_storage_bytes > 0

    # Delete VOD
    deleted = storage_reloaded.delete_vod("vod_json_1")
    assert deleted == 1
    assert storage_reloaded.get_chunk("j_chunk_1") is None


def test_create_vector_storage_factory(tmp_path: Path):
    """Verify create_vector_storage resolves to correct storage engine."""
    from stream_fusion.knowledge.search import (
        JsonVectorStorage,
        LocalVectorStorage,
        PgVectorStorage,
        create_vector_storage,
    )

    # JSON URL
    s_json = create_vector_storage(f"json://{tmp_path}/index.json")
    assert isinstance(s_json, JsonVectorStorage)

    # SQLite URL
    s_sqlite = create_vector_storage(f"sqlite:///{tmp_path}/test.db")
    assert isinstance(s_sqlite, LocalVectorStorage)


def test_pgvector_storage_sql_execution(monkeypatch):
    """Verify PgVectorStorage schema setup, HNSW index generation, and vector cosine queries with mock."""
    from unittest.mock import MagicMock
    from stream_fusion.knowledge.search import PgVectorStorage

    # Mock psycopg2
    mock_cursor = MagicMock()
    mock_cursor.__enter__.return_value = mock_cursor
    mock_cursor.rowcount = 1
    mock_cursor.fetchone.return_value = (10, 2, 1)
    mock_cursor.fetchall.return_value = [
        ("speech", 8),
        ("claim", 2),
    ]

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_conn.__enter__.return_value = mock_conn

    mock_psycopg2 = MagicMock()
    mock_psycopg2.connect.return_value = mock_conn
    monkeypatch.setattr("stream_fusion.knowledge.search.psycopg2", mock_psycopg2, raising=False)
    monkeypatch.setattr("stream_fusion.knowledge.search.psycopg", None, raising=False)

    import sys
    sys.modules["psycopg2"] = mock_psycopg2

    storage = PgVectorStorage("postgresql://testuser:testpass@localhost:5432/streamfusion", dim=128)

    # Verify extension and table creation were executed
    executed_statements = [call[0][0] for call in mock_cursor.execute.call_args_list]
    assert any("CREATE EXTENSION IF NOT EXISTS vector" in stmt for stmt in executed_statements)
    assert any("vector(128)" in stmt for stmt in executed_statements)
    assert any("USING hnsw (embedding vector_cosine_ops)" in stmt for stmt in executed_statements)

    # Test insert
    emb = np.zeros(128, dtype=np.float32)
    storage.insert_chunk(
        chunk_id="pg_1",
        vod_id="vod_pg",
        streamer_id="asmongold",
        timestamp_sec=10.0,
        timestamp_end_sec=20.0,
        content_type="speech",
        text="PostgreSQL with pgvector is scalable",
        metadata={"priority": "high"},
        embedding=emb,
    )

    # Test nearest neighbor query
    mock_cursor.fetchall.return_value = [
        ("pg_1", "vod_pg", "asmongold", 10.0, 20.0, "speech", "PostgreSQL with pgvector is scalable", '{"priority": "high"}', "[0,0,0]", 0.98),
    ]
    results = storage.query_nearest_neighbors(query_vec=emb, top_k=5)
    assert len(results) == 1
    assert results[0][0]["chunk_id"] == "pg_1"
    assert results[0][1] == 0.98

    # Verify HNSW cosine operator <=> was queried
    last_statements = [call[0][0] for call in mock_cursor.execute.call_args_list]
    assert any("<=> %s::vector" in stmt for stmt in last_statements)

