"""Unit and Integration Tests for Spec 29: Polyglot Data Stack & Decoupled Homelab Topology."""

import numpy as np
import pytest

from stream_fusion.knowledge.clickhouse import ClickHouseChatStorage
from stream_fusion.knowledge.search import (
    BaseVectorStorage,
    LocalVectorStorage,
    QdrantVectorStorage,
    create_vector_storage,
)
from stream_fusion.models.schemas import DashboardConfig


# ---------------------------------------------------------------------------
# 1. DashboardConfig Polyglot Endpoints
# ---------------------------------------------------------------------------

def test_dashboard_config_polyglot_endpoints():
    """Verify DashboardConfig supports PostgreSQL, ClickHouse, Qdrant, and NAS endpoints."""
    # Default state: optional endpoints are None
    cfg_default = DashboardConfig()
    assert cfg_default.postgres_url is None
    assert cfg_default.clickhouse_url is None
    assert cfg_default.qdrant_url is None
    assert cfg_default.media_storage_path is None

    # Custom polyglot topology configuration
    cfg = DashboardConfig(
        postgres_url="postgresql://streamfusion:secret@192.168.1.100:5432/streamfusion",
        clickhouse_url="http://192.168.1.100:8123",
        qdrant_url="http://192.168.1.100:6333",
        media_storage_path="//192.168.1.100/storage/media",
    )

    assert cfg.postgres_url == "postgresql://streamfusion:secret@192.168.1.100:5432/streamfusion"
    assert cfg.clickhouse_url == "http://192.168.1.100:8123"
    assert cfg.qdrant_url == "http://192.168.1.100:6333"
    assert cfg.media_storage_path == "//192.168.1.100/storage/media"

    dumped = cfg.model_dump()
    assert dumped["postgres_url"] == cfg.postgres_url
    assert dumped["clickhouse_url"] == cfg.clickhouse_url
    assert dumped["qdrant_url"] == cfg.qdrant_url
    assert dumped["media_storage_path"] == cfg.media_storage_path


# ---------------------------------------------------------------------------
# 2. QdrantVectorStorage CRUD and Binary Quantization
# ---------------------------------------------------------------------------

def test_qdrant_vector_storage_crud_and_bq():
    """Verify QdrantVectorStorage CRUD operations and Binary Quantization parameters."""
    storage = QdrantVectorStorage(
        url="qdrant://192.168.1.100:6333",
        dim=128,
        enable_binary_quantization=True,
        force_mock=True,
    )

    assert storage.dim == 128
    assert storage.base_url == "http://192.168.1.100:6333"
    assert storage.enable_binary_quantization is True
    assert storage.quantization_config == {"binary": {"always_ram": True}}
    assert storage.is_mock is True

    # 1. Insert single chunk
    emb = np.ones(128, dtype=np.float32) / np.sqrt(128)
    storage.insert_chunk(
        chunk_id="vod1_speech_001",
        vod_id="vod1",
        streamer_id="asmongold",
        timestamp_sec=14.5,
        timestamp_end_sec=22.0,
        content_type="speech",
        text="Blizzard announced WoW fresh progression servers",
        metadata={"speaker_id": "SPEAKER_00", "burst": False},
        embedding=emb,
    )

    # 2. Retrieve chunk
    retrieved = storage.get_chunk("vod1_speech_001")
    assert retrieved is not None
    assert retrieved["chunk_id"] == "vod1_speech_001"
    assert retrieved["streamer_id"] == "asmongold"
    assert retrieved["text"] == "Blizzard announced WoW fresh progression servers"
    assert np.isclose(np.linalg.norm(retrieved["embedding"]), 1.0, atol=1e-4)

    # 3. Check stats
    stats = storage.get_stats()
    assert stats.total_chunks == 1
    assert stats.total_vods == 1
    assert stats.total_streamers == 1
    assert stats.chunks_by_type.get("speech") == 1

    # 4. Delete VOD
    deleted_count = storage.delete_vod("vod1")
    assert deleted_count == 1
    assert storage.get_chunk("vod1_speech_001") is None
    assert storage.get_stats().total_chunks == 0


def test_qdrant_vector_storage_batch_filtering_and_knn():
    """Verify QdrantVectorStorage batch insertions, metadata filtering, and k-NN query."""
    storage = QdrantVectorStorage(dim=64, force_mock=True)

    vec_a = np.zeros(64, dtype=np.float32)
    vec_a[0] = 1.0

    vec_b = np.zeros(64, dtype=np.float32)
    vec_b[1] = 1.0

    vec_c = np.zeros(64, dtype=np.float32)
    vec_c[2] = 1.0

    chunks = [
        {
            "chunk_id": "c1",
            "vod_id": "vod_asmon",
            "streamer_id": "asmongold",
            "timestamp_sec": 10.0,
            "timestamp_end_sec": 20.0,
            "content_type": "speech",
            "text": "Deadlock match pacing",
            "metadata": {},
            "embedding": vec_a,
        },
        {
            "chunk_id": "c2",
            "vod_id": "vod_asmon",
            "streamer_id": "asmongold",
            "timestamp_sec": 30.0,
            "timestamp_end_sec": 40.0,
            "content_type": "chat_burst",
            "text": "KEKW burst in chat",
            "metadata": {"dominant_emote": "KEKW"},
            "embedding": vec_b,
        },
        {
            "chunk_id": "c3",
            "vod_id": "vod_shroud",
            "streamer_id": "shroud",
            "timestamp_sec": 15.0,
            "timestamp_end_sec": 25.0,
            "content_type": "speech",
            "text": "Shroud weapon recoil",
            "metadata": {},
            "embedding": vec_c,
        },
    ]

    inserted = storage.insert_chunks_batch(chunks)
    assert inserted == 3
    assert storage.get_stats().total_chunks == 3

    # Metadata filtering by streamer
    asmon_cands = storage.query_candidate_chunks(streamer_ids=["asmongold"])
    assert len(asmon_cands) == 2
    assert all(c["streamer_id"] == "asmongold" for c in asmon_cands)

    # Metadata filtering by content type
    burst_cands = storage.query_candidate_chunks(content_types=["chat_burst"])
    assert len(burst_cands) == 1
    assert burst_cands[0]["chunk_id"] == "c2"

    # Time range filtering
    time_cands = storage.query_candidate_chunks(start_time=12.0, end_time=35.0)
    assert len(time_cands) == 2
    cand_ids = {c["chunk_id"] for c in time_cands}
    assert cand_ids == {"c2", "c3"}

    # Nearest neighbor search
    query_vec = np.zeros(64, dtype=np.float32)
    query_vec[0] = 0.95
    query_vec[1] = 0.05
    query_vec /= np.linalg.norm(query_vec)

    results = storage.query_nearest_neighbors(query_vec=query_vec, top_k=2)
    assert len(results) == 2
    assert results[0][0]["chunk_id"] == "c1"
    assert results[0][1] > results[1][1]

    # Nearest neighbor with streamer filter
    shroud_results = storage.query_nearest_neighbors(
        query_vec=query_vec, top_k=5, streamer_ids=["shroud"]
    )
    assert len(shroud_results) == 1
    assert shroud_results[0][0]["chunk_id"] == "c3"


def test_create_vector_storage_qdrant_routing():
    """Verify create_vector_storage factory routes qdrant:// and port 6333 URLs to QdrantVectorStorage."""
    s1 = create_vector_storage("qdrant://127.0.0.1:6333", dim=128)
    assert isinstance(s1, QdrantVectorStorage)
    assert s1.base_url == "http://127.0.0.1:6333"
    assert s1.dim == 128

    s2 = create_vector_storage("http://192.168.1.50:6333", dim=256)
    assert isinstance(s2, QdrantVectorStorage)
    assert s2.base_url == "http://192.168.1.50:6333"
    assert s2.dim == 256

    s_sqlite = create_vector_storage(":memory:", dim=128)
    assert isinstance(s_sqlite, LocalVectorStorage)


# ---------------------------------------------------------------------------
# 3. ClickHouseChatStorage Schema, Ingestion, and ASOF Join
# ---------------------------------------------------------------------------

def test_clickhouse_chat_storage_schema_and_batch_insert():
    """Verify ClickHouseChatStorage schema initialization and batch insertions."""
    storage = ClickHouseChatStorage(url="http://localhost:8123", force_mock=True)
    assert storage.base_url == "http://localhost:8123"
    assert storage.is_mock is True

    storage.init_schema()

    # Ingest chat events
    chat_events = [
        {
            "streamer_id": "asmongold",
            "vod_id": "vod_100",
            "timestamp_sec": 10.5,
            "chatter_username": "greg",
            "message_text": "KEKW THAT AIM",
            "emotes": ["KEKW"],
            "sentiment_score": 0.75,
            "is_subscriber": 1,
            "burst_flag": 1,
        },
        {
            "streamer_id": "asmongold",
            "vod_id": "vod_100",
            "timestamp_sec": 13.0,
            "chatter_username": "sarah",
            "message_text": "classic blizzard",
            "emotes": ["LULW"],
            "sentiment_score": -0.2,
            "is_subscriber": 0,
            "burst_flag": 0,
        },
    ]

    inserted_chats = storage.insert_chat_events_batch(chat_events)
    assert inserted_chats == 2
    assert storage.get_chat_events_count("vod_100") == 2
    assert storage.get_chat_events_count("other_vod") == 0

    # Ingest speech segments
    speech_segments = [
        {
            "streamer_id": "asmongold",
            "vod_id": "vod_100",
            "start_sec": 9.0,
            "end_sec": 12.0,
            "speaker_id": "SPEAKER_00",
            "transcript": "I can not believe this game is real.",
        }
    ]

    inserted_speech = storage.insert_speech_segments_batch(speech_segments)
    assert inserted_speech == 1
    assert storage.get_speech_segments_count("vod_100") == 1


def test_clickhouse_asof_query_generation_and_alignment():
    """Verify ClickHouse ASOF JOIN query generation and temporal reaction alignment."""
    storage = ClickHouseChatStorage(url="clickhouse://192.168.1.100:8123", force_mock=True)

    # 1. Verify SQL Query Generation
    query = storage.generate_asof_query(vod_id="vod_200", window_sec=8.0, streamer_id="shroud")
    assert "FROM speech_segments s" in query
    assert "ASOF LEFT JOIN chat_events c" in query
    assert "s.vod_id = c.vod_id" in query
    assert "s.streamer_id = c.streamer_id" in query
    assert "s.start_ms <= c.timestamp_ms" in query
    assert "c.timestamp_ms <= s.start_ms + 8000" in query
    assert "s.streamer_id = 'shroud'" in query

    # 2. Insert speech segments & chat reactions
    storage.insert_speech_segments_batch([
        {
            "streamer_id": "shroud",
            "vod_id": "vod_200",
            "start_ms": 10000,  # 10.0s
            "end_ms": 14000,    # 14.0s
            "speaker_id": "shroud",
            "transcript": "Look at this insane 180 flick shot!",
        },
        {
            "streamer_id": "shroud",
            "vod_id": "vod_200",
            "start_ms": 60000,  # 60.0s
            "end_ms": 65000,    # 65.0s
            "speaker_id": "shroud",
            "transcript": "Quiet lobby right now...",
        },
    ])

    storage.insert_chat_events_batch([
        {
            "streamer_id": "shroud",
            "vod_id": "vod_200",
            "timestamp_ms": 12500,  # 12.5s (2.5s after speech 1 -> within 8s window)
            "chatter_username": "aimbot_god",
            "message_text": "POGGERS WHAT A SHOT",
            "emotes": ["POGGERS"],
        },
        {
            "streamer_id": "shroud",
            "vod_id": "vod_200",
            "timestamp_ms": 15000,  # 15.0s (later reaction)
            "chatter_username": "chill_dude",
            "message_text": "insane",
            "emotes": [],
        },
    ])

    # 3. Perform ASOF Alignment
    aligned = storage.asof_align_chat_reactions(vod_id="vod_200", window_sec=8.0, streamer_id="shroud")
    assert len(aligned) == 2

    # Speech segment 1: matched closest reaction 'aimbot_god' at 12.5s
    seg1 = aligned[0]
    assert seg1["transcript"] == "Look at this insane 180 flick shot!"
    assert seg1["chatter_username"] == "aimbot_god"
    assert seg1["message_text"] == "POGGERS WHAT A SHOT"
    assert seg1["emotes"] == ["POGGERS"]

    # Speech segment 2: at 60.0s, no chat message exists in [60s, 68s] -> NULL left join behavior
    seg2 = aligned[1]
    assert seg2["transcript"] == "Quiet lobby right now..."
    assert seg2["chatter_username"] is None
    assert seg2["message_text"] is None
    assert seg2["emotes"] == []
