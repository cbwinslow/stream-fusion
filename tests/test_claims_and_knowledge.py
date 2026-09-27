"""Unit tests for Streamer Knowledge Graph, Claims, and Vector Index (Spec 13)."""

from pathlib import Path
import pytest
from stream_fusion.knowledge.claims import (
    BeliefGraphEngine,
    ClaimExtractor,
    StreamerKnowledgeStore,
)
from stream_fusion.knowledge.vector_index import LocalVectorIndex
from stream_fusion.models.schemas import AudioSegment


def test_claim_extractor_opinion_detection():
    extractor = ClaimExtractor()

    # Day 1: Asmon critiquing CGI in trailer
    seg1 = AudioSegment(
        segment_id=1,
        start_sec=10.0,
        end_sec=16.0,
        speaker_label="STREAMER",
        transcript="The CGI special effects on that monster look terrible and cooked.",
    )
    claims1 = extractor.extract_claims_from_transcript(
        seg1, creator_id="asmongold", vod_id="vod_day1", entity_hint="Godzilla Minus One"
    )
    assert len(claims1) == 1
    assert claims1[0].subject_entity == "Godzilla Minus One"
    assert claims1[0].predicate == "has_special_effects"
    assert claims1[0].stance == "DISAPPROVAL"
    assert claims1[0].polarity < 0.0

    # Day 7: Asmon reviewing the full film
    seg2 = AudioSegment(
        segment_id=2,
        start_sec=100.0,
        end_sec=105.0,
        speaker_label="STREAMER",
        transcript="I really liked the movie, it was super good and emotional.",
    )
    claims2 = extractor.extract_claims_from_transcript(
        seg2, creator_id="asmongold", vod_id="vod_day7", entity_hint="Godzilla Minus One"
    )
    assert len(claims2) == 1
    assert claims2[0].subject_entity == "Godzilla Minus One"
    assert claims2[0].predicate == "has_overall_verdict"
    assert claims2[0].stance == "APPROVAL"
    assert claims2[0].polarity > 0.0


def test_additive_belief_synthesis():
    engine = BeliefGraphEngine()
    extractor = ClaimExtractor()

    seg1 = AudioSegment(
        segment_id=1, start_sec=10.0, end_sec=15.0, speaker_label="STREAMER",
        transcript="The CGI special effects look terrible.",
    )
    seg2 = AudioSegment(
        segment_id=2, start_sec=100.0, end_sec=105.0, speaker_label="STREAMER",
        transcript="I really liked the movie overall, amazing story.",
    )

    c1 = extractor.extract_claims_from_transcript(seg1, "asmongold", "v1", entity_hint="Godzilla")[0]
    c2 = extractor.extract_claims_from_transcript(seg2, "asmongold", "v2", entity_hint="Godzilla")[0]

    record = engine.synthesize_stance("asmongold", "Godzilla", [c1, c2])
    assert record.subject_entity == "Godzilla"
    assert record.claims_count == 2
    # Verify sub-attributes decomposition:
    assert "special_effects" in record.sub_attributes
    assert record.sub_attributes["special_effects"] < 0.0
    assert ("overall_verdict" in record.sub_attributes or "story" in record.sub_attributes)
    pos_attr = "overall_verdict" if "overall_verdict" in record.sub_attributes else "story"
    assert record.sub_attributes[pos_attr] > 0.0


def test_local_vector_index(tmp_path: Path):
    idx_path = tmp_path / "test_vec.json"
    index = LocalVectorIndex(storage_path=idx_path)

    index.add_document("doc1", "Asmongold reacts to World of Warcraft expansion", metadata={"creator": "asmon"})
    index.add_document("doc2", "Hutch playing DayZ with BurntPeanut", metadata={"creator": "hutch"})
    index.add_document("doc3", "Cooking steak with Gordon Ramsay", metadata={"creator": "cooking"})

    # Query
    matches = index.query("World of Warcraft raid", top_k=2)
    assert len(matches) > 0
    assert matches[0]["doc_id"] == "doc1"

    # Query with creator filter
    filtered = index.query("DayZ survival", top_k=2, filter_metadata={"creator": "hutch"})
    assert len(filtered) == 1
    assert filtered[0]["doc_id"] == "doc2"


def test_streamer_knowledge_store_end_to_end(tmp_path: Path):
    store = StreamerKnowledgeStore(db_path=":memory:")

    segments = [
        AudioSegment(
            segment_id=1, start_sec=42.0, end_sec=48.0, speaker_label="STREAMER",
            transcript="I really loved the ending of that Godzilla movie.",
        )
    ]

    claims = store.ingest_audio_segments(segments, creator_id="asmongold", vod_id="vod_asmon_01", entity_hint="Godzilla")
    assert len(claims) == 1

    result = store.query_streamer_knowledge("Godzilla ending", creator_id="asmongold")
    assert len(result["vector_matches"]) > 0
    assert len(result["synthesized_stances"]) == 1
    assert result["synthesized_stances"][0].subject_entity == "Godzilla"
