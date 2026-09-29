"""Streamer Knowledge Graph, Semantic Claim Extraction & Additive Belief Synthesis (Spec 13).

Extracts stances, opinions, and critiques from streamer speech, models beliefs as
additive graphs over time, and supports natural language VOD querying.
"""

from datetime import datetime, timezone
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from stream_fusion.chat.nlp import tokenize_for_similarity
from stream_fusion.knowledge.vector_index import LocalVectorIndex
from stream_fusion.models.schemas import (
    AudioSegment,
    EntityStanceRecord,
    StreamerClaim,
)


OPINION_TRIGGERS: List[Tuple[str, str, float]] = [
    # (regex_pattern, stance, polarity)
    (r"(liked|really liked|loved|enjoyed|super good|amazing|masterpiece|best)", "APPROVAL", 0.8),
    (r"(hate|hated|terrible|awful|garbage|trash|worst|cooked|bad)", "DISAPPROVAL", -0.8),
    (r"(okay|decent|mid|average|fine|not bad)", "NEUTRAL", 0.1),
]

SUB_ATTRIBUTE_KEYWORDS: Dict[str, str] = {
    "special effects": "special_effects",
    "cgi": "special_effects",
    "vfx": "special_effects",
    "graphics": "graphics",
    "gameplay": "gameplay",
    "combat": "gameplay",
    "story": "story",
    "plot": "story",
    "ending": "story",
    "soundtrack": "soundtrack",
    "music": "soundtrack",
    "performance": "performance",
    "fps": "performance",
    "movie": "overall_verdict",
    "film": "overall_verdict",
    "game": "overall_verdict",
}


class ClaimExtractor:
    """Extracts structured claims, stances, and polarities from transcribed streamer audio."""

    def extract_claims(
        self,
        audio_segments: Optional[List[AudioSegment]] = None,
        keyframes: Optional[List[Any]] = None,
        creator_id: str = "Streamer",
        vod_id: str = "unknown_vod",
    ) -> List[StreamerClaim]:
        """Extracts claims across a list of audio segments."""
        all_claims: List[StreamerClaim] = []
        if audio_segments:
            for seg in audio_segments:
                claims = self.extract_claims_from_transcript(seg, creator_id=creator_id, vod_id=vod_id)
                all_claims.extend(claims)
        return all_claims

    def extract_claims_from_transcript(

        self,
        segment: AudioSegment,
        creator_id: str,
        vod_id: str = "unknown_vod",
        entity_hint: Optional[str] = None,
    ) -> List[StreamerClaim]:
        """Scans an audio segment for opinions and claims."""
        text = segment.transcript.strip()
        text_lower = text.lower()
        claims: List[StreamerClaim] = []

        # Find opinion stance
        matched_stance = "NEUTRAL"
        matched_polarity = 0.0

        for pattern, stance, pol in OPINION_TRIGGERS:
            if re.search(pattern, text_lower):
                matched_stance = stance
                matched_polarity = pol
                break

        if matched_stance == "NEUTRAL" and matched_polarity == 0.0:
            return []  # No strong stance detected in this segment

        # Identify sub-attribute
        sub_attr = "overall_verdict"
        for kw, attr in SUB_ATTRIBUTE_KEYWORDS.items():
            if kw in text_lower:
                sub_attr = attr
                break

        # Extract or infer subject entity
        subject = entity_hint
        if not subject:
            # Fallback to noun phrases or capitalized tokens
            words = text.split()
            caps = [w.strip(".,!?") for w in words if w and w[0].isupper() and w.lower() not in {"i", "the", "a", "this", "it"}]
            subject = " ".join(caps[:2]) if caps else "Stream Topic"

        claim = StreamerClaim(
            claim_id=f"claim_{creator_id}_{vod_id}_{int(segment.start_sec)}",
            creator_id=creator_id,
            vod_id=vod_id,
            timestamp_sec=segment.start_sec,
            end_sec=segment.end_sec,
            topic_category="ENTERTAINMENT",
            subject_entity=subject,
            predicate=f"has_{sub_attr}",
            object_value=f"{matched_stance.lower()}_{sub_attr}",
            stance=matched_stance,
            polarity=matched_polarity,
            confidence=round(segment.confidence, 2),
            raw_quote=text,
            visual_context_summary="",
            supersedes_claim_id=None,
            is_stance_reversal=False,
        )
        claims.append(claim)
        return claims


class BeliefGraphEngine:
    """Synthesizes claims across time into additive, nuanced entity stance records."""

    def synthesize_stance(
        self,
        creator_id: str,
        subject_entity: str,
        claims: List[StreamerClaim],
    ) -> EntityStanceRecord:
        """Aggregates multiple claims into an entity belief record."""
        sub_attrs: Dict[str, List[float]] = {}

        for c in claims:
            attr = c.predicate.replace("has_", "")
            sub_attrs.setdefault(attr, []).append(c.polarity)

        # Compute average polarity per sub-attribute
        sub_attribute_averages = {
            attr: round(sum(pols) / len(pols), 2) for attr, pols in sub_attrs.items()
        }

        # Overall aggregate polarity across all claims
        total_pols = [c.polarity for c in claims]
        agg_pol = round(sum(total_pols) / len(total_pols), 2) if total_pols else 0.0

        if agg_pol >= 0.35:
            overall_stance = "APPROVAL"
        elif agg_pol <= -0.35:
            overall_stance = "DISAPPROVAL"
        else:
            overall_stance = "MIXED"

        return EntityStanceRecord(
            creator_id=creator_id,
            subject_entity=subject_entity,
            aggregate_polarity=agg_pol,
            overall_stance=overall_stance,
            claims_count=len(claims),
            sub_attributes=sub_attribute_averages,
            last_updated=datetime.now(timezone.utc).isoformat(),
        )


class StreamerKnowledgeStore:
    """Relational SQLite database and vector index for streamer knowledge archival."""

    def __init__(self, db_path: str = ":memory:", vector_index: Optional[LocalVectorIndex] = None):
        self.conn = sqlite3.connect(db_path)
        self.vector_index = vector_index or LocalVectorIndex()
        self.extractor = ClaimExtractor()
        self.belief_engine = BeliefGraphEngine()
        self._init_db()

    def _init_db(self):
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS claims (
                claim_id TEXT PRIMARY KEY,
                creator_id TEXT NOT NULL,
                vod_id TEXT NOT NULL,
                timestamp_sec REAL NOT NULL,
                end_sec REAL NOT NULL,
                subject_entity TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object_value TEXT NOT NULL,
                stance TEXT NOT NULL,
                polarity REAL NOT NULL,
                raw_quote TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def ingest_audio_segments(
        self,
        segments: List[AudioSegment],
        creator_id: str,
        vod_id: str = "vod_sample",
        entity_hint: Optional[str] = None,
    ) -> List[StreamerClaim]:
        """Extracts and stores claims from streamer segments, indexing transcripts for semantic search."""
        all_claims: List[StreamerClaim] = []
        cur = self.conn.cursor()

        for seg in segments:
            # Index all streamer speech into vector database
            self.vector_index.add_document(
                doc_id=f"{creator_id}_{vod_id}_{int(seg.start_sec)}",
                text=seg.transcript,
                metadata={
                    "creator_id": creator_id,
                    "vod_id": vod_id,
                    "timestamp_sec": seg.start_sec,
                    "end_sec": seg.end_sec,
                },
            )

            # Extract structured claims
            claims = self.extractor.extract_claims_from_transcript(
                seg, creator_id=creator_id, vod_id=vod_id, entity_hint=entity_hint
            )
            for c in claims:
                cur.execute("""
                    INSERT OR REPLACE INTO claims VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    c.claim_id, c.creator_id, c.vod_id, c.timestamp_sec, c.end_sec,
                    c.subject_entity, c.predicate, c.object_value, c.stance, c.polarity, c.raw_quote
                ))
                all_claims.append(c)

        self.conn.commit()
        return all_claims

    def query_streamer_knowledge(
        self, query_text: str, creator_id: Optional[str] = None, top_k: int = 5
    ) -> Dict[str, Any]:
        """Performs semantic search across VOD transcripts and returns associated claim synthesis."""
        filt = {"creator_id": creator_id} if creator_id else None
        matches = self.vector_index.query(query_text, top_k=top_k, filter_metadata=filt)

        # Retrieve relevant claims from relational DB
        cur = self.conn.cursor()
        if creator_id:
            cur.execute("SELECT * FROM claims WHERE creator_id = ?", (creator_id,))
        else:
            cur.execute("SELECT * FROM claims")
        rows = cur.fetchall()

        claims = [
            StreamerClaim(
                claim_id=r[0], creator_id=r[1], vod_id=r[2], timestamp_sec=r[3], end_sec=r[4],
                subject_entity=r[5], predicate=r[6], object_value=r[7], stance=r[8],
                polarity=r[9], raw_quote=r[10],
            )
            for r in rows
        ]

        # Group claims by entity
        entity_groups: Dict[str, List[StreamerClaim]] = {}
        for c in claims:
            entity_groups.setdefault(c.subject_entity, []).append(c)

        stances = [
            self.belief_engine.synthesize_stance(creator_id or "all", ent, cl_list)
            for ent, cl_list in entity_groups.items()
        ]

        return {
            "query": query_text,
            "vector_matches": matches,
            "synthesized_stances": stances,
        }
