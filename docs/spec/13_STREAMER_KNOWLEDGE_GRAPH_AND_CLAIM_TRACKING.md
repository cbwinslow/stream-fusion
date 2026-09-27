# StreamFusion: Streamer Knowledge Graph, Claim Archival & Vector Index (Spec 13)

## 1. North Star & Objectives
Streamers produce hundreds of hours of unscripted commentary monthly, expressing opinions on video games, movies, technology, politics, current events, and online culture. Today, this massive repository of streamer knowledge is lost in unindexed video files.

### Core Objectives
1. **Semantic Claim & Stance Extraction**: Automatically extract factual claims, opinions, reviews, and stances from diarized streamer audio:
   $$\text{Claim} = (\text{Entity/Subject}, \text{Attribute/Predicate}, \text{Value/Object}, \text{Stance}, \text{Polarity}, \text{Timestamp}, \text{VOD\_ID})$$
2. **Dedicated Local Vector DB + Relational Graph Storage**:
   - Store dense transcript chunk embeddings in ChromaDB / Qdrant.
   - Maintain relational metadata and claim graphs in SQLite / PostgreSQL.
3. **Temporal Additive Belief & Stance Evolution**:
   - Model beliefs as evolutionary, additive knowledge.
   - Example:
     * *Day 1*: Asmon reacts to trailer: "The special effects look terrible." $\to$ `[Godzilla, SpecialEffects, Bad, Stance: DISAPPROVAL]`
     * *Day 7*: Asmon watches film: "I actually really liked the Godzilla movie." $\to$ `[Godzilla, FilmOverall, Good, Stance: APPROVAL]`
     * *Synthesized Knowledge*: Asmon watched the movie; enjoyed the film overall, but disliked the visual effects.
4. **Natural Language VOD Querying**:
   - Semantic natural language search returning exact video timestamps, transcript snippets, on-screen context, and automatic 9:16 clip generation links.

---

## 2. Claim Extraction Engine

### 2.1 Extraction Schema (`StreamerClaim`)
```json
{
  "claim_id": "claim_08492",
  "creator_id": "asmongold",
  "vod_id": "asmon_vod_2026_09_15",
  "timestamp_sec": 412.5,
  "end_sec": 425.0,
  "topic_category": "ENTERTAINMENT_MOVIE",
  "subject_entity": "Godzilla Minus One",
  "predicate": "has_visual_effects",
  "object_value": "poor_special_effects",
  "stance": "DISAPPROVAL",
  "polarity": -0.75,
  "confidence": 0.91,
  "raw_quote": "The CGI on that monster looks completely cooked, it's not good.",
  "visual_context_summary": "Streamer watching official movie trailer on YouTube"
}
```

### 2.2 Additive Temporal Synthesis Engine
When multiple claims touch the same `(creator_id, subject_entity)` over time:
1. **Stance Aggregation Matrix**:
   $$S_{\text{aggregate}}(\text{Entity}) = \sum_{i=1}^M w(t_i) \cdot \text{Polarity}_i$$
   where $w(t_i) = e^{-\lambda (t_{\text{now}} - t_i)}$ is an optional temporal recency decay.
2. **Sub-Attribute Disambiguation**: Claims targeting specific sub-attributes (`special_effects`, `story`, `pacing`, `soundtrack`) do not overwrite the overarching entity verdict (`overall_movie`); they enrich the entity's hierarchical property graph.
3. **Contradiction & Flip-Flop Detection**: Flags when a creator reverses an overall stance on the same sub-attribute (e.g. from `POSITIVE` to `NEGATIVE` within 30 days), recording the before-and-after timestamps for analysis.

---

## 3. Storage Architecture: ChromaDB + SQLite/Postgres

```
┌─────────────────────────────────────────────────────────────┐
│                 Streamer Knowledge Layer                    │
├──────────────────────────────┬──────────────────────────────┤
│    ChromaDB / Qdrant         │    SQLite / PostgreSQL       │
│  - 768-d text-embedding-3    │  - StreamerClaim table       │
│  - Transcript chunks         │  - EntityGraph & Edges       │
│  - Semantic Nearest Neighbor │  - Temporal Timestamps & VODs│
│  - Fast semantic recall      │  - Exact metadata filtering  │
└──────────────────────────────┴──────────────────────────────┘
```

### 3.1 VOD Semantic Search Query Pipeline
1. User asks: *"What did Asmon say about the Godzilla movie?"*
2. System queries ChromaDB collection `streamer_transcripts` filtered by `creator_id == 'asmongold'`.
3. Relational join against `StreamerClaim` table retrieves sub-attribute breakdown (CGI vs overall).
4. System returns structured synthesis + direct jump links:
   * `00:06:52` - Trailer reaction (critiquing CGI).
   * `01:42:10` - Post-movie review (praising plot and ending).

---

## 4. Acceptance Criteria & Deliverables
* **Module**: `src/stream_fusion/knowledge/claims.py` (`ClaimExtractor`, `BeliefGraphEngine`, `StreamerKnowledgeStore`).
* **Vector Store Integration**: `src/stream_fusion/knowledge/vector_index.py` (ChromaDB / Qdrant adapter).
* **CLI Command**: `streamfusion query "Godzilla movie" --creator asmongold` and `streamfusion index-vod <vod_matrix_parquet>`.
* **Unit Tests**: `tests/test_claims_and_knowledge.py` validating claim parsing, additive stance resolution, and vector recall.
