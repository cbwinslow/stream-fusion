# Spec 28: Semantic Vector Search & Multimodal RAG Knowledge Engine

**Status**: In Progress / Target Complete  
**Implementation Modules**: `src/stream_fusion/knowledge/search.py`, `src/stream_fusion/web/routes/search.py`, `src/stream_fusion/models/schemas.py`, `src/stream_fusion/web/app.py`  
**Test Suite**: `tests/test_semantic_rag_engine.py`  
**Registry Schemas**: `SearchQueryRequest`, `SearchResultItem`, `SearchResponse`, `RagSynthesisRequest`, `RagSourceCitation`, `RagSynthesisResponse`, `IndexVodRequest`, `IndexVodResponse`, `VectorIndexStats` exported to `docs/schemas/`  

---

## 1. Executive Summary & Objective

Across Specs 01 through 27, StreamFusion has established a continuous 24/7 ingestion, multimodal comprehension, and homelab archiving pipeline. The system processes hundreds of hours of livestreams, producing:
- Whisper audio transcript segments with word-level alignments.
- Dense OCR keyframe text extractions and Florence-2 visual descriptions.
- Synchronized chat logs, emote bursts, sentiment distributions, and chatter profiles.
- Verified streamer knowledge graph claims with web-grounding citations and temporal stance shifts.

**Spec 28** introduces the **Semantic Vector Search & Multimodal RAG Knowledge Engine**: an interactive, cross-stream knowledge retrieval and conversational synthesis platform. It allows users, researchers, and automated clipping agents to search across hundreds of VODs using natural language, gamer slang, or player handles, and obtain synthesized answers with millisecond-exact video timestamp links (`/api/media/<vod_id>/video#t=142`) and synchronized chat reaction context.

```
+-------------------------------------------------------------------------------------------------------------+
|                                    StreamFusion Multimodal Search & RAG                                    |
|                                                                                                             |
|  Query: "What did Asmongold and Shroud say about Deadlock's matchmaking and recoil?"                        |
+-------------------------------------------------------------------------------------------------------------+
                                                     │
                                                     ▼
+-------------------------------------------------------------------------------------------------------------+
|                                        HybridSearchEngine (Dense + Sparse)                                  |
|                                                                                                             |
|   ┌─────────────────────────────────────────┐           ┌───────────────────────────────────────────────┐   |
|   │     Dense Vector Retrieval              │           │       Sparse BM25 Index                       │   |
|   │  • Deterministic subword/n-gram embedder│           │  • Okapi BM25 (k1=1.5, b=0.75)                │   |
|   │  • Cosine similarity matrix dot product │           │  • Gamer slang & handle preserving tokenizer  │   |
|   │  • Fallback to sentence-transformers    │           │  • Exact entity & term frequency scoring      │   |
|   └─────────────────────────────────────────┘           └───────────────────────────────────────────────┘   |
|                                                     │                                                       |
|                                                     ▼                                                       |
|   ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐   |
|   │                               Reciprocal Rank / Normalized Linear Fusion                            │   |
|   │                  Score = (1 - alpha) * SparseScore + alpha * DenseScore                             │   |
|   │                  Metadata Filters: streamer_id, vod_id, content_type, time_range                    │   |
|   └─────────────────────────────────────────────────────────────────────────────────────────────────────┘   |
+-------------------------------------------------------------------------------------------------------------+
                                                     │
                                                     ▼ Top-K Evidence Chunks
+-------------------------------------------------------------------------------------------------------------+
|                                       MultimodalRagSynthesizer Agent                                        |
|                                                                                                             |
|   • Cross-Stream Viewpoint Alignment: Compares stances between Asmongold and Shroud                         |
|   • Multimodal Context Injection: Correlates speech quotes with OCR screen text and chat burst reactions     |
|   • Exact Citation Generation: Video timestamp deep-links (/api/media/<vod>/video#t=142)                   |
|   • Temporal Synthesis: Identifies shifts in opinion over time                                              |
+-------------------------------------------------------------------------------------------------------------+
                                                     │
                                                     ▼
+-------------------------------------------------------------------------------------------------------------+
|                                          FastAPI Endpoints & Studio UI                                      |
|   • POST /api/search/semantic    • POST /api/search/rag    • POST /api/index/vod/{vod_id}                  |
|   • GET  /api/search/stats       • DELETE /api/index/vod/{vod_id}                                           |
+-------------------------------------------------------------------------------------------------------------+
```

---

## 2. Core Architectural Pillars

### 2.1 Local-First Vector Indexing & Embedding Engine
- **Pluggable & Zero-Dependency Embedded Fallback**: Supports local transformer models (`sentence-transformers`, `all-MiniLM-L6-v2`) if installed, with a built-in deterministic subword/n-gram hashing projection embedder (128-dim or 384-dim normalized vector) when external packages are unavailable.
- **Persistent Storage**: Backed by a high-performance SQLite + JSON vector store (`vector_index.db`) saving document text, normalized embeddings, timecodes, and metadata.
- **Fast Vector Math**: Vectorized NumPy cosine distance calculation with batch dot product matrix multiplication.

### 2.2 Sparse BM25 Search Engine
- **Okapi BM25**: Implements standard BM25 ranking ($k_1=1.5$, $b=0.75$) with inverse document frequency (IDF) weighting.
- **Slang-Preserving Tokenizer**: Keeps gamer slang (e.g. `KEKW`, `LULW`, `POG`), game titles (`Deadlock`, `WoW`), and punctuation-free handles intact.

### 2.3 Time-Windowed Multimodal Chunking
- Ingests VOD outputs from `homelab_storage/analyzed/<vod_id>/`:
  1. **Speech Chunks**: Sliding windows of whisper audio transcriptions with speaker voiceprint attribution.
  2. **Chat Chunks**: Aggregated chat density bursts, top emotes, and chatter sentiment.
  3. **OCR / Screen Chunks**: Dense keyframe text extractions and detected UI elements.
  4. **Claim Chunks**: Grounded knowledge claims, truthfulness verdicts, and topic labels.
- Deep-linked media references: Every chunk includes timestamp seconds and a player URL (`/api/media/<vod_id>/video#t=<sec>`).

### 2.4 Cross-Stream RAG Synthesizer
- Synthesizes comprehensive answers to multi-streamer and temporal queries:
  - Dissects multi-streamer agreements and disagreements.
  - Summarizes audience chat consensus (e.g., "Chat largely agreed with 72% positive sentiment").
  - Formats verifiable citations with exact quotes, time offsets, and video links.

### 2.5 REST API & Dashboard Studio Integration
- Seamlessly mounts `/api/search` into the FastAPI web app.
- Extends the Studio Web UI with an interactive Search & RAG Chat panel for immediate user testing and query verification.

---

## 3. Schema Definitions
New schemas registered under `src/stream_fusion/schema/registry.py`:
- `SearchQueryRequest`
- `SearchResultItem`
- `SearchResponse`
- `RagSynthesisRequest`
- `RagSourceCitation`
- `RagSynthesisResponse`
- `IndexVodRequest`
- `IndexVodResponse`
- `VectorIndexStats`

---

## 4. Verification Plan
- Unit tests for BM25 ranking, local dense embedder, hybrid scoring, and chunking.
- Integration tests for cross-stream RAG synthesis across multiple streamer personas.
- FastAPI endpoint tests for search, indexing, stats, and RAG routes.
- Full regression verification: 209 existing tests + new Spec 28 tests passing 100% green.
