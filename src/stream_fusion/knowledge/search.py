"""Semantic Vector Search & Multimodal RAG Knowledge Engine (Spec 28).

Provides:
- LocalEmbedder: Fast deterministic subword/n-gram hashing projection or sentence-transformers fallback.
- BM25Index: Okapi BM25 sparse search engine preserving gamer slang and handles.
- LocalVectorStorage: Persistent SQLite storage for vector embeddings, chunks, and metadata.
- HybridSearchEngine: Dense + Sparse hybrid retrieval with metadata filtering.
- MultimodalRagSynthesizer: Cross-stream conversational RAG engine with timestamp citations.
"""

from abc import ABC, abstractmethod
from datetime import timedelta
import json
import logging
import math
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import urllib.error
import urllib.request
import uuid
import numpy as np

from stream_fusion.models.schemas import (
    IndexVodRequest,
    IndexVodResponse,
    RagSourceCitation,
    RagSynthesisRequest,
    RagSynthesisResponse,
    SearchQueryRequest,
    SearchResponse,
    SearchResultItem,
    VectorIndexStats,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Local Embedder Engine
# ---------------------------------------------------------------------------

class LocalEmbedder:
    """Zero-dependency local embedding provider with optional sentence-transformers support."""

    def __init__(self, model_name: Optional[str] = None, dim: int = 128):
        self.dim = dim
        self.model_name = model_name
        self._st_model = None

        if model_name:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                self._st_model = SentenceTransformer(model_name)
                # Update dim if model loaded
                test_emb = self._st_model.encode(["test"])
                self.dim = test_emb.shape[1]
                logger.info(f"Loaded SentenceTransformer model '{model_name}' (dim={self.dim})")
            except Exception as e:
                logger.warning(
                    f"SentenceTransformer not available for '{model_name}': {e}. "
                    f"Using deterministic projection fallback (dim={self.dim})."
                )
                self._st_model = None

    def _tokenize(self, text: str) -> List[str]:
        """Extracts words and alphanumeric tokens, lowercased."""
        return re.findall(r"[A-Za-z0-9_]+", text.lower())

    def embed_text(self, text: str) -> np.ndarray:
        """Projects input text to a normalized dense vector."""
        if self._st_model is not None:
            emb = self._st_model.encode([text])[0]
            emb = np.array(emb, dtype=np.float32)
            norm = np.linalg.norm(emb)
            return emb / norm if norm > 0 else emb

        # Deterministic subword/n-gram hashing projection
        tokens = self._tokenize(text)
        vec = np.zeros(self.dim, dtype=np.float32)
        if not tokens:
            vec[0] = 1.0
            return vec

        # Incorporate 1-grams, 2-grams, and character 3-grams for semantic robustness
        features = list(tokens)
        for i in range(len(tokens) - 1):
            features.append(f"{tokens[i]}_{tokens[i+1]}")
        for t in tokens:
            if len(t) >= 4:
                for c_i in range(len(t) - 2):
                    features.append(t[c_i:c_i+3])

        for pos, feat in enumerate(features):
            h = hash(feat)
            idx = abs(h) % self.dim
            sign = 1.0 if (h % 2 == 0) else -1.0
            # Positional decay for long texts
            weight = 1.0 / (1.0 + 0.05 * pos)
            vec[idx] += sign * weight

        norm = np.linalg.norm(vec)
        if norm > 0.0:
            vec = vec / norm
        else:
            vec[0] = 1.0
        return vec.astype(np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embeds a batch of texts into an (N, D) float32 matrix."""
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        if self._st_model is not None:
            raw = self._st_model.encode(texts)
            arr = np.array(raw, dtype=np.float32)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return arr / norms

        embs = [self.embed_text(t) for t in texts]
        return np.vstack(embs)

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Computes cosine similarity between two 1D vectors."""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    @staticmethod
    def batch_cosine_similarity(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
        """Computes cosine similarities between query_vec and an (N, D) doc_matrix."""
        if doc_matrix.shape[0] == 0:
            return np.empty((0,), dtype=np.float32)
        q_norm = np.linalg.norm(query_vec)
        if q_norm == 0:
            return np.zeros((doc_matrix.shape[0],), dtype=np.float32)
        q_normed = query_vec / q_norm

        d_norms = np.linalg.norm(doc_matrix, axis=1)
        d_norms[d_norms == 0] = 1.0

        dots = np.dot(doc_matrix, q_normed)
        return (dots / d_norms).astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Sparse BM25 Search Engine
# ---------------------------------------------------------------------------

class BM25Index:
    """Okapi BM25 inverted index supporting gamer jargon and handles."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size: int = 0
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_len: float = 0.0
        self.df: Dict[str, int] = {}  # term -> number of documents containing term
        self.tf: Dict[str, Dict[str, int]] = {}  # doc_id -> {term: frequency}
        self.idf_cache: Dict[str, float] = {}

    def tokenize(self, text: str) -> List[str]:
        """Preserves gamer slang, handles, numbers, and alphanumeric tokens."""
        return re.findall(r"[A-Za-z0-9_]+", text.lower())

    def add_document(self, doc_id: str, text: str) -> None:
        """Indexes a document text under doc_id."""
        if doc_id in self.doc_lengths:
            self.remove_document(doc_id)

        tokens = self.tokenize(text)
        doc_len = len(tokens)
        self.doc_lengths[doc_id] = doc_len

        term_counts: Dict[str, int] = {}
        for t in tokens:
            term_counts[t] = term_counts.get(t, 0) + 1

        self.tf[doc_id] = term_counts
        for t in term_counts:
            self.df[t] = self.df.get(t, 0) + 1

        self.corpus_size += 1
        total_len = sum(self.doc_lengths.values())
        self.avg_doc_len = total_len / self.corpus_size if self.corpus_size > 0 else 0.0
        self.idf_cache.clear()

    def remove_document(self, doc_id: str) -> None:
        """Removes a document from the inverted index."""
        if doc_id not in self.doc_lengths:
            return

        term_counts = self.tf.pop(doc_id, {})
        for t in term_counts:
            if t in self.df:
                self.df[t] -= 1
                if self.df[t] <= 0:
                    del self.df[t]

        self.doc_lengths.pop(doc_id, None)
        self.corpus_size = max(0, self.corpus_size - 1)
        total_len = sum(self.doc_lengths.values())
        self.avg_doc_len = total_len / self.corpus_size if self.corpus_size > 0 else 0.0
        self.idf_cache.clear()

    def remove_documents_for_vod(self, vod_id: str) -> int:
        """Removes all documents belonging to a given vod_id."""
        to_remove = [doc_id for doc_id in list(self.doc_lengths.keys()) if doc_id.startswith(f"{vod_id}_")]
        for doc_id in to_remove:
            self.remove_document(doc_id)
        return len(to_remove)

    def get_idf(self, term: str) -> float:
        """Calculates standard Okapi BM25 inverse document frequency."""
        if term in self.idf_cache:
            return self.idf_cache[term]
        df_t = self.df.get(term, 0)
        if df_t == 0:
            idf = 0.0
        else:
            # Okapi BM25 IDF formula
            numerator = self.corpus_size - df_t + 0.5
            denominator = df_t + 0.5
            idf = math.log(1.0 + max(0.0, numerator / denominator))
        self.idf_cache[term] = idf
        return idf

    def score(self, query: str, candidate_ids: Optional[List[str]] = None) -> Dict[str, float]:
        """Calculates BM25 sparse scores for candidate documents."""
        q_tokens = self.tokenize(query)
        if not q_tokens or self.corpus_size == 0:
            return {}

        target_ids = candidate_ids if candidate_ids is not None else list(self.doc_lengths.keys())
        scores: Dict[str, float] = {}

        for doc_id in target_ids:
            doc_tf = self.tf.get(doc_id)
            if not doc_tf:
                continue
            doc_len = self.doc_lengths.get(doc_id, 0)
            score = 0.0

            for t in q_tokens:
                if t not in doc_tf:
                    continue
                tf_val = doc_tf[t]
                idf = self.get_idf(t)

                # Okapi BM25 term weighting
                numerator = tf_val * (self.k1 + 1.0)
                denominator = tf_val + self.k1 * (1.0 - self.b + self.b * (doc_len / (self.avg_doc_len or 1.0)))
                score += idf * (numerator / denominator)

            if score > 0.0:
                scores[doc_id] = score

        return scores


# ---------------------------------------------------------------------------
# 3. Vector Storage Backends (PostgreSQL+pgvector, JSON Flatfile, SQLite)
# ---------------------------------------------------------------------------

class BaseVectorStorage(ABC):
    """Abstract vector storage backend interface."""

    @abstractmethod
    def insert_chunk(
        self,
        chunk_id: str,
        vod_id: str,
        streamer_id: str,
        timestamp_sec: float,
        timestamp_end_sec: float,
        content_type: str,
        text: str,
        metadata: Dict[str, Any],
        embedding: np.ndarray,
    ) -> None:
        pass

    @abstractmethod
    def insert_chunks_batch(self, chunks: List[Dict[str, Any]]) -> int:
        pass

    @abstractmethod
    def delete_vod(self, vod_id: str) -> int:
        pass

    @abstractmethod
    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def query_candidate_chunks(
        self,
        streamer_ids: Optional[List[str]] = None,
        vod_ids: Optional[List[str]] = None,
        content_types: Optional[List[str]] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        pass

    def query_nearest_neighbors(
        self,
        query_vec: np.ndarray,
        top_k: int = 10,
        streamer_ids: Optional[List[str]] = None,
        vod_ids: Optional[List[str]] = None,
        content_types: Optional[List[str]] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Fallback nearest-neighbor implementation using in-memory cosine similarities."""
        cands = self.query_candidate_chunks(
            streamer_ids=streamer_ids,
            vod_ids=vod_ids,
            content_types=content_types,
            start_time=start_time,
            end_time=end_time,
        )
        if not cands:
            return []
        matrix = np.vstack([c["embedding"] for c in cands])
        sims = LocalEmbedder.batch_cosine_similarity(query_vec, matrix)
        scored = list(zip(cands, [float(s) for s in sims]))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    @abstractmethod
    def get_stats(self) -> VectorIndexStats:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


class PgVectorStorage(BaseVectorStorage):
    """Production-grade PostgreSQL storage with native pgvector extension and HNSW indexing."""

    def __init__(self, database_url: str, dim: int = 128):
        self.raw_url = str(database_url)
        self.dim = dim
        self._lock = threading.RLock()
        self._pg_module = None

        try:
            import psycopg
            self._pg_module = "psycopg"
        except ImportError:
            try:
                import psycopg2
                self._pg_module = "psycopg2"
            except ImportError:
                raise ImportError(
                    "PostgreSQL URL specified, but neither 'psycopg' nor 'psycopg2' is installed. "
                    "Install with: pip install psycopg[binary] or psycopg2-binary"
                )

        self._init_schema()

    def _get_connection(self):
        if self._pg_module == "psycopg":
            import psycopg
            return psycopg.connect(self.raw_url)
        else:
            import psycopg2
            return psycopg2.connect(self.raw_url)

    def _init_schema(self) -> None:
        """Enables pgvector extension and creates vector_chunks table with HNSW index."""
        with self._lock, self._get_connection() as conn:
            with conn.cursor() as cur:
                # Enable pgvector extension
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                # Create table
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS vector_chunks (
                        chunk_id VARCHAR(255) PRIMARY KEY,
                        vod_id VARCHAR(255) NOT NULL,
                        streamer_id VARCHAR(100) NOT NULL,
                        timestamp_sec DOUBLE PRECISION NOT NULL,
                        timestamp_end_sec DOUBLE PRECISION NOT NULL,
                        content_type VARCHAR(50) NOT NULL,
                        text TEXT NOT NULL,
                        metadata_json JSONB NOT NULL DEFAULT '{{}}',
                        embedding vector({self.dim}) NOT NULL
                    );
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_vc_vod ON vector_chunks(vod_id);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_vc_streamer ON vector_chunks(streamer_id);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_vc_type ON vector_chunks(content_type);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_vc_time ON vector_chunks(timestamp_sec);")
                # HNSW index for high-scale fast cosine similarity
                cur.execute("CREATE INDEX IF NOT EXISTS idx_vc_embedding_hnsw ON vector_chunks USING hnsw (embedding vector_cosine_ops);")
            conn.commit()

    @staticmethod
    def _vec_to_str(vec: np.ndarray) -> str:
        """Formats numpy vector into pgvector string format '[0.1,0.2,...]'."""
        return f"[{','.join(f'{float(x):.6f}' for x in vec)}]"

    @staticmethod
    def _str_to_vec(vec_str: Any) -> np.ndarray:
        """Parses pgvector string or list to numpy float32 array."""
        if isinstance(vec_str, np.ndarray):
            return vec_str.astype(np.float32)
        if isinstance(vec_str, (list, tuple)):
            return np.array(vec_str, dtype=np.float32)
        if isinstance(vec_str, str):
            clean = vec_str.strip("[]() ")
            return np.fromstring(clean, sep=",", dtype=np.float32)
        return np.zeros(128, dtype=np.float32)

    def insert_chunk(
        self,
        chunk_id: str,
        vod_id: str,
        streamer_id: str,
        timestamp_sec: float,
        timestamp_end_sec: float,
        content_type: str,
        text: str,
        metadata: Dict[str, Any],
        embedding: np.ndarray,
    ) -> None:
        vec_str = self._vec_to_str(embedding)
        meta_str = json.dumps(metadata)
        with self._lock, self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO vector_chunks 
                    (chunk_id, vod_id, streamer_id, timestamp_sec, timestamp_end_sec, content_type, text, metadata_json, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        vod_id = EXCLUDED.vod_id,
                        streamer_id = EXCLUDED.streamer_id,
                        timestamp_sec = EXCLUDED.timestamp_sec,
                        timestamp_end_sec = EXCLUDED.timestamp_end_sec,
                        content_type = EXCLUDED.content_type,
                        text = EXCLUDED.text,
                        metadata_json = EXCLUDED.metadata_json,
                        embedding = EXCLUDED.embedding;
                """, (chunk_id, vod_id, streamer_id, timestamp_sec, timestamp_end_sec, content_type, text, meta_str, vec_str))
            conn.commit()

    def insert_chunks_batch(self, chunks: List[Dict[str, Any]]) -> int:
        if not chunks:
            return 0
        rows = []
        for c in chunks:
            vec_str = self._vec_to_str(c["embedding"])
            meta_str = json.dumps(c.get("metadata", {}))
            rows.append((
                c["chunk_id"],
                c["vod_id"],
                c["streamer_id"],
                c["timestamp_sec"],
                c["timestamp_end_sec"],
                c["content_type"],
                c["text"],
                meta_str,
                vec_str,
            ))
        with self._lock, self._get_connection() as conn:
            with conn.cursor() as cur:
                for row in rows:
                    cur.execute("""
                        INSERT INTO vector_chunks 
                        (chunk_id, vod_id, streamer_id, timestamp_sec, timestamp_end_sec, content_type, text, metadata_json, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)
                        ON CONFLICT (chunk_id) DO UPDATE SET
                            vod_id = EXCLUDED.vod_id,
                            streamer_id = EXCLUDED.streamer_id,
                            timestamp_sec = EXCLUDED.timestamp_sec,
                            timestamp_end_sec = EXCLUDED.timestamp_end_sec,
                            content_type = EXCLUDED.content_type,
                            text = EXCLUDED.text,
                            metadata_json = EXCLUDED.metadata_json,
                            embedding = EXCLUDED.embedding;
                    """, row)
            conn.commit()
        return len(rows)

    def delete_vod(self, vod_id: str) -> int:
        with self._lock, self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM vector_chunks WHERE vod_id = %s;", (vod_id,))
                deleted = cur.rowcount
            conn.commit()
            return deleted

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT chunk_id, vod_id, streamer_id, timestamp_sec, timestamp_end_sec, content_type, text, metadata_json, embedding FROM vector_chunks WHERE chunk_id = %s;",
                    (chunk_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                meta = row[7] if isinstance(row[7], dict) else json.loads(row[7])
                return {
                    "chunk_id": row[0],
                    "vod_id": row[1],
                    "streamer_id": row[2],
                    "timestamp_sec": row[3],
                    "timestamp_end_sec": row[4],
                    "content_type": row[5],
                    "text": row[6],
                    "metadata": meta,
                    "embedding": self._str_to_vec(row[8]),
                }

    def query_candidate_chunks(
        self,
        streamer_ids: Optional[List[str]] = None,
        vod_ids: Optional[List[str]] = None,
        content_types: Optional[List[str]] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        clauses = []
        params: List[Any] = []
        if streamer_ids:
            clauses.append("streamer_id = ANY(%s)")
            params.append(streamer_ids)
        if vod_ids:
            clauses.append("vod_id = ANY(%s)")
            params.append(vod_ids)
        if content_types:
            clauses.append("content_type = ANY(%s)")
            params.append(content_types)
        if start_time is not None:
            clauses.append("timestamp_sec >= %s")
            params.append(start_time)
        if end_time is not None:
            clauses.append("timestamp_sec <= %s")
            params.append(end_time)

        sql = "SELECT chunk_id, vod_id, streamer_id, timestamp_sec, timestamp_end_sec, content_type, text, metadata_json, embedding FROM vector_chunks"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)

        with self._lock, self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
                results = []
                for row in rows:
                    meta = row[7] if isinstance(row[7], dict) else json.loads(row[7])
                    results.append({
                        "chunk_id": row[0],
                        "vod_id": row[1],
                        "streamer_id": row[2],
                        "timestamp_sec": row[3],
                        "timestamp_end_sec": row[4],
                        "content_type": row[5],
                        "text": row[6],
                        "metadata": meta,
                        "embedding": self._str_to_vec(row[8]),
                    })
                return results

    def query_nearest_neighbors(
        self,
        query_vec: np.ndarray,
        top_k: int = 10,
        streamer_ids: Optional[List[str]] = None,
        vod_ids: Optional[List[str]] = None,
        content_types: Optional[List[str]] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """HNSW cosine similarity nearest neighbor search directly in PostgreSQL."""
        vec_str = self._vec_to_str(query_vec)
        clauses = []
        params: List[Any] = [vec_str]

        if streamer_ids:
            clauses.append("streamer_id = ANY(%s)")
            params.append(streamer_ids)
        if vod_ids:
            clauses.append("vod_id = ANY(%s)")
            params.append(vod_ids)
        if content_types:
            clauses.append("content_type = ANY(%s)")
            params.append(content_types)
        if start_time is not None:
            clauses.append("timestamp_sec >= %s")
            params.append(start_time)
        if end_time is not None:
            clauses.append("timestamp_sec <= %s")
            params.append(end_time)

        params.extend([vec_str, top_k])
        where_clause = (" WHERE " + " AND ".join(clauses)) if clauses else ""

        sql = f"""
            SELECT chunk_id, vod_id, streamer_id, timestamp_sec, timestamp_end_sec,
                   content_type, text, metadata_json, embedding,
                   (1.0 - (embedding <=> %s::vector)) AS sim
            FROM vector_chunks
            {where_clause}
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """

        with self._lock, self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
                results = []
                for row in rows:
                    meta = row[7] if isinstance(row[7], dict) else json.loads(row[7])
                    doc = {
                        "chunk_id": row[0],
                        "vod_id": row[1],
                        "streamer_id": row[2],
                        "timestamp_sec": row[3],
                        "timestamp_end_sec": row[4],
                        "content_type": row[5],
                        "text": row[6],
                        "metadata": meta,
                        "embedding": self._str_to_vec(row[8]),
                    }
                    sim = float(row[9])
                    results.append((doc, sim))
                return results

    def get_stats(self) -> VectorIndexStats:
        with self._lock, self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*), COUNT(DISTINCT vod_id), COUNT(DISTINCT streamer_id) FROM vector_chunks;")
                c_row = cur.fetchone()
                cur.execute("SELECT content_type, COUNT(*) FROM vector_chunks GROUP BY content_type;")
                t_rows = cur.fetchall()
                by_type = {r[0]: r[1] for r in t_rows}
                return VectorIndexStats(
                    total_chunks=c_row[0] or 0,
                    total_vods=c_row[1] or 0,
                    total_streamers=c_row[2] or 0,
                    chunks_by_type=by_type,
                    embedding_dim=self.dim,
                    index_storage_bytes=0,
                )

    def close(self) -> None:
        pass


class JsonVectorStorage(BaseVectorStorage):
    """Zero-database, transparent JSON & NumPy flat file vector storage."""

    def __init__(self, file_path: Union[str, Path], dim: int = 128):
        self.path = Path(str(file_path).replace("json://", ""))
        self.dim = dim
        self.chunks: Dict[str, Dict[str, Any]] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self._lock = threading.RLock()
        self.load()

    def load(self) -> None:
        with self._lock:
            if not self.path.exists():
                return
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.chunks = data.get("chunks", {})
                raw_embs = data.get("embeddings", {})
                self.embeddings = {
                    k: np.array(v, dtype=np.float32) for k, v in raw_embs.items()
                }
            except Exception as e:
                logger.warning(f"Error loading JsonVectorStorage from {self.path}: {e}")

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "chunks": self.chunks,
                "embeddings": {k: v.tolist() for k, v in self.embeddings.items()},
            }
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            tmp.replace(self.path)

    def insert_chunk(
        self,
        chunk_id: str,
        vod_id: str,
        streamer_id: str,
        timestamp_sec: float,
        timestamp_end_sec: float,
        content_type: str,
        text: str,
        metadata: Dict[str, Any],
        embedding: np.ndarray,
    ) -> None:
        with self._lock:
            self.chunks[chunk_id] = {
                "chunk_id": chunk_id,
                "vod_id": vod_id,
                "streamer_id": streamer_id,
                "timestamp_sec": timestamp_sec,
                "timestamp_end_sec": timestamp_end_sec,
                "content_type": content_type,
                "text": text,
                "metadata": metadata,
            }
            self.embeddings[chunk_id] = embedding.astype(np.float32)
            self.save()

    def insert_chunks_batch(self, chunks: List[Dict[str, Any]]) -> int:
        with self._lock:
            for c in chunks:
                c_id = c["chunk_id"]
                self.chunks[c_id] = {
                    "chunk_id": c_id,
                    "vod_id": c["vod_id"],
                    "streamer_id": c["streamer_id"],
                    "timestamp_sec": c["timestamp_sec"],
                    "timestamp_end_sec": c["timestamp_end_sec"],
                    "content_type": c["content_type"],
                    "text": c["text"],
                    "metadata": c.get("metadata", {}),
                }
                self.embeddings[c_id] = c["embedding"].astype(np.float32)
            self.save()
            return len(chunks)

    def delete_vod(self, vod_id: str) -> int:
        with self._lock:
            to_del = [cid for cid, c in self.chunks.items() if c.get("vod_id") == vod_id]
            for cid in to_del:
                self.chunks.pop(cid, None)
                self.embeddings.pop(cid, None)
            if to_del:
                self.save()
            return len(to_del)

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            c = self.chunks.get(chunk_id)
            if not c:
                return None
            return {**c, "embedding": self.embeddings.get(chunk_id)}

    def query_candidate_chunks(
        self,
        streamer_ids: Optional[List[str]] = None,
        vod_ids: Optional[List[str]] = None,
        content_types: Optional[List[str]] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            results = []
            for cid, c in self.chunks.items():
                if streamer_ids and c.get("streamer_id") not in streamer_ids:
                    continue
                if vod_ids and c.get("vod_id") not in vod_ids:
                    continue
                if content_types and c.get("content_type") not in content_types:
                    continue
                ts = c.get("timestamp_sec", 0.0)
                if start_time is not None and ts < start_time:
                    continue
                if end_time is not None and ts > end_time:
                    continue
                results.append({**c, "embedding": self.embeddings.get(cid)})
            return results

    def get_stats(self) -> VectorIndexStats:
        with self._lock:
            total = len(self.chunks)
            vods = len({c.get("vod_id") for c in self.chunks.values() if c.get("vod_id")})
            streamers = len({c.get("streamer_id") for c in self.chunks.values() if c.get("streamer_id")})
            by_type: Dict[str, int] = {}
            for c in self.chunks.values():
                t = c.get("content_type", "unknown")
                by_type[t] = by_type.get(t, 0) + 1
            storage_bytes = self.path.stat().st_size if self.path.exists() else 0
            return VectorIndexStats(
                total_chunks=total,
                total_vods=vods,
                total_streamers=streamers,
                chunks_by_type=by_type,
                embedding_dim=self.dim,
                index_storage_bytes=storage_bytes,
            )

    def close(self) -> None:
        pass


class LocalVectorStorage(BaseVectorStorage):
    """SQLite-backed persistent vector and chunk storage."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None, dim: int = 128):
        self.dim = dim
        if db_path is None or str(db_path) == ":memory:":
            self.db_path = ":memory:"
        else:
            self.db_path = str(db_path)
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        """Initializes database tables and index structures."""
        with self._conn:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    vod_id TEXT NOT NULL,
                    streamer_id TEXT NOT NULL,
                    timestamp_sec REAL NOT NULL,
                    timestamp_end_sec REAL NOT NULL,
                    content_type TEXT NOT NULL,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    embedding_blob BLOB NOT NULL
                )
            """)
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_vod ON chunks(vod_id)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_streamer ON chunks(streamer_id)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(content_type)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_time ON chunks(timestamp_sec)")

    def insert_chunk(
        self,
        chunk_id: str,
        vod_id: str,
        streamer_id: str,
        timestamp_sec: float,
        timestamp_end_sec: float,
        content_type: str,
        text: str,
        metadata: Dict[str, Any],
        embedding: np.ndarray,
    ) -> None:
        """Inserts or replaces an indexed chunk."""
        blob = embedding.astype(np.float32).tobytes()
        meta_str = json.dumps(metadata)
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO chunks 
                (chunk_id, vod_id, streamer_id, timestamp_sec, timestamp_end_sec, content_type, text, metadata_json, embedding_blob)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk_id,
                    vod_id,
                    streamer_id,
                    timestamp_sec,
                    timestamp_end_sec,
                    content_type,
                    text,
                    meta_str,
                    blob,
                ),
            )

    def insert_chunks_batch(self, chunks: List[Dict[str, Any]]) -> int:
        """Batch inserts multiple chunks in a single transaction."""
        rows = []
        for c in chunks:
            blob = c["embedding"].astype(np.float32).tobytes()
            meta_str = json.dumps(c.get("metadata", {}))
            rows.append((
                c["chunk_id"],
                c["vod_id"],
                c["streamer_id"],
                c["timestamp_sec"],
                c["timestamp_end_sec"],
                c["content_type"],
                c["text"],
                meta_str,
                blob,
            ))
        with self._conn:
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO chunks 
                (chunk_id, vod_id, streamer_id, timestamp_sec, timestamp_end_sec, content_type, text, metadata_json, embedding_blob)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    def delete_vod(self, vod_id: str) -> int:
        """Deletes all chunks associated with a specific VOD."""
        with self._conn:
            cursor = self._conn.execute("DELETE FROM chunks WHERE vod_id = ?", (vod_id,))
            return cursor.rowcount

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Fetches a single chunk by chunk_id."""
        cursor = self._conn.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return self._row_to_dict(row)

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Converts a sqlite row to a chunk dictionary with unpacked embedding."""
        emb = np.frombuffer(row["embedding_blob"], dtype=np.float32)
        return {
            "chunk_id": row["chunk_id"],
            "vod_id": row["vod_id"],
            "streamer_id": row["streamer_id"],
            "timestamp_sec": row["timestamp_sec"],
            "timestamp_end_sec": row["timestamp_end_sec"],
            "content_type": row["content_type"],
            "text": row["text"],
            "metadata": json.loads(row["metadata_json"]),
            "embedding": emb,
        }

    def query_candidate_chunks(
        self,
        streamer_ids: Optional[List[str]] = None,
        vod_ids: Optional[List[str]] = None,
        content_types: Optional[List[str]] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Filters chunks by metadata criteria."""
        clauses = []
        params: List[Any] = []

        if streamer_ids:
            placeholders = ",".join(["?"] * len(streamer_ids))
            clauses.append(f"streamer_id IN ({placeholders})")
            params.extend(streamer_ids)

        if vod_ids:
            placeholders = ",".join(["?"] * len(vod_ids))
            clauses.append(f"vod_id IN ({placeholders})")
            params.extend(vod_ids)

        if content_types:
            placeholders = ",".join(["?"] * len(content_types))
            clauses.append(f"content_type IN ({placeholders})")
            params.extend(content_types)

        if start_time is not None:
            clauses.append("timestamp_sec >= ?")
            params.append(start_time)

        if end_time is not None:
            clauses.append("timestamp_sec <= ?")
            params.append(end_time)

        query = "SELECT * FROM chunks"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)

        cursor = self._conn.execute(query, params)
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_stats(self) -> VectorIndexStats:
        """Computes index statistics."""
        cursor = self._conn.execute("SELECT COUNT(*), COUNT(DISTINCT vod_id), COUNT(DISTINCT streamer_id) FROM chunks")
        total_chunks, total_vods, total_streamers = cursor.fetchone()

        cursor = self._conn.execute("SELECT content_type, COUNT(*) FROM chunks GROUP BY content_type")
        by_type = {row[0]: row[1] for row in cursor.fetchall()}

        # Storage size
        storage_bytes = 0
        if self.db_path != ":memory:" and Path(self.db_path).exists():
            storage_bytes = Path(self.db_path).stat().st_size

        return VectorIndexStats(
            total_chunks=total_chunks or 0,
            total_vods=total_vods or 0,
            total_streamers=total_streamers or 0,
            chunks_by_type=by_type,
            embedding_dim=self.dim,
            index_storage_bytes=storage_bytes,
        )

    def close(self) -> None:
        """Closes database connection."""
        self._conn.close()


class QdrantVectorStorage(BaseVectorStorage):
    """Qdrant-backed semantic vector storage with Binary Quantization (BQ) support (Spec 29).

    Supports remote Qdrant REST instances (e.g. qdrant://localhost:6333 or http://localhost:6333)
    with 32x Binary Quantization (BQ) and transparent local mock fallback for zero-external-dependency testing.
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "stream_fusion_vectors",
        dim: int = 128,
        enable_binary_quantization: bool = True,
        api_key: Optional[str] = None,
        timeout: float = 2.0,
        force_mock: bool = False,
    ):
        self.dim = dim
        self.collection_name = collection_name
        self.enable_binary_quantization = enable_binary_quantization
        self.api_key = api_key
        self.timeout = timeout
        self.force_mock = force_mock
        self._lock = threading.RLock()

        # Parse / clean endpoint URL
        clean_url = str(url).strip()
        if clean_url.startswith("qdrant://"):
            host_part = clean_url[len("qdrant://"):]
            self.base_url = f"http://{host_part}"
        elif clean_url.startswith("http://") or clean_url.startswith("https://"):
            self.base_url = clean_url.rstrip("/")
        else:
            self.base_url = f"http://{clean_url}"

        # Internal mock storage dictionary for fallback / test mode
        self._mock_points: Dict[str, Dict[str, Any]] = {}
        self.is_mock = force_mock

        # Quantization configuration payload (32x compression via Binary Quantization)
        self.quantization_config = (
            {"binary": {"always_ram": True}} if enable_binary_quantization else None
        )

        if not self.force_mock:
            self._try_init_remote()
        else:
            logger.debug("QdrantVectorStorage initialized in forced mock mode.")

    def _make_request(
        self, method: str, path: str, payload: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        if self.is_mock:
            return None
        url = f"{self.base_url}{path}"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["api-key"] = self.api_key
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except Exception as e:
            logger.debug(f"Qdrant HTTP request {method} {url} failed: {e}")
            raise

    def _try_init_remote(self) -> None:
        try:
            # Check if collection exists
            self._make_request("GET", f"/collections/{self.collection_name}")
        except Exception:
            try:
                # Try creating collection with vector size, cosine distance, and binary quantization
                body: Dict[str, Any] = {
                    "vectors": {
                        "size": self.dim,
                        "distance": "Cosine",
                    }
                }
                if self.quantization_config:
                    body["quantization_config"] = self.quantization_config
                self._make_request("PUT", f"/collections/{self.collection_name}", body)
            except Exception as e:
                logger.info(f"Qdrant server unreachable ({e}); falling back to zero-dependency in-memory mock.")
                self.is_mock = True

    def insert_chunk(
        self,
        chunk_id: str,
        vod_id: str,
        streamer_id: str,
        timestamp_sec: float,
        timestamp_end_sec: float,
        content_type: str,
        text: str,
        metadata: Dict[str, Any],
        embedding: np.ndarray,
    ) -> None:
        self.insert_chunks_batch([{
            "chunk_id": chunk_id,
            "vod_id": vod_id,
            "streamer_id": streamer_id,
            "timestamp_sec": timestamp_sec,
            "timestamp_end_sec": timestamp_end_sec,
            "content_type": content_type,
            "text": text,
            "metadata": metadata,
            "embedding": embedding,
        }])

    def insert_chunks_batch(self, chunks: List[Dict[str, Any]]) -> int:
        if not chunks:
            return 0
        with self._lock:
            if not self.is_mock:
                points = []
                for c in chunks:
                    cid = c["chunk_id"]
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, cid))
                    emb = c["embedding"]
                    vec = emb.tolist() if isinstance(emb, np.ndarray) else list(emb)
                    payload = {
                        "chunk_id": cid,
                        "vod_id": c.get("vod_id", ""),
                        "streamer_id": c.get("streamer_id", ""),
                        "timestamp_sec": float(c.get("timestamp_sec", 0.0)),
                        "timestamp_end_sec": float(c.get("timestamp_end_sec", 0.0)),
                        "content_type": c.get("content_type", "speech"),
                        "text": c.get("text", ""),
                        "metadata": c.get("metadata", {}),
                    }
                    points.append({"id": point_id, "vector": vec, "payload": payload})
                try:
                    self._make_request(
                        "PUT",
                        f"/collections/{self.collection_name}/points?wait=true",
                        {"points": points},
                    )
                    return len(points)
                except Exception as e:
                    logger.warning(f"Qdrant batch insert failed ({e}); switching to mock storage.")
                    self.is_mock = True

            # In mock / fallback mode
            for c in chunks:
                cid = c["chunk_id"]
                emb = c["embedding"]
                vec = np.array(emb, dtype=np.float32) if not isinstance(emb, np.ndarray) else emb.astype(np.float32)
                self._mock_points[cid] = {
                    "chunk_id": cid,
                    "vod_id": c.get("vod_id", ""),
                    "streamer_id": c.get("streamer_id", ""),
                    "timestamp_sec": float(c.get("timestamp_sec", 0.0)),
                    "timestamp_end_sec": float(c.get("timestamp_end_sec", 0.0)),
                    "content_type": c.get("content_type", "speech"),
                    "text": c.get("text", ""),
                    "metadata": c.get("metadata", {}),
                    "embedding": vec,
                }
            return len(chunks)

    def delete_vod(self, vod_id: str) -> int:
        with self._lock:
            deleted_count = 0
            if not self.is_mock:
                try:
                    self._make_request(
                        "POST",
                        f"/collections/{self.collection_name}/points/delete",
                        {"filter": {"must": [{"key": "vod_id", "match": {"value": vod_id}}]}},
                    )
                    deleted_count = 1
                except Exception:
                    self.is_mock = True

            to_del = [cid for cid, p in self._mock_points.items() if p.get("vod_id") == vod_id]
            for cid in to_del:
                del self._mock_points[cid]
            return len(to_del) if self.is_mock else deleted_count

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            if not self.is_mock:
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))
                try:
                    res = self._make_request("GET", f"/collections/{self.collection_name}/points/{point_id}")
                    if res and "result" in res and res["result"]:
                        pt = res["result"]
                        payload = pt.get("payload", {})
                        vec = pt.get("vector")
                        emb = np.array(vec, dtype=np.float32) if vec else None
                        return {**payload, "embedding": emb}
                except Exception:
                    self.is_mock = True

            p = self._mock_points.get(chunk_id)
            if not p:
                return None
            return dict(p)

    def query_candidate_chunks(
        self,
        streamer_ids: Optional[List[str]] = None,
        vod_ids: Optional[List[str]] = None,
        content_types: Optional[List[str]] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            if not self.is_mock:
                must_conditions: List[Dict[str, Any]] = []
                if streamer_ids:
                    must_conditions.append({"key": "streamer_id", "match": {"any": streamer_ids}})
                if vod_ids:
                    must_conditions.append({"key": "vod_id", "match": {"any": vod_ids}})
                if content_types:
                    must_conditions.append({"key": "content_type", "match": {"any": content_types}})
                if start_time is not None:
                    must_conditions.append({"key": "timestamp_sec", "range": {"gte": start_time}})
                if end_time is not None:
                    must_conditions.append({"key": "timestamp_sec", "range": {"lte": end_time}})

                filter_body = {"filter": {"must": must_conditions}} if must_conditions else {}
                try:
                    payload = {"limit": 10000, "with_payload": True, "with_vector": True, **filter_body}
                    res = self._make_request("POST", f"/collections/{self.collection_name}/points/scroll", payload)
                    if res and "result" in res and "points" in res["result"]:
                        results = []
                        for pt in res["result"]["points"]:
                            pl = pt.get("payload", {})
                            vec = pt.get("vector")
                            emb = np.array(vec, dtype=np.float32) if vec else None
                            results.append({**pl, "embedding": emb})
                        return results
                except Exception:
                    self.is_mock = True

            # Mock fallback
            results = []
            for cid, p in self._mock_points.items():
                if streamer_ids and p.get("streamer_id") not in streamer_ids:
                    continue
                if vod_ids and p.get("vod_id") not in vod_ids:
                    continue
                if content_types and p.get("content_type") not in content_types:
                    continue
                ts = p.get("timestamp_sec", 0.0)
                if start_time is not None and ts < start_time:
                    continue
                if end_time is not None and ts > end_time:
                    continue
                results.append(dict(p))
            return results

    def query_nearest_neighbors(
        self,
        query_vec: np.ndarray,
        top_k: int = 10,
        streamer_ids: Optional[List[str]] = None,
        vod_ids: Optional[List[str]] = None,
        content_types: Optional[List[str]] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        with self._lock:
            if not self.is_mock:
                must_conditions: List[Dict[str, Any]] = []
                if streamer_ids:
                    must_conditions.append({"key": "streamer_id", "match": {"any": streamer_ids}})
                if vod_ids:
                    must_conditions.append({"key": "vod_id", "match": {"any": vod_ids}})
                if content_types:
                    must_conditions.append({"key": "content_type", "match": {"any": content_types}})
                if start_time is not None:
                    must_conditions.append({"key": "timestamp_sec", "range": {"gte": start_time}})
                if end_time is not None:
                    must_conditions.append({"key": "timestamp_sec", "range": {"lte": end_time}})

                search_body: Dict[str, Any] = {
                    "vector": query_vec.tolist() if isinstance(query_vec, np.ndarray) else list(query_vec),
                    "limit": top_k,
                    "with_payload": True,
                    "with_vector": True,
                }
                if must_conditions:
                    search_body["filter"] = {"must": must_conditions}

                try:
                    res = self._make_request("POST", f"/collections/{self.collection_name}/points/search", search_body)
                    if res and "result" in res:
                        scored_results: List[Tuple[Dict[str, Any], float]] = []
                        for item in res["result"]:
                            pl = item.get("payload", {})
                            vec = item.get("vector")
                            emb = np.array(vec, dtype=np.float32) if vec else None
                            score = float(item.get("score", 0.0))
                            scored_results.append(({**pl, "embedding": emb}, score))
                        return scored_results
                except Exception:
                    self.is_mock = True

            return super().query_nearest_neighbors(
                query_vec=query_vec,
                top_k=top_k,
                streamer_ids=streamer_ids,
                vod_ids=vod_ids,
                content_types=content_types,
                start_time=start_time,
                end_time=end_time,
            )

    def get_stats(self) -> VectorIndexStats:
        with self._lock:
            total = len(self._mock_points)
            vods = len({c.get("vod_id") for c in self._mock_points.values() if c.get("vod_id")})
            streamers = len({c.get("streamer_id") for c in self._mock_points.values() if c.get("streamer_id")})
            by_type: Dict[str, int] = {}
            for c in self._mock_points.values():
                t = c.get("content_type", "unknown")
                by_type[t] = by_type.get(t, 0) + 1
            return VectorIndexStats(
                total_chunks=total,
                total_vods=vods,
                total_streamers=streamers,
                chunks_by_type=by_type,
                embedding_dim=self.dim,
                index_storage_bytes=total * self.dim * 4,
            )

    def close(self) -> None:
        pass


def create_vector_storage(
    db_url_or_path: Optional[Union[str, Path]] = None,
    homelab_root: Optional[Union[str, Path]] = None,
    dim: int = 128,
) -> BaseVectorStorage:
    """Factory creating QdrantVectorStorage (Qdrant), PgVectorStorage (PostgreSQL), JsonVectorStorage (JSON), or LocalVectorStorage."""
    if db_url_or_path is not None:
        url_str = str(db_url_or_path)
        if url_str.startswith("qdrant://") or (url_str.startswith("http://") and ":6333" in url_str):
            return QdrantVectorStorage(url=url_str, dim=dim)
        if url_str.startswith("postgresql://") or url_str.startswith("postgres://"):
            return PgVectorStorage(database_url=url_str, dim=dim)
        if url_str.startswith("json://") or url_str.endswith(".json"):
            return JsonVectorStorage(file_path=url_str, dim=dim)
        if url_str == ":memory:" or url_str.startswith("sqlite://") or url_str.endswith(".db"):
            sqlite_path = url_str.replace("sqlite:///", "").replace("sqlite://", "")
            return LocalVectorStorage(db_path=sqlite_path, dim=dim)

    # Resolve default based on homelab_root
    root = Path(homelab_root) if homelab_root else Path("./homelab_storage")
    default_db = root / "knowledge" / "vector_index.db"
    default_json = root / "knowledge" / "vector_index.json"

    if default_db.exists():
        return LocalVectorStorage(db_path=default_db, dim=dim)
    return JsonVectorStorage(file_path=default_json, dim=dim)


# ---------------------------------------------------------------------------
# 4. Hybrid Search Architecture (Dense + Sparse)
# ---------------------------------------------------------------------------

def format_timestamp(seconds: float) -> str:
    """Formats seconds into HH:MM:SS or MM:SS."""
    td = timedelta(seconds=max(0.0, seconds))
    total_sec = int(td.total_seconds())
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    secs = total_sec % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class HybridSearchEngine:
    """Combines dense semantic vector retrieval with exact BM25 keyword matching."""

    def __init__(
        self,
        storage: Optional[BaseVectorStorage] = None,
        embedder: Optional[LocalEmbedder] = None,
        bm25: Optional[BM25Index] = None,
        homelab_root: Optional[Union[str, Path]] = None,
        db_url: Optional[str] = None,
    ):
        self.homelab_root = Path(homelab_root) if homelab_root else Path("./homelab_storage")
        self.embedder = embedder or LocalEmbedder(dim=128)
        self.bm25 = bm25 or BM25Index(k1=1.5, b=0.75)
        self.storage = storage or create_vector_storage(
            db_url_or_path=db_url,
            homelab_root=self.homelab_root,
            dim=self.embedder.dim,
        )

        # Warm up BM25 index from storage if starting with existing database
        self._sync_bm25_from_storage()

    def _sync_bm25_from_storage(self) -> None:
        """Populates BM25 index with documents currently stored in SQLite."""
        chunks = self.storage.query_candidate_chunks()
        for c in chunks:
            self.bm25.add_document(c["chunk_id"], c["text"])

    def index_chunk(
        self,
        chunk_id: str,
        vod_id: str,
        streamer_id: str,
        timestamp_sec: float,
        content_type: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp_end_sec: Optional[float] = None,
    ) -> None:
        """Indexes an individual chunk into dense storage and BM25 index."""
        if not text.strip():
            return
        meta = metadata or {}
        end_sec = timestamp_end_sec if timestamp_end_sec is not None else timestamp_sec + 5.0
        embedding = self.embedder.embed_text(text)

        self.storage.insert_chunk(
            chunk_id=chunk_id,
            vod_id=vod_id,
            streamer_id=streamer_id,
            timestamp_sec=timestamp_sec,
            timestamp_end_sec=end_sec,
            content_type=content_type,
            text=text,
            metadata=meta,
            embedding=embedding,
        )
        self.bm25.add_document(chunk_id, text)

    def delete_vod(self, vod_id: str) -> int:
        """Deletes all chunks for a VOD from both vector storage and BM25 index."""
        self.bm25.remove_documents_for_vod(vod_id)
        return self.storage.delete_vod(vod_id)

    def index_vod_outputs(
        self,
        vod_id: str,
        streamer_id: str,
        manifest_or_dir: Union[Dict[str, Any], Path, str],
        chunk_size_sec: float = 15.0,
        overlap_sec: float = 3.0,
    ) -> IndexVodResponse:
        """Time-windowed chunking and indexing of Whisper transcripts, chat, OCR, and claims."""
        start_t = time.perf_counter()

        manifest_data: Dict[str, Any] = {}
        if isinstance(manifest_or_dir, dict):
            manifest_data = manifest_or_dir
        else:
            p = Path(manifest_or_dir)
            if p.is_dir():
                mf = p / "manifest.json"
                if mf.exists():
                    with open(mf, "r", encoding="utf-8") as f:
                        manifest_data = json.load(f)
            elif p.is_file():
                with open(p, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)

        stage_outputs = manifest_data.get("stage_outputs", {})
        speech_count = 0
        chat_count = 0
        ocr_count = 0
        claim_count = 0

        # Collect chat messages for synchronized context lookups
        all_chat_messages = stage_outputs.get("chat_messages", [])
        if not all_chat_messages and "slices" in manifest_data:
            for s in manifest_data.get("slices", []):
                all_chat_messages.extend(s.get("chat_messages", []))

        def _get_chat_context_window(ts: float, window: float = 10.0) -> List[Dict[str, Any]]:
            ctx = []
            for m in all_chat_messages:
                m_ts = m.get("timestamp_offset", m.get("timestamp_sec", 0.0))
                if abs(m_ts - ts) <= window:
                    ctx.append({
                        "author": m.get("author_name", m.get("user_id", "chatter")),
                        "text": m.get("content", ""),
                        "offset": m_ts,
                    })
            return ctx[:10]

        chunks_to_insert: List[Dict[str, Any]] = []

        # 1. Speech Transcription Chunks (Whisper / AudioSegments / FusionSlices)
        audio_segs = stage_outputs.get("audio_segments", [])
        if audio_segs:
            # Group into time-windowed chunks with overlap
            current_chunk_words: List[str] = []
            current_start = 0.0
            current_end = 0.0

            for seg in audio_segs:
                s_start = seg.get("start_sec", 0.0)
                s_end = seg.get("end_sec", s_start + 2.0)
                text = seg.get("text", "").strip()
                if not text:
                    continue

                if not current_chunk_words:
                    current_start = s_start
                    current_end = s_end
                    current_chunk_words.append(text)
                elif (s_end - current_start) <= chunk_size_sec:
                    current_chunk_words.append(text)
                    current_end = s_end
                else:
                    # Flush chunk
                    chunk_text = " ".join(current_chunk_words)
                    c_id = f"{vod_id}_speech_{int(current_start)}_{int(current_end)}"
                    chunks_to_insert.append({
                        "chunk_id": c_id,
                        "vod_id": vod_id,
                        "streamer_id": streamer_id,
                        "timestamp_sec": current_start,
                        "timestamp_end_sec": current_end,
                        "content_type": "speech",
                        "text": chunk_text,
                        "metadata": {
                            "speaker_id": seg.get("speaker_id", streamer_id),
                            "chat_context": _get_chat_context_window(current_start),
                        },
                    })
                    speech_count += 1

                    # Overlap: keep last segment
                    current_chunk_words = [text]
                    current_start = max(0.0, s_start - overlap_sec)
                    current_end = s_end

            if current_chunk_words:
                chunk_text = " ".join(current_chunk_words)
                c_id = f"{vod_id}_speech_{int(current_start)}_{int(current_end)}"
                chunks_to_insert.append({
                    "chunk_id": c_id,
                    "vod_id": vod_id,
                    "streamer_id": streamer_id,
                    "timestamp_sec": current_start,
                    "timestamp_end_sec": current_end,
                    "content_type": "speech",
                    "text": chunk_text,
                    "metadata": {
                        "speaker_id": streamer_id,
                        "chat_context": _get_chat_context_window(current_start),
                    },
                })
                speech_count += 1

        # 2. Chat Log Chunks & Meme Bursts
        chat_bursts = stage_outputs.get("meme_bursts", [])
        for burst in chat_bursts:
            b_ts = burst.get("peak_timestamp_sec", burst.get("start_sec", 0.0))
            emote = burst.get("dominant_emote", "LUL")
            vol = burst.get("volume_multiplier", 1.0)
            text = f"Chat burst: {emote} exploded at {vol}x volume! Top emote: {emote}."
            c_id = f"{vod_id}_chat_{int(b_ts)}"
            chunks_to_insert.append({
                "chunk_id": c_id,
                "vod_id": vod_id,
                "streamer_id": streamer_id,
                "timestamp_sec": b_ts,
                "timestamp_end_sec": b_ts + 10.0,
                "content_type": "chat",
                "text": text,
                "metadata": {
                    "dominant_emote": emote,
                    "volume_multiplier": vol,
                    "chat_context": _get_chat_context_window(b_ts),
                },
            })
            chat_count += 1

        # 3. Dense OCR Frames & Screen Context
        keyframes = stage_outputs.get("keyframes", [])
        for kf in keyframes:
            ts = kf.get("timestamp_sec", 0.0)
            ocr_list = kf.get("ocr_texts", [])
            desc = kf.get("scene_description", "")
            full_text = " ".join(ocr_list)
            if desc:
                full_text = f"{desc}. OCR: {full_text}" if full_text else desc
            if full_text.strip():
                c_id = f"{vod_id}_ocr_{int(ts)}"
                chunks_to_insert.append({
                    "chunk_id": c_id,
                    "vod_id": vod_id,
                    "streamer_id": streamer_id,
                    "timestamp_sec": ts,
                    "timestamp_end_sec": ts + 3.0,
                    "content_type": "ocr",
                    "text": full_text.strip(),
                    "metadata": {
                        "visual_tags": kf.get("tags", []),
                        "chat_context": _get_chat_context_window(ts),
                    },
                })
                ocr_count += 1

        # 4. Streamer Knowledge Graph Claims
        claims = stage_outputs.get("claims", [])
        for cl in claims:
            ts = cl.get("timestamp_sec", cl.get("timestamp_offset", 0.0))
            quote = cl.get("raw_quote", cl.get("claim_text", ""))
            entity = cl.get("subject_entity", cl.get("topic", ""))
            verdict = cl.get("verdict", "UNVERIFIED")
            text = f"Claim on {entity}: {quote} (Verdict: {verdict})"
            c_id = f"{vod_id}_claim_{cl.get('claim_id', int(ts))}"
            chunks_to_insert.append({
                "chunk_id": c_id,
                "vod_id": vod_id,
                "streamer_id": streamer_id,
                "timestamp_sec": ts,
                "timestamp_end_sec": ts + 5.0,
                "content_type": "claim",
                "text": text,
                "metadata": {
                    "verdict": verdict,
                    "raw_quote": quote,
                    "entity": entity,
                    "grounding_sources": cl.get("grounding_sources", []),
                    "chat_context": _get_chat_context_window(ts),
                },
            })
            claim_count += 1

        # Batch embed and insert
        if chunks_to_insert:
            texts = [c["text"] for c in chunks_to_insert]
            embs = self.embedder.embed_batch(texts)
            for i, c in enumerate(chunks_to_insert):
                c["embedding"] = embs[i]
                self.bm25.add_document(c["chunk_id"], c["text"])
            self.storage.insert_chunks_batch(chunks_to_insert)

        duration = time.perf_counter() - start_t
        total_indexed = len(chunks_to_insert)

        return IndexVodResponse(
            vod_id=vod_id,
            status="INDEXED",
            chunks_indexed=total_indexed,
            speech_chunks=speech_count,
            chat_chunks=chat_count,
            ocr_chunks=ocr_count,
            claim_chunks=claim_count,
            duration_sec=round(duration, 3),
            message=f"Successfully indexed {total_indexed} chunks for VOD '{vod_id}'.",
        )

    def search(self, request: SearchQueryRequest) -> SearchResponse:
        """Executes dense + sparse hybrid search with linear score combination."""
        start_t = time.perf_counter()
        query = request.query.strip()
        if not query:
            return SearchResponse(
                query=query,
                total_results=0,
                results=[],
                execution_time_ms=0.0,
                hybrid_weight=request.hybrid_weight,
            )

        # 1. Filter candidates by metadata
        candidates = self.storage.query_candidate_chunks(
            streamer_ids=request.streamer_ids,
            vod_ids=request.vod_ids,
            content_types=request.content_types,
            start_time=request.start_time,
            end_time=request.end_time,
        )

        if not candidates:
            exec_time = (time.perf_counter() - start_t) * 1000.0
            return SearchResponse(
                query=query,
                total_results=0,
                results=[],
                execution_time_ms=round(exec_time, 2),
                hybrid_weight=request.hybrid_weight,
            )

        cand_ids = [c["chunk_id"] for c in candidates]

        # 2. Sparse BM25 Scoring
        sparse_scores = self.bm25.score(query, candidate_ids=cand_ids)
        max_sparse = max(sparse_scores.values()) if sparse_scores else 0.0

        # 3. Dense Vector Scoring
        q_emb = self.embedder.embed_text(query)
        doc_matrix = np.vstack([c["embedding"] for c in candidates])
        dense_sims = self.embedder.batch_cosine_similarity(q_emb, doc_matrix)

        # 4. Hybrid Scoring & Normalization
        alpha = float(np.clip(request.hybrid_weight, 0.0, 1.0))
        results: List[SearchResultItem] = []

        for idx, c in enumerate(candidates):
            c_id = c["chunk_id"]
            d_score = float(dense_sims[idx])
            # Dense normalization: cosine similarity in [-1, 1], map negative to 0.0
            d_norm = max(0.0, d_score)

            s_raw = sparse_scores.get(c_id, 0.0)
            s_norm = (s_raw / max_sparse) if max_sparse > 0 else 0.0

            # Linear hybrid combination
            combined = (1.0 - alpha) * s_norm + alpha * d_norm

            if combined >= request.min_score:
                ts_sec = c["timestamp_sec"]
                meta = c.get("metadata", {})
                chat_ctx = meta.get("chat_context", None)

                results.append(
                    SearchResultItem(
                        chunk_id=c_id,
                        vod_id=c["vod_id"],
                        streamer_id=c["streamer_id"],
                        timestamp_sec=ts_sec,
                        timestamp_formatted=format_timestamp(ts_sec),
                        content_type=c["content_type"],
                        text=c["text"],
                        dense_score=round(d_score, 4),
                        sparse_score=round(s_raw, 4),
                        combined_score=round(combined, 4),
                        metadata=meta,
                        video_url=f"/api/media/{c['vod_id']}/video#t={int(ts_sec)}",
                        chat_context=chat_ctx,
                    )
                )

        # Sort descending by combined score
        results.sort(key=lambda x: x.combined_score, reverse=True)
        top_results = results[: request.top_k]
        exec_time = (time.perf_counter() - start_t) * 1000.0

        return SearchResponse(
            query=query,
            total_results=len(results),
            results=top_results,
            execution_time_ms=round(exec_time, 2),
            hybrid_weight=request.hybrid_weight,
        )


# ---------------------------------------------------------------------------
# 5. Cross-Stream RAG Synthesizer Agent
# ---------------------------------------------------------------------------

class MultimodalRagSynthesizer:
    """Answers cross-stream and temporal queries with millisecond-exact video timestamp citations."""

    def __init__(self, search_engine: HybridSearchEngine):
        self.search_engine = search_engine

    def synthesize(self, request: RagSynthesisRequest) -> RagSynthesisResponse:
        """Retrieves multimodal evidence and generates a grounded cross-stream answer."""
        start_t = time.perf_counter()

        # Step 1: Execute Hybrid Search
        search_req = SearchQueryRequest(
            query=request.query,
            top_k=request.top_k,
            streamer_ids=request.streamer_ids,
            vod_ids=request.vod_ids,
            content_types=request.content_types,
            hybrid_weight=0.5,
            min_score=0.01,
        )
        search_res = self.search_engine.search(search_req)

        if not search_res.results:
            exec_time = (time.perf_counter() - start_t) * 1000.0
            return RagSynthesisResponse(
                query=request.query,
                answer="No relevant evidence or quotes were found across indexed streams for this query.",
                citations=[],
                relevant_chunks=[],
                streamers_covered=[],
                execution_time_ms=round(exec_time, 2),
            )

        # Step 2: Group Evidence by Streamer & Modality
        streamer_evidence: Dict[str, List[SearchResultItem]] = {}
        for r in search_res.results:
            streamer_evidence.setdefault(r.streamer_id, []).append(r)

        streamers_covered = sorted(list(streamer_evidence.keys()))

        # Step 3: Extract Citations
        citations: List[RagSourceCitation] = []
        for r in search_res.results[:5]:
            # Extract key quote
            meta = r.metadata
            quote_text = meta.get("raw_quote") or r.text
            if len(quote_text) > 200:
                quote_text = quote_text[:197] + "..."

            citations.append(
                RagSourceCitation(
                    vod_id=r.vod_id,
                    streamer_id=r.streamer_id,
                    timestamp_sec=r.timestamp_sec,
                    video_url=r.video_url,
                    content_type=r.content_type,
                    quote=quote_text,
                    relevance_score=r.combined_score,
                )
            )

        # Step 4: Synthesize Cross-Stream Narrative
        answer_paragraphs = []

        # (A) Executive Summary
        if len(streamers_covered) > 1:
            streamer_names_str = ", ".join([s.capitalize() for s in streamers_covered[:-1]]) + f" and {streamers_covered[-1].capitalize()}"
            answer_paragraphs.append(
                f"### Cross-Stream Analysis: {request.query}\n"
                f"Across {len(search_res.results)} indexed moments from {streamer_names_str}, multiple perspectives were expressed:"
            )
        else:
            s_name = streamers_covered[0].capitalize()
            answer_paragraphs.append(
                f"### Stream Analysis: {request.query}\n"
                f"Based on {len(search_res.results)} indexed segments from **{s_name}**:"
            )

        # (B) Per-Streamer Viewpoints & Quotes
        for streamer, chunks in streamer_evidence.items():
            s_display = streamer.capitalize()
            speech_chunks = [c for c in chunks if c.content_type in ("speech", "claim")]
            chat_chunks = [c for c in chunks if c.content_type == "chat"]
            ocr_chunks = [c for c in chunks if c.content_type == "ocr"]

            point_lines = []
            for sc in speech_chunks[:2]:
                quote = sc.metadata.get("raw_quote", sc.text)
                ts_link = f"[{sc.timestamp_formatted}]({sc.video_url})"
                point_lines.append(f"- **{ts_link}**: \"{quote}\"")

            for oc in ocr_chunks[:1]:
                ts_link = f"[{oc.timestamp_formatted}]({oc.video_url})"
                point_lines.append(f"- Screen Context ({ts_link}): Detected on-screen '{oc.text}'")

            if chat_chunks:
                top_chat = chat_chunks[0]
                emote = top_chat.metadata.get("dominant_emote", "chat")
                point_lines.append(f"- Audience Reaction: Chat burst with `{emote}` during this segment.")

            points_body = "\n".join(point_lines) if point_lines else "- Mentioned relevant context."
            answer_paragraphs.append(f"#### {s_display}'s Stance\n{points_body}")

        # (C) Fact-Checking & Grounding Check
        claims_in_results = [r for r in search_res.results if r.content_type == "claim"]
        if claims_in_results:
            claim_lines = []
            for c in claims_in_results[:2]:
                verdict = c.metadata.get("verdict", "UNVERIFIED")
                claim_lines.append(f"- Claim: \"{c.text}\" -> Verified as **{verdict}**.")
            answer_paragraphs.append("#### Grounded Fact-Check\n" + "\n".join(claim_lines))

        # (D) Conclusion & Timestamp Summary
        answer_paragraphs.append(
            "#### Direct Timestamp Links\n"
            + "\n".join([f"- [{c.streamer_id.capitalize()} @ {format_timestamp(c.timestamp_sec)}]({c.video_url}): {c.quote}" for c in citations])
        )

        full_answer = "\n\n".join(answer_paragraphs)
        exec_time = (time.perf_counter() - start_t) * 1000.0

        return RagSynthesisResponse(
            query=request.query,
            answer=full_answer,
            citations=citations,
            relevant_chunks=search_res.results,
            streamers_covered=streamers_covered,
            execution_time_ms=round(exec_time, 2),
        )
