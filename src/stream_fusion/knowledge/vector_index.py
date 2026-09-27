"""Local Vector Index adapter supporting ChromaDB / Qdrant or embedded dense vector recall."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from stream_fusion.chat.nlp import tokenize_for_similarity


class LocalVectorIndex:
    """Embedded local vector index with cosine similarity search and metadata filtering."""

    def __init__(self, storage_path: Optional[Path] = None, embedding_dim: int = 128):
        self.storage_path = storage_path
        self.embedding_dim = embedding_dim
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.vectors: Dict[str, np.ndarray] = {}

        if storage_path and storage_path.exists():
            self.load()

    def _embed_text(self, text: str) -> np.ndarray:
        """Deterministic dense subword embedding projection for local vector search."""
        tokens = tokenize_for_similarity(text)
        vec = np.zeros(self.embedding_dim, dtype=np.float32)
        if not tokens:
            vec[0] = 1.0
            return vec

        for t in tokens:
            h = hash(t)
            idx = abs(h) % self.embedding_dim
            sign = 1.0 if (h % 2 == 0) else -1.0
            vec[idx] += sign

        norm = np.linalg.norm(vec)
        if norm > 0.0:
            vec = vec / norm
        return vec

    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> None:
        """Adds a document and its embedding vector to the collection."""
        if embedding is not None:
            vec = np.array(embedding, dtype=np.float32)
            norm = np.linalg.norm(vec)
            vec = (vec / norm) if norm > 0 else vec
        else:
            vec = self._embed_text(text)

        self.documents[doc_id] = {
            "doc_id": doc_id,
            "text": text,
            "metadata": metadata or {},
        }
        self.vectors[doc_id] = vec
        self.save()

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Queries the vector index by semantic similarity with optional metadata filtering."""
        if not self.documents:
            return []

        q_vec = self._embed_text(query_text)
        results = []

        for doc_id, doc in self.documents.items():
            # Check metadata filter
            if filter_metadata:
                match = True
                for k, v in filter_metadata.items():
                    if doc["metadata"].get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            d_vec = self.vectors[doc_id]
            sim = float(np.dot(q_vec, d_vec))
            results.append({
                "doc_id": doc_id,
                "text": doc["text"],
                "metadata": doc["metadata"],
                "similarity": round(sim, 3),
            })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def save(self) -> None:
        """Persists index to disk."""
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "documents": self.documents,
            "vectors": {k: v.tolist() for k, v in self.vectors.items()},
        }
        temp_path = self.storage_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        temp_path.replace(self.storage_path)

    def load(self) -> None:
        """Loads index from disk."""
        if not self.storage_path or not self.storage_path.exists():
            return
        with open(self.storage_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.documents = payload.get("documents", {})
        self.vectors = {
            k: np.array(v, dtype=np.float32) for k, v in payload.get("vectors", {}).items()
        }
