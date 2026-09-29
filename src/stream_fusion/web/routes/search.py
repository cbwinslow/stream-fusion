"""Semantic Vector Search & Multimodal RAG API Routes (Spec 28)."""

import json
from pathlib import Path
from typing import Any, Dict, Optional
from fastapi import APIRouter, HTTPException, Query, Request

from stream_fusion.knowledge.search import HybridSearchEngine, MultimodalRagSynthesizer
from stream_fusion.models.schemas import (
    IndexVodRequest,
    IndexVodResponse,
    RagSynthesisRequest,
    RagSynthesisResponse,
    SearchQueryRequest,
    SearchResponse,
    VectorIndexStats,
)

router = APIRouter(tags=["Search & RAG"])


def get_search_engine(request: Request) -> HybridSearchEngine:
    """Retrieves or creates singleton HybridSearchEngine attached to app.state."""
    app_state = request.app.state
    if not hasattr(app_state, "search_engine") or app_state.search_engine is None:
        config = getattr(app_state, "config", None)
        homelab_root = getattr(app_state, "homelab_root", "./homelab_storage")
        db_url = getattr(config, "vector_db_url", None) if config else None
        app_state.search_engine = HybridSearchEngine(homelab_root=homelab_root, db_url=db_url)
    return app_state.search_engine


# --- Search & RAG Endpoints ---

@router.post("/api/search/semantic", response_model=SearchResponse)
def search_semantic(req: SearchQueryRequest, request: Request) -> SearchResponse:
    """Executes dense + sparse hybrid search across indexed VOD chunks."""
    engine = get_search_engine(request)
    return engine.search(req)


@router.post("/api/search/rag", response_model=RagSynthesisResponse)
def synthesize_rag(req: RagSynthesisRequest, request: Request) -> RagSynthesisResponse:
    """Performs cross-stream conversational RAG synthesis with millisecond-exact video timestamp citations."""
    engine = get_search_engine(request)
    synthesizer = MultimodalRagSynthesizer(engine)
    return synthesizer.synthesize(req)


@router.get("/api/search/stats", response_model=VectorIndexStats)
def get_index_stats(request: Request) -> VectorIndexStats:
    """Retrieves vector database and BM25 index statistics."""
    engine = get_search_engine(request)
    return engine.storage.get_stats()


# --- VOD Indexing Management Endpoints ---

@router.post("/api/index/vod/{vod_id}", response_model=IndexVodResponse)
def index_vod_endpoint(vod_id: str, request: Request, body: Optional[IndexVodRequest] = None) -> IndexVodResponse:
    """Indexes a harvested and analyzed VOD into the vector and BM25 indexes."""
    engine = get_search_engine(request)
    catalog = getattr(request.app.state, "catalog", None)
    homelab_root = getattr(request.app.state, "homelab_root", "./homelab_storage")

    streamer_id = "unknown"
    manifest_data = None

    # Check catalog first
    if catalog:
        vod = catalog.get_harvested_vod(vod_id)
        if vod:
            streamer_id = vod.streamer_id
            if vod.metadata_path and Path(vod.metadata_path).exists():
                try:
                    with open(vod.metadata_path, "r", encoding="utf-8") as f:
                        manifest_data = json.load(f)
                except Exception:
                    pass

    # Fallback to homelab_storage/analyzed/<vod_id>/manifest.json
    if not manifest_data:
        analyzed_manifest = Path(homelab_root) / "analyzed" / vod_id / "manifest.json"
        if analyzed_manifest.exists():
            try:
                with open(analyzed_manifest, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
                    streamer_id = manifest_data.get("streamer_id", streamer_id)
            except Exception:
                pass

    if not manifest_data:
        raise HTTPException(
            status_code=404,
            detail=f"Analyzed manifest for VOD '{vod_id}' not found in catalog or homelab storage.",
        )

    # Reindex if requested
    if body and body.reindex:
        engine.delete_vod(vod_id)

    chunk_size = body.chunk_size_sec if body else 15.0
    overlap = body.overlap_sec if body else 3.0

    return engine.index_vod_outputs(
        vod_id=vod_id,
        streamer_id=streamer_id,
        manifest_or_dir=manifest_data,
        chunk_size_sec=chunk_size,
        overlap_sec=overlap,
    )


@router.delete("/api/index/vod/{vod_id}")
def delete_vod_index(vod_id: str, request: Request) -> Dict[str, Any]:
    """Deletes all indexed chunks for a VOD from vector and sparse indexes."""
    engine = get_search_engine(request)
    deleted = engine.delete_vod(vod_id)
    return {"status": "DELETED", "vod_id": vod_id, "chunks_deleted": deleted}
