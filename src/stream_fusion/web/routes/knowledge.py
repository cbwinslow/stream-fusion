"""Knowledge Graph, Fact-Checking Claims & Stance Explorer Routes (Spec 27)."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, Request

router = APIRouter(prefix="/api/knowledge", tags=["Knowledge & Stances"])


def _extract_all_manifest_field(homelab_root: str, field_name: str) -> List[Dict[str, Any]]:
    """Gathers records across all analyzed VOD manifests for a given stage output field."""
    results = []
    analyzed_dir = Path(homelab_root) / "analyzed"
    if not analyzed_dir.exists():
        return results

    for manifest_path in analyzed_dir.glob("*/manifest.json"):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                vod_id = data.get("vod_id", manifest_path.parent.name)
                items = data.get("stage_outputs", {}).get(field_name, [])
                for it in items:
                    if isinstance(it, dict) and "vod_id" not in it:
                        it["vod_id"] = vod_id
                    results.append(it)
        except Exception:
            pass
    return results


@router.get("/claims")
def list_streamer_claims(
    streamer_id: Optional[str] = Query(None, description="Filter claims by streamer"),
    verdict: Optional[str] = Query(None, description="Filter by verdict: TRUE, FALSE, MIXED, UNVERIFIABLE"),
    search: Optional[str] = Query(None, description="Search query in claim text"),
    request: Request = None,
) -> List[Dict[str, Any]]:
    """Retrieves fact-checked streamer claims with web grounding citations."""
    root = getattr(request.app.state, "homelab_root", "./homelab_storage")
    claims = _extract_all_manifest_field(root, "claims")

    if not claims:
        # Provide sample grounded claims for UI exploration if none yet analyzed
        claims = [
            {
                "claim_id": "claim_001",
                "vod_id": "asmon_sample_60s",
                "streamer_id": "asmongold",
                "claim_text": "Dragon Age: The Veilguard sold fewer copies than Concord in its opening weekend.",
                "verdict": "FALSE",
                "confidence_score": 0.94,
                "timestamp_offset": 18.5,
                "grounding_sources": [
                    {
                        "title": "Steam Concurrent Player Charts",
                        "url": "https://steamdb.info/app/1845910/charts/",
                        "snippet": "Dragon Age peaked at 89,418 concurrent players on Steam, drastically higher than Concord.",
                    }
                ],
                "topic": "Gaming Industry",
            },
            {
                "claim_id": "claim_002",
                "vod_id": "asmon_sample_60s",
                "streamer_id": "asmongold",
                "claim_text": "World of Warcraft Classic Fresh servers launched in November 2024.",
                "verdict": "TRUE",
                "confidence_score": 0.98,
                "timestamp_offset": 34.0,
                "grounding_sources": [
                    {
                        "title": "Blizzard Entertainment Announcement",
                        "url": "https://worldofwarcraft.blizzard.com/news",
                        "snippet": "World of Warcraft Classic 20th Anniversary Edition fresh realms released November 21, 2024.",
                    }
                ],
                "topic": "World of Warcraft",
            },
        ]

    filtered = claims
    if streamer_id:
        filtered = [c for c in filtered if c.get("streamer_id") == streamer_id]
    if verdict:
        filtered = [c for c in filtered if str(c.get("verdict", "")).upper() == verdict.upper()]
    if search:
        s_lower = search.lower()
        filtered = [c for c in filtered if s_lower in str(c.get("claim_text", "")).lower()]

    return filtered


@router.get("/stances")
def list_entity_stances(
    entity: Optional[str] = Query(None, description="Target entity e.g. Blizzard, Valve, Kick"),
    request: Request = None,
) -> List[Dict[str, Any]]:
    """Retrieves streamer opinion shifts and polarity scores [-1.0, +1.0] across entities."""
    root = getattr(request.app.state, "homelab_root", "./homelab_storage")
    stances = _extract_all_manifest_field(root, "stances")

    if not stances:
        # Default stances for UI display
        stances = [
            {
                "entity": "Blizzard",
                "streamer_id": "asmongold",
                "net_polarity": 0.45,
                "stance_label": "LEAN_FAVORABLE",
                "observation_count": 14,
                "last_expressed_at": "2026-09-29T10:00:00Z",
                "key_topics": ["WoW Fresh", "Mythic Plus", "Subscription Numbers"],
                "polarity_history": [
                    {"date": "2026-09-01", "score": -0.6},
                    {"date": "2026-09-10", "score": -0.2},
                    {"date": "2026-09-20", "score": 0.25},
                    {"date": "2026-09-29", "score": 0.45},
                ],
            },
            {
                "entity": "Twitch",
                "streamer_id": "asmongold",
                "net_polarity": -0.35,
                "stance_label": "LEAN_CRITICAL",
                "observation_count": 8,
                "last_expressed_at": "2026-09-29T10:30:00Z",
                "key_topics": ["Ad Policy", "Bitrates", "Partner Splits"],
                "polarity_history": [
                    {"date": "2026-09-01", "score": -0.2},
                    {"date": "2026-09-15", "score": -0.4},
                    {"date": "2026-09-29", "score": -0.35},
                ],
            },
        ]

    if entity:
        e_lower = entity.lower()
        return [s for s in stances if e_lower in str(s.get("entity", "")).lower()]
    return stances


@router.get("/sponsors")
def list_sponsor_impacts(
    brand: Optional[str] = Query(None, description="Filter by brand name"),
    request: Request = None,
) -> List[Dict[str, Any]]:
    """Retrieves detected sponsor impressions, screen durations, and sentiment audits."""
    root = getattr(request.app.state, "homelab_root", "./homelab_storage")
    sponsors = _extract_all_manifest_field(root, "sponsor_segments")

    if not sponsors:
        sponsors = [
            {
                "sponsor_id": "spons_01",
                "vod_id": "asmon_sample_60s",
                "brand_name": "Starforge Systems",
                "start_sec": 5.0,
                "end_sec": 25.0,
                "duration_sec": 20.0,
                "visual_prominence": 0.85,
                "audio_mention_count": 3,
                "chat_sentiment_during_segment": 0.72,
                "brand_safety_score": 0.99,
                "estimated_impressions": 45000,
            }
        ]

    if brand:
        b_lower = brand.lower()
        return [s for s in sponsors if b_lower in str(s.get("brand_name", "")).lower()]
    return sponsors
