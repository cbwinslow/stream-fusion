"""Live Web Grounding & Semantic Fact-Checking Engine (Spec 20)."""

from datetime import datetime, timezone
import logging
import re
from typing import Any, Callable, Dict, List, Optional
import urllib.parse

from stream_fusion.models.schemas import (
    FactCheckVerdict,
    GroundedClaimResult,
    ScreenWebContext,
    SocialPostCard,
    StreamerClaim,
    WebGroundingCitation,
)

logger = logging.getLogger(__name__)


class LiveWebGroundingEngine:
    """Verifies streamer claims against live web sources, news articles, and social posts."""

    def __init__(
        self,
        search_provider: Optional[Callable[[str, int], List[WebGroundingCitation]]] = None,
        confidence_threshold: float = 0.65,
    ):
        self.search_provider = search_provider or self._default_mock_search_provider
        self.confidence_threshold = confidence_threshold
        self._grounding_cache: Dict[str, GroundedClaimResult] = {}

    @staticmethod
    def formulate_search_query(claim: StreamerClaim) -> str:
        """Formulates a targeted web search query from a structured claim."""
        parts = []
        if claim.subject:
            parts.append(claim.subject.strip())
        if claim.predicate:
            parts.append(claim.predicate.strip())
        if claim.object and claim.object.strip() != claim.subject.strip():
            parts.append(claim.object.strip())

        raw_query = " ".join(parts)
        # Clean special chars
        cleaned = re.sub(r"[^\w\s\-\.\']", " ", raw_query)
        return " ".join(cleaned.split())

    def ground_claim(
        self,
        claim: StreamerClaim,
        web_context: Optional[ScreenWebContext] = None,
        max_citations: int = 3,
    ) -> GroundedClaimResult:
        """Grounds a single claim against search citations and optional on-screen web context."""
        query = self.formulate_search_query(claim)

        # Retrieve citations
        citations = self.search_provider(query, max_citations)

        # Also incorporate on-screen web context if relevant
        if web_context and web_context.social_post_cards:
            for card in web_context.social_post_cards:
                if any(word.lower() in card.post_text.lower() for word in query.split() if len(word) > 3):
                    citations.append(
                        WebGroundingCitation(
                            url=web_context.detected_url or f"https://{web_context.domain or 'screen.local'}",
                            domain=web_context.domain or "screen.local",
                            title=f"On-Screen Post by {card.author_name} ({card.author_handle})",
                            snippet=card.post_text[:200],
                            confidence_score=0.9,
                        )
                    )

        # Semantic verification evaluation
        verdict, explanation = self._evaluate_verdict(claim, citations)

        result = GroundedClaimResult(
            claim_id=claim.claim_id,
            verdict=verdict,
            search_query=query,
            citations=citations,
            explanation=explanation,
            original_claim_statement=claim.statement,
        )

        self._grounding_cache[claim.claim_id] = result
        return result

    def _evaluate_verdict(
        self,
        claim: StreamerClaim,
        citations: List[WebGroundingCitation],
    ) -> tuple[FactCheckVerdict, str]:
        """Evaluates citations against claim text to produce structured verdict."""
        if not citations:
            return (
                FactCheckVerdict.UNSUBSTANTIATED,
                f"No credible citations or search results found for query matching '{claim.subject}'."
            )

        statement_lower = claim.statement.lower()

        contradiction_keywords = [
            "false", "hoax", "debunked", "denied", "not true", "refuted", "rumor", "fake", "incorrect"
        ]
        outdated_keywords = [
            "outdated", "previously", "former", "reversed", "superseded", "no longer"
        ]
        confirmation_keywords = [
            "confirmed", "announced", "official", "released", "stated", "verified", "true"
        ]

        contradiction_matches = 0
        outdated_matches = 0
        confirmation_matches = 0

        for c in citations:
            snippet_lower = c.snippet.lower()
            if any(k in snippet_lower for k in contradiction_keywords):
                contradiction_matches += 1
            if any(k in snippet_lower for k in outdated_keywords):
                outdated_matches += 1
            if any(k in snippet_lower for k in confirmation_keywords):
                confirmation_matches += 1

        if contradiction_matches > 0:
            return (
                FactCheckVerdict.CONTRADICTED,
                f"Claim is contradicted by authoritative sources ({contradiction_matches} refutations found)."
            )
        elif outdated_matches > 0:
            return (
                FactCheckVerdict.OUTDATED,
                f"Claim may have been previously accurate but has been superseded by newer updates."
            )
        elif confirmation_matches > 0:
            return (
                FactCheckVerdict.VERIFIED_TRUE,
                f"Claim is supported by {confirmation_matches} verified web citation(s)."
            )
        else:
            return (
                FactCheckVerdict.UNSUBSTANTIATED,
                "Claim content is plausible but lacks direct official confirmation in retrieved snippets."
            )

    @staticmethod
    def _default_mock_search_provider(query: str, max_results: int = 3) -> List[WebGroundingCitation]:
        """Built-in provider for offline, testing, and standard demo scenarios."""
        q_lower = query.lower()
        results: List[WebGroundingCitation] = []

        if "shutting down" in q_lower or "fake" in q_lower or "hoax" in q_lower:
            results.append(
                WebGroundingCitation(
                    url="https://gamingnews.com/debunked",
                    domain="gamingnews.com",
                    title="Server Shutdown Rumors Debunked as Hoax",
                    snippet="Reports circulating regarding immediate shutdowns are false and debunked by studio spokespersons.",
                    confidence_score=0.90,
                )
            )
        elif "world of warcraft" in q_lower or "blizzard" in q_lower:
            results.append(
                WebGroundingCitation(
                    url="https://worldofwarcraft.blizzard.com/news/123",
                    domain="blizzard.com",
                    title="World of Warcraft Official Patch Notes & Server Status",
                    snippet="Blizzard announced official patch updates and confirmed active server operations across all realms.",
                    confidence_score=0.95,
                )
            )
        elif "starforge" in q_lower:
            results.append(
                WebGroundingCitation(
                    url="https://starforgesystems.com/about",
                    domain="starforgesystems.com",
                    title="Starforge Systems Official PC Specifications",
                    snippet="Official builder of custom performance PCs with verified warranty and creator partnerships.",
                    confidence_score=0.92,
                )
            )
        else:
            # Generic fallback citation
            results.append(
                WebGroundingCitation(
                    url=f"https://en.wikipedia.org/wiki/{urllib.parse.quote(query[:30])}",
                    domain="wikipedia.org",
                    title=f"Overview of {query[:40]}",
                    snippet=f"General reference encyclopedia entry discussing {query[:40]}.",
                    confidence_score=0.75,
                )
            )

        return results[:max_results]
