"""Fact-Checker & Policy Agent for Autonomous Short Production (Spec 21).

Enforces brand safety, FTC sponsor disclosures, and knowledge graph claim grounding.
"""

from typing import List, Optional

from stream_fusion.models.schemas import (
    AudioSegment,
    AuditStatus,
    ChatMessage,
    ContentAuditReport,
    ShortCandidate,
    SponsorSegment,
    StreamerClaim,
)

# Standard safety trigger list (expandable via config)
DEFAULT_TOXIC_TERMS = [
    "hate",
    "scam",
    "fraud",
    "kys",
    "slur",
    "dox",
    "leak",
]


class PolicyAgent:
    """Enforces platform safety, knowledge claim grounding, and FTC advertising compliance."""

    def __init__(self, toxic_filter_words: Optional[List[str]] = None):
        self.toxic_filter_words = [t.lower() for t in (toxic_filter_words or DEFAULT_TOXIC_TERMS)]

    def audit_candidate(
        self,
        candidate: ShortCandidate,
        audio_segments: Optional[List[AudioSegment]] = None,
        chat_messages: Optional[List[ChatMessage]] = None,
        claims: Optional[List[StreamerClaim]] = None,
        sponsor_segments: Optional[List[SponsorSegment]] = None,
    ) -> ContentAuditReport:
        """Audits a candidate clip against policy, safety, sponsor disclosure, and truth grounding."""
        # 1. Gather all spoken and chatter text
        spoken_text = ""
        if audio_segments:
            relevant_audio = [
                s for s in audio_segments
                if s.end_sec >= candidate.start_sec and s.start_sec <= candidate.end_sec
            ]
            spoken_text = " ".join(s.transcript.lower() for s in relevant_audio)

        chat_text = ""
        if chat_messages:
            relevant_chat = [
                m for m in chat_messages
                if candidate.start_sec <= m.timestamp_offset <= candidate.end_sec
            ]
            chat_text = " ".join(m.content.lower() for m in relevant_chat)

        combined_text = f"{spoken_text} {chat_text}"

        # 2. Check toxic terms
        flagged_terms: List[str] = []
        for word in self.toxic_filter_words:
            if word in combined_text:
                flagged_terms.append(word)

        toxicity_score = min(1.0, len(flagged_terms) * 0.35)

        # 3. Check Sponsor Segments (FTC Compliance)
        has_sponsored_content = False
        sponsor_brand_names: List[str] = []
        if sponsor_segments:
            for sp in sponsor_segments:
                # Check overlap between candidate [start_sec, end_sec] and sponsor [start_sec, end_sec]
                if not (candidate.end_sec < sp.start_sec or candidate.start_sec > sp.end_sec):
                    has_sponsored_content = True
                    brand = getattr(sp, "brand_name", None) or getattr(sp, "brand_id", None) or getattr(sp, "sponsor_name", "Sponsored Partner")
                    if brand not in sponsor_brand_names:
                        sponsor_brand_names.append(brand)

        ftc_disclosure_required = has_sponsored_content
        disclosure_tag = "#ad #sponsored" if ftc_disclosure_required else None

        # 4. Check Knowledge Graph Claims
        claim_verified = True
        claim_notes: Optional[str] = None
        if claims:
            window_claims = [
                c for c in claims
                if candidate.start_sec <= c.timestamp_sec <= candidate.end_sec
            ]
            if window_claims:
                notes = []
                for c in window_claims:
                    claim_str = getattr(c, "raw_quote", None) or getattr(c, "claim_text", "")
                    status = getattr(c, "verification_status", None) or getattr(c, "stance", "RECORDED")
                    notes.append(f"Claim: '{claim_str}' (Status: {status})")
                    if status in ["DEBUNKED", "FALSE"]:
                        claim_verified = False
                claim_notes = "; ".join(notes)

        # 5. Determine Verdict
        if toxicity_score >= 0.7:
            verdict = AuditStatus.REJECTED
        elif toxicity_score >= 0.35 or not claim_verified:
            verdict = AuditStatus.FLAGGED
        else:
            verdict = AuditStatus.PASSED

        return ContentAuditReport(
            audit_status=verdict,
            toxicity_score=round(toxicity_score, 2),
            flagged_terms=flagged_terms,
            has_sponsored_content=has_sponsored_content,
            sponsor_brand_names=sponsor_brand_names,
            ftc_disclosure_required=ftc_disclosure_required,
            disclosure_tag=disclosure_tag,
            claim_verified=claim_verified,
            claim_notes=claim_notes,
        )
