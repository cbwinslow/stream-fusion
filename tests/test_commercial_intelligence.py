"""Unit tests for Commercial Intelligence Suite: Sponsor Audits, Creator Package, and Brand Safety (Spec 33)."""

from pathlib import Path
import pytest

from stream_fusion.commercial.creator_package import CreatorStudioPackager
from stream_fusion.commercial.safety import BrandSafetyScanner
from stream_fusion.commercial.sponsor_audit import SponsorAuditGenerator
from stream_fusion.models.schemas import (
    AudioSegment,
    ChatMessage,
    SponsorContractTerms,
    SponsorImpactReport,
    SponsorSegment,
    StreamAnalysisResult,
    VisualKeyframe,
)


def test_sponsor_audit_generator(tmp_path):
    """Verifies that SponsorAuditGenerator computes talking points, sentiment, and EMV."""
    generator = SponsorAuditGenerator()

    contract = SponsorContractTerms(
        brand_name="Factor Meals",
        mandatory_keywords=["ready in two minutes", "fresh never frozen", "healthy"],
        promo_code="ZACK50",
        contracted_duration_sec=60.0,
    )

    sponsor_seg = SponsorSegment(
        segment_id=1,
        brand_id="Factor Meals",
        start_sec=600.0,
        end_sec=670.0,
        matched_audio_transcripts=["Factor meals are ready in two minutes and fresh never frozen healthy meals use code zack50 for fifty percent off"],
        matched_ocr_texts=["FACTOR MEALS LOGO"],
    )
    impact_report = SponsorImpactReport(
        brand_id="Factor Meals",
        brand_name="Factor Meals",
        sponsor_segment=sponsor_seg,
        chat_mention_count=3,
        mention_velocity=0.04,
        window_start_sec=600.0,
        window_end_sec=670.0,
        sentiment_during_sponsor=0.6,
        stream_baseline_sentiment=0.5,
        sentiment_delta=0.1,
        backlash_index=0.0,
        brand_attention_score=0.8,
        summary="Positive Factor read",
    )

    chat_messages = [
        ChatMessage(
            message_id="m1",
            timestamp_offset=610.0,
            user_id="u1",
            author_name="user1",
            content="W sponsor Factor is actually good",
            metadata={"sentiment_score": 0.8},
        ),
        ChatMessage(
            message_id="m2",
            timestamp_offset=625.0,
            user_id="u2",
            author_name="user2",
            content="what is the discount code?",
            metadata={"sentiment_score": 0.2},
        ),
        ChatMessage(
            message_id="m3",
            timestamp_offset=640.0,
            user_id="u3",
            author_name="user3",
            content="just ordered the 10 box with code zack50",
            metadata={"sentiment_score": 0.9},
        ),
    ]

    report = generator.generate_audit(
        contract=contract,
        sponsor_impact_reports=[impact_report],
        chat_messages=chat_messages,
    )

    assert report.brand_name == "Factor Meals"
    assert report.total_airtime_sec == 70.0
    assert report.talking_points_compliance_pct >= 66.0
    assert report.promo_code_mentioned is True
    assert report.chat_purchase_intent_count >= 2  # "discount code", "ordered", "code"
    assert report.earned_media_value_usd > 1000.0
    assert report.compliance_passed is True

    # Test Markdown Export
    out_file = tmp_path / "factor_audit.md"
    saved = generator.export_audit_markdown(report, out_file)
    assert saved.exists()
    content = saved.read_text(encoding="utf-8")
    assert "Sponsor Proof-of-Performance Audit: Factor Meals" in content
    assert "COMPLIANCE PASSED" in content


def test_creator_studio_packager():
    """Verifies that CreatorStudioPackager generates chapters, CTR titles, and thumbnail picks."""
    packager = CreatorStudioPackager()

    highlights = [
        {"timestamp_sec": 300.0, "score": 0.85},
        {"timestamp_sec": 1800.0, "score": 0.92},
        {"timestamp_sec": 3600.0, "score": 0.75},
    ]
    analysis = StreamAnalysisResult(
        stream_id="test-creator-stream",
        duration_sec=7200.0,
        highlights=highlights,
        slices=[],
    )

    package = packager.generate_youtube_package(analysis=analysis, creator_name="Zackrawrr")

    assert len(package.chapters) >= 3
    assert package.chapters[0]["timestamp"] == "00:00"
    assert len(package.suggested_titles) == 3
    assert any("Zackrawrr" in t for t in package.suggested_titles)
    assert len(package.thumbnail_candidate_timestamps) >= 3
    assert len(package.daily_recap_bullets) >= 2
    assert "Timestamps & Chapters:" in package.seo_description


def test_brand_safety_scanner():
    """Verifies that BrandSafetyScanner detects simulated OCR leaks."""
    scanner = BrandSafetyScanner()

    keyframes = [
        VisualKeyframe(
            frame_index=1,
            timestamp_sec=120.0,
            frame_path="frame1.jpg",
            ocr_text_blocks=["Just checking my bank: 4532 8923 1092 3849 for payment"],
        ),
        VisualKeyframe(
            frame_index=2,
            timestamp_sec=250.0,
            frame_path="frame2.jpg",
            ocr_text_blocks=["Contact business manager at private.streamer@loaded.gg for details"],
        ),
        VisualKeyframe(
            frame_index=3,
            timestamp_sec=400.0,
            frame_path="frame3.jpg",
            ocr_text_blocks=["Server IP: 198.51.100.42 connected"],
        ),
        VisualKeyframe(
            frame_index=4,
            timestamp_sec=600.0,
            frame_path="frame4.jpg",
            ocr_text_blocks=["Authorization header: Bearer abc123def456ghi789jkl012mno345pqr"],
        ),
        VisualKeyframe(
            frame_index=5,
            timestamp_sec=800.0,
            frame_path="frame5.jpg",
            ocr_text_blocks=["Normal gameplay screen with HUD and killfeed"],
        ),
    ]

    alerts = scanner.scan_ocr_for_leaks(keyframes)

    assert len(alerts) == 4
    risk_types = [a.risk_type for a in alerts]
    assert all(r == "ocr_leak" for r in risk_types)

    severities = [a.severity for a in alerts]
    assert "critical" in severities  # CC and Auth Token
    assert "warning" in severities   # Email and IP
