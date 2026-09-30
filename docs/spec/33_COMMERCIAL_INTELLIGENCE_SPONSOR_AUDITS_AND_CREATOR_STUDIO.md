# SPEC 33: Commercial Intelligence: Sponsor Proof-of-Performance Audits & Creator Studio Packages

## 1. Overview & Objectives

StreamFusion's unified multimodal engine collects word-level speech transcripts, millisecond-level chat sentiment, OCR screen extractions, and facecam presence. While this data is indexed into PostgreSQL and ClickHouse, turning it into direct monetization requires specialized, high-margin reporting and packaging engines.

**Spec 33** establishes the **Commercial Intelligence Suite**:
1. **Automated Sponsor Proof-of-Performance Audit (`SponsorAuditGenerator`)**:
   - Computes exact on-screen logo exposure time and bounding-box screen percentage.
   - Verifies verbal talking points and promo codes against contracted campaign requirements.
   - Calculates real-time chat sentiment distribution ($% \text{positive}$, $% \text{negative}$, $% \text{conversion/code intent}$) during the ad read.
   - Computes Earned Media Value (EMV) based on audience engagement and airtime duration.
   - Auto-generates a ready-to-deliver Markdown / PDF executive compliance deck for brand CMOs and talent agencies.
2. **Creator Studio & YouTube Long-Form Package (`CreatorStudioPackager`)**:
   - Generates precise, timestamped YouTube chapters with high-engagement topic labels.
   - Produces 3 algorithmic high-CTR YouTube video titles derived from peak controversy/humor moments.
   - Generates full SEO description, hashtags, and game credits.
   - Selects 5 high-contrast thumbnail freeze-frame candidates based on streamer facial expressions (shocked, laughing, rage) and visual sharpness.
   - Generates a 3-minute "Daily Briefing / Catch-Up" executive summary for community newsletters and Discord updates.
3. **Brand Safety & TOS Violation Scanner (`BrandSafetyScanner`)**:
   - Analyzes OCR streams for accidental personal information leaks (credit cards, email addresses, phone numbers, private Discord DMs, IP addresses).
   - Flags potential DMCA audio risk windows and community guidelines friction points.
4. **Chatter Purchase Intent & Superfan Radar (`ChatterLeadRadar`)**:
   - Extracts product purchase intent questions (*"what monitor?", "is that game worth it?", "code didn't work"*), gift sub frequency, and top influential chat leaders.

---

## 2. Technical Architecture & Components

### 2.1 Configuration Schema Additions (`src/stream_fusion/models/schemas.py`)

```python
class SponsorContractTerms(BaseModel):
    brand_name: str
    mandatory_keywords: List[str] = Field(default_factory=list)
    promo_code: Optional[str] = None
    contracted_duration_sec: float = 60.0
    expected_logo_presence_sec: float = 30.0

class SponsorAuditReport(BaseModel):
    audit_id: str
    brand_name: str
    airtime_start_sec: float
    airtime_end_sec: float
    total_airtime_sec: float
    logo_exposure_sec: float
    talking_points_compliance_pct: float
    promo_code_mentioned: bool
    chat_positive_sentiment_pct: float
    chat_purchase_intent_count: int
    earned_media_value_usd: float
    executive_summary: str
    compliance_passed: bool

class YouTubeMetadataPackage(BaseModel):
    suggested_titles: List[str]
    chapters: List[Dict[str, str]]  # [{"timestamp": "01:23:45", "label": "Testing New MMO"}]
    seo_description: str
    seo_tags: List[str]
    thumbnail_candidate_timestamps: List[float]
    daily_recap_bullets: List[str]

class BrandSafetyAlert(BaseModel):
    timestamp_sec: float
    risk_type: str  # "ocr_leak" | "dmca_risk" | "tos_flag"
    description: str
    severity: str  # "warning" | "critical"
    redaction_recommended: bool
```

### 2.2 Sponsor Audit Engine (`src/stream_fusion/commercial/sponsor_audit.py`)

Implements `SponsorAuditGenerator`:
- `generate_audit(sponsor_report: SponsorReport, chat_messages: List[ChatMessage], contract: SponsorContractTerms) -> SponsorAuditReport`
- `export_audit_markdown(report: SponsorAuditReport, output_file: Path) -> Path`

### 2.3 Creator Studio Packager (`src/stream_fusion/commercial/creator_package.py`)

Implements `CreatorStudioPackager`:
- `generate_youtube_package(analysis: StreamAnalysisResult, audio_segments: List[AudioSegment], claims: List[GroundedClaim]) -> YouTubeMetadataPackage`
- Identifies macro-topic transitions to build accurate chapter markers.
- Extracts peak emotional frames for thumbnail recommendations.

### 2.4 Brand Safety Scanner (`src/stream_fusion/commercial/safety.py`)

Implements `BrandSafetyScanner`:
- Scans OCR text streams with regular expressions for credit card patterns, IPv4 addresses, and email patterns.
- Returns timestamped alerts for immediate editor redaction.

---

## 3. Verification Criteria

1. **Unit Tests (`tests/test_commercial_intelligence.py`)**:
   - Verify `SponsorAuditGenerator` accurately calculates talking-point compliance, promo code detection, and chat conversion sentiment.
   - Verify `CreatorStudioPackager` outputs valid YouTube chapter formatting (`HH:MM:SS Title`) and thumbnail suggestions.
   - Verify `BrandSafetyScanner` detects simulated screen leaks (emails, credentials) and emits high-severity alerts.
2. **Integration Verification**:
   - Verify pipeline execution exports commercial intelligence reports into the output directory alongside manifest records.
   - 100% test pass rate with zero regressions.
