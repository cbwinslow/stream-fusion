# StreamFusion: Sponsor & Brand Performance Quantifier Specification (Spec 09)

## 1. Overview & Objectives
Sponsorships on live streams often lack granular return-on-attention (ROA) analytics. Brands pay based on average concurrent viewers (CCV), but have no automated proof of whether the audience actually paid attention, was hyped, or exhibited backlash.

This specification defines the **Sponsor & Brand Quantifier**:
1. Multi-modal Brand Mention Detection across audio transcripts and visual OCR blocks.
2. Temporal Sponsor Segment Isolation ($t_{\text{sponsor\_start}}$ to $t_{\text{sponsor\_end}}$).
3. Chat Impact Window Analysis ($t_{\text{sponsor}} \pm 60\text{s}$) measuring:
   - Brand Mention Velocity in chat.
   - Net Sentiment Polarity & Delta relative to stream baseline.
   - Sponsor Backlash / Skepticism Index (e.g. "SELLOUT", "ADW", "skip").
4. Executive Sponsor Report Card generation (JSON / Markdown / HTML).

---

## 2. Brand Detection Multi-Modal Engine

### 2.1 Brand Registry
A configurable brand profile containing:
```json
{
  "brand_id": "starforge_systems",
  "brand_name": "Starforge Systems",
  "aliases": ["starforge", "star forge", "starforgesystems"],
  "promo_codes": ["ASMON", "STREAMFUSION"],
  "product_keywords": ["pc", "computer", "rig", "gpu", "desktop"]
}
```

### 2.2 Audio & Visual Detection
- **Audio Matching:** Matches brand names, aliases, and promo codes against `AudioSegment.transcript` (handling phonetic fuzzy matching with Levenshtein distance $\le 1$ or normalized phonemes).
- **OCR Matching:** Scans `VisualKeyframe.ocr_text_blocks` for brand logos, banner text, and website URLs.
- **Segment Boundary Identification:** When brand mentions occur in audio or video continuously within a threshold (e.g. gaps $< 30$ seconds), the pipeline aggregates them into a discrete `SponsorSegment(start_sec, end_sec)`.

---

## 3. Metrics & Mathematical Formulations

### 3.1 Chat Mention Velocity
$$\text{MentionVelocity} = \frac{\text{Count}(\text{chat messages containing brand keywords})}{\text{Duration of Sponsor Window (sec)}}$$

### 3.2 Sentiment Delta ($\Delta S$)
$$\Delta S = S_{\text{sponsor}} - S_{\text{stream\_baseline}}$$
Where $S \in [-1.0, +1.0]$. A positive $\Delta S$ indicates genuine audience enthusiasm; a negative $\Delta S$ flags negative sentiment or perceived sellout fatigue.

### 3.3 Sponsor Backlash Index ($B_{\text{sponsor}}$)
$$B_{\text{sponsor}} = \frac{\text{Count}(\text{backlash tokens: 'sellout', 'ad', 'skip', 'cringe', 'L'})}{\text{Total Chat Messages in Window}}$$

### 3.4 Brand Attention Score ($A_{\text{brand}}$)
A normalized compound index ($0 - 100$):
$$A_{\text{brand}} = 100 \times \left(0.4 \cdot \min(1.0, \frac{\text{Mentions}}{50}) + 0.3 \cdot \frac{S_{\text{sponsor}} + 1}{2} + 0.3 \cdot (1.0 - B_{\text{sponsor}})\right)$$

---

## 4. Deliverables & Acceptance Criteria
- Module: `src/stream_fusion/analytics/sponsor_quantifier.py` (`BrandProfile`, `SponsorDetector`, `SponsorReportGenerator`).
- Schemas: `SponsorSegment`, `SponsorImpactReport`.
- Unit tests: `tests/test_sponsor_quantifier.py` with 100% test coverage.
