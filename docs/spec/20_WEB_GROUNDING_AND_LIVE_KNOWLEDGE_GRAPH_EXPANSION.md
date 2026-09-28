# StreamFusion: Web Grounding & Live Knowledge Graph Expansion (Spec 20)

## 1. North Star & Objectives
Streamers constantly make claims, react to breaking internet news, cite unconfirmed rumors, and change their opinions over time. Spec 13 built the base Knowledge Graph and Claim Extractor, while Spec 14 extracted on-screen web pages and social media cards.

Spec 20 connects these systems into an **Autonomous Web Grounding & Temporal Stance Shift Engine**, delivering real-time factual verification and multi-month creator opinion tracking.

### Core Objectives:
1. **Live Web Grounding & Fact-Checking Engine (`LiveWebGroundingEngine`)**:
   - Synthesizes targeted search queries from extracted `StreamerClaim` records and on-screen `ScreenWebContext` / `SocialPostCard` items.
   - Queries external information sources (web search engines, news feeds, Wikipedia, and community wikis) to verify claims.
   - Assigns a structured `FactCheckVerdict` (`VERIFIED_TRUE`, `CONTRADICTED`, `UNSUBSTANTIATED`, `OUTDATED`, `UNVERIFIABLE`) with citation URLs, confidence scores, and explanatory rationale.

2. **Cross-Broadcast Temporal Stance Shift Tracker (`TemporalStanceShiftTracker`)**:
   - Tracks a streamer's stance on key entities (companies, video games, personalities, policies) across months of sequential broadcasts.
   - Detects **Stance Reversals** (e.g. `POSITIVE` $\rightarrow$ `NEGATIVE`), **Nuance Drifts**, and calculates a quantitative **Stance Volatility Index** ($0.0 \le V \le 1.0$).
   - Computes temporal shift velocity ($\Delta \text{Stance} / \Delta t$) with verbatim evidence quotes from both stream timestamps.

3. **Cross-Stream Opinion Synthesis & Contradiction Flagger (`CrossStreamOpinionSynthesizer`)**:
   - Compiles all historical claims and stances regarding an entity into a unified longitudinal consensus profile.
   - Detects direct contradictions across streams (e.g. Day 1: "This update saved the game" vs Day 20: "This update completely killed the game").
   - Generates structured executive summaries and emits typed `StreamFusionEnvelope` events (`CLAIM_GROUNDED`, `STANCE_SHIFT_DETECTED`).

---

## 2. Architecture & Knowledge Bus

```
    Streamer Audio Claims           On-Screen Web Cards (Spec 14)
         (Spec 13)                               │
             │                                   │
             └─────────────────┬─────────────────┘
                               ▼
                ┌─────────────────────────────┐
                │   LiveWebGroundingEngine    │
                │  - Search Query Formulator  │
                │  - Live Citation Retrieval  │
                │  - Semantic Fact-Checker    │
                └──────────────┬──────────────┘
                               │
                               ▼
                   [ GroundedClaimResult ]
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
┌─────────────────────────────┐       ┌─────────────────────────────┐
│ TemporalStanceShiftTracker  │       │ CrossStreamOpinionSynthesis │
│  - Multi-stream Timeline    │       │  - Entity Consensus Profile │
│  - Stance Volatility Score  │       │  - Contradiction Detector   │
│  - Reversal Alert Generator │       │  - Long-form Briefing       │
└─────────────────────────────┘       └─────────────────────────────┘
```

---

## 3. Data Contracts & Models

### 3.1 Fact-Checking & Web Grounding (`GroundedClaimResult`, `WebGroundingCitation`)
- `verdict: FactCheckVerdict`: `VERIFIED_TRUE`, `CONTRADICTED`, `UNSUBSTANTIATED`, `OUTDATED`, `UNVERIFIABLE`.
- `citations: List[WebGroundingCitation]`: URL, domain, title, snippet, published_date, confidence.
- `search_query: str`: Query submitted to verify the claim.
- `explanation: str`: Detailed summary explaining why the claim was verified or contradicted.

### 3.2 Stance Shift Tracking (`StanceShiftRecord`)
- `entity_name: str`: Target entity.
- `previous_stance: StancePolarity`: Earlier recorded stance (`POSITIVE`, `NEGATIVE`, `NEUTRAL`).
- `new_stance: StancePolarity`: Current recorded stance.
- `shift_delta: float`: Numerical delta (-2.0 to +2.0).
- `previous_stream_id: str`: ID of stream with previous stance.
- `new_stream_id: str`: ID of current stream.
- `evidence_quote_before: str`: Verbatim streamer quote from earlier stream.
- `evidence_quote_after: str`: Verbatim streamer quote from current stream.
- `is_reversal: bool`: True if sign inverted (`POSITIVE` $\leftrightarrow$ `NEGATIVE`).

### 3.3 Opinion Synthesis (`EntityOpinionSynthesis`)
- `entity_name: str`
- `overall_consensus_stance: str`
- `total_claims_count: int`
- `contradiction_count: int`
- `volatility_score: float`: Measure of how frequently the streamer's opinion fluctuates.
- `summary: str`: High-level synthesis of the streamer's perspective over time.

---

## 4. CLI & Agent RPC 2.0 Interface

### CLI Commands:
- `streamfusion knowledge ground <claims.json>`: Ground claims against search engine and produce verification report.
- `streamfusion knowledge shifts --entity <name> [--store-dir <path>]`: Display temporal stance shift timeline.
- `streamfusion knowledge synthesize --entity <name>`: Produce longitudinal opinion summary.

### Agent RPC 2.0 Methods:
- `streamfusion.groundClaim`: Submit a claim for live web fact-checking.
- `streamfusion.queryStanceShifts`: Query detected stance shifts for an entity.
- `streamfusion.synthesizeEntityOpinions`: Query comprehensive opinion synthesis.
