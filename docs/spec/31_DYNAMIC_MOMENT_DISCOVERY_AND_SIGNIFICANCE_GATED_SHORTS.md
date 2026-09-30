# SPEC 31: Dynamic Moment Discovery & Significance-Gated Short Production

## 1. Overview & Objectives

Previous iterations of the multi-agent short production studio (`Director`, `ShortProductionOrchestrator`) relied on a hardcoded slice count (`top_k=3` or `top_k=5`). While functional for quick unit tests, this static ceiling introduces severe production defects:
- In a 9.5-hour broadcast (such as Zackrawrr VOD `2886498935`), a hardcoded limit of 3–5 clips discards dozens of high-virality, high-engagement moments.
- In short or low-energy broadcasts, forcing a fixed number of clips produces low-quality filler.

**Spec 31** replaces arbitrary top-k constraints with an autonomous, **Significance-Gated Dynamic Moment Discovery Engine**:
1. **Multimodal Significance Gating**: Eliminates fixed quotas in favor of mathematical event thresholds:
   - Chat burst velocity spike ($Z$-score $\ge \tau_z$, default $2.5\sigma$).
   - Multimodal highlight / fusion score ($\ge \tau_h$, default $0.70$).
   - Critical claim debates or stance flip points.
2. **Temporal Non-Overlap & Cluster Merging**: Enforces a minimum temporal spacing buffer (e.g. 60.0s) so rapid successive chat spikes merge into single cohesive narrative shorts rather than fragmented duplicates.
3. **Pristine 1080p Extraction Precedence**: All dynamically discovered short packages are extracted and rendered from the pristine 1080p60 source video *before* any video downsampling or archival lifecycle actions occur.
4. **Safety & Budget Controls**: Provides configurable soft and hard bounds (`min_shorts`, `max_shorts`) to guarantee graceful operation and prevent runaway compute.
5. **Rich Commercial Metadata Generation**: For every candidate, compiles virality metrics, chat velocity profile, facecam framing anchors, and suggested viral hooks.

---

## 2. Technical Architecture & Components

### 2.1 Configuration Schema Additions (`src/stream_fusion/models/schemas.py` & `src/stream_fusion/config.py`)

Extend `FullSpectrumConfig` and `ShortsProductionConfig`:
```python
class DynamicMomentThresholds(BaseModel):
    min_highlight_score: float = 0.65
    chat_burst_zscore: float = 2.5
    min_separation_sec: float = 60.0
    min_duration_sec: float = 20.0
    max_duration_sec: float = 60.0
    safety_max_shorts: Optional[int] = 50
    require_speech: bool = True
```

In `FullSpectrumConfig`:
```python
    dynamic_shorts: bool = True
    short_thresholds: DynamicMomentThresholds = Field(default_factory=DynamicMomentThresholds)
```

### 2.2 Dynamic Anchor Discovery (`src/stream_fusion/production/director.py`)

Refactor `Director.identify_short_candidates`:
- Accept `dynamic: bool = True`, `thresholds: Optional[DynamicMomentThresholds] = None`, and optional `top_k: Optional[int] = None`.
- If `dynamic=True`, filter anchor candidates by:
  $$\text{Score}(m) \ge \tau_h \quad \lor \quad Z_{\text{chat}}(m) \ge \tau_z$$
- Cluster anchors temporally using a sliding non-overlap window:
  $$|t_i - t_j| \ge \text{min\_separation\_sec}$$
- Retain all qualifying moments up to `safety_max_shorts`.
- If `dynamic=False` or `top_k` explicitly provided, gracefully preserve legacy top-k truncation behavior for backwards compatibility.

### 2.3 Short Production Orchestrator (`src/stream_fusion/production/orchestrator.py`)

Update `ShortProductionOrchestrator.produce_shorts`:
- Pass through `dynamic` and `thresholds` parameters to `Director`.
- Render and assemble full 9:16 vertical shorts packages for all dynamically discovered candidates.
- Record dynamic candidate yield and gating statistics in the pipeline manifest.

### 2.4 Integration into `FullSpectrumPipeline` (`src/stream_fusion/orchestration/full_spectrum.py`)

- Pass `config.dynamic_shorts` and `config.short_thresholds` into `self.short_orchestrator.produce_shorts`.
- Record `shorts_produced_count` and stage summary with dynamic discovery yield.

---

## 3. Verification Criteria

1. **Unit Tests (`tests/test_dynamic_moment_discovery.py`)**:
   - Verify `Director` yields all qualifying moments when multiple spikes exceed threshold.
   - Verify temporal deduplication prevents overlapping candidates within `min_separation_sec`.
   - Verify `safety_max_shorts` safely caps runaway moments if synthetic noise is injected.
   - Verify backward compatibility: when `top_k` is explicitly passed or `dynamic=False`, exact legacy behavior is maintained.
2. **Integration Verification**:
   - Ensure `FullSpectrumPipeline` runs cleanly with dynamic shorts enabled and records proper manifest metrics.
   - Verify 100% test suite pass rate across all 233+ tests.
