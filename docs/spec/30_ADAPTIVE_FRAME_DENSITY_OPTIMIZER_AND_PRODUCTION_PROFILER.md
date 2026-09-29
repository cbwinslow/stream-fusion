# SPEC 30: Adaptive Frame Density Optimizer & Production Profiler

## 1. Overview & Objectives

In live stream video processing, fixed frame sampling (e.g. 1 frame every 2.0s) creates an unavoidable dilemma:
- In slow segments (idle screens, pauses, monologue, BRB screens), 2.0s sampling wastes GPU VRAM and cycles on duplicate, low-information frames.
- In climactic segments (gameplay pop-offs, rapid HUD changes, clutch kills, sudden visual jokes), 2.0s sampling is too coarse and often misses the decisive 0.3s visual moment.

**Spec 30** introduces the **Adaptive Frame Density Optimizer & Production Profiler**:
1. **Multimodal Event-Driven Frame Densification**: Dynamically scales frame sampling between `max_interval_sec` (e.g. 5.0s during idle/static moments) down to `min_interval_sec` (e.g. 0.25s–0.5s during chat floods, audio spikes, and scene cuts).
2. **Fast Scene-Change & Motion Detection**: Uses lightweight FFmpeg scene analysis or frame difference heuristics to guarantee frames are captured immediately upon camera angle or screen layout switches.
3. **Timestamp-Targeted Demuxing (`extract_frames_at_timestamps`)**: Extends `MediaDemuxer` to extract non-uniform, timestamp-targeted frames in a single bounded pass.
4. **Pipeline & Bounded Buffer Integration**: Seamlessly integrates into `BoundedBufferManager` and `FullSpectrumPipeline` with zero regressions on existing fixed workflows.
5. **Production Profiler & Telemetry**: Collects compute savings metrics (e.g. frames saved, % GPU reduction, highlight capture accuracy, memory peak) to validate production runs on multi-hour streams.

---

## 2. Technical Architecture & Components

### 2.1 Configuration Schema Additions (`src/stream_fusion/config.py` & `src/stream_fusion/models/schemas.py`)

Extend `VisionConfig` with adaptive sampling controls:
```python
class VisionConfig(BaseModel):
    # Existing settings...
    sample_interval_sec: float = 2.0
    
    # Adaptive Density Optimizer (Spec 30)
    sampling_mode: str = "fixed"  # "fixed" | "adaptive"
    min_interval_sec: float = 0.5  # High-density burst interval (2 FPS)
    max_interval_sec: float = 5.0  # Low-density idle interval (0.2 FPS)
    burst_window_sec: float = 12.0  # Duration to maintain dense sampling after trigger
    chat_burst_zscore_threshold: float = 2.5
    scene_change_threshold: float = 0.35
```

### 2.2 Adaptive Frame Optimizer (`src/stream_fusion/vision/optimizer.py`)

Implements `AdaptiveFrameOptimizer`:
- `compute_optimal_timestamps(duration_sec, chat_messages, audio_segments, scene_cuts) -> List[float]`
- Combines:
  1. Base sparse grid at `max_interval_sec`
  2. Scene change events (insert keyframe at scene cut + 0.1s)
  3. Chat burst intervals (boost to `min_interval_sec` across $[t_{\text{start}} - 2s, t_{\text{end}} + burst\_window]$)
  4. Audio excitement/loudness bursts (boost to `min_interval_sec` during rapid speech / high volume)
- Deduplicates and rounds timestamps to millisecond precision.

### 2.3 Timestamp-Targeted Media Demuxer (`src/stream_fusion/ingest/demuxer.py`)

Adds:
```python
def extract_frames_at_timestamps(
    self,
    video_path: Path,
    output_dir: Path,
    timestamps: List[float],
) -> List[Tuple[float, Path]]:
    """Extracts non-uniform frames at specific timestamps."""
```

### 2.4 Production Profiler (`src/stream_fusion/telemetry/profiler.py`)

Calculates:
- `total_potential_frames` (duration * 60 FPS)
- `fixed_mode_frames` (duration / sample_interval_sec)
- `adaptive_mode_frames`
- `frame_reduction_pct`
- `estimated_gpu_time_saved_sec`
- `climax_density_multiplier`

---

## 3. Verification Criteria

1. **Unit Tests**:
   - `AdaptiveFrameOptimizer` produces sparse grids in idle periods and dense grids in burst periods.
   - `extract_frames_at_timestamps` accurately demuxes at requested timestamps.
   - Seamless fallback to `"fixed"` mode preserves exact legacy behavior.
2. **Integration Tests**:
   - Pipeline runs with `sampling_mode="adaptive"` producing valid `StreamAnalysisResult`.
   - Production profiler generates valid performance benchmarks.
3. **Regression Tests**: All existing 226 tests continue to pass without error.
