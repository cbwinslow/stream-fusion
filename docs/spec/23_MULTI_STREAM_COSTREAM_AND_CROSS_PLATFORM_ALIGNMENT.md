# StreamFusion: Multi-Stream Co-Stream & Cross-Platform Alignment Specification (Spec 23)

## 1. Overview & North Star Goals

Modern live broadcasting has evolved beyond isolated single-creator streams. High-profile cultural and competitive events—such as game reveals, esports championships, creator award shows, political debates, and group gaming sessions—are routinely broadcast as **co-streams**. Multiple creators simultaneously stream their live reactions and commentary across differing platforms (e.g. Twitch, Kick, and YouTube Live) while participating in a shared voice call or watching the same source feed.

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   LIVE CO-STREAM BROADCASTS                                      │
│      Stream A (Twitch)                   Stream B (Kick)                  Stream C (YouTube Live)        │
│   Asmongold (Transcode +4.2s)         xQc (Transcode +2.1s)             Tim (Transcode +6.5s)            │
└────────────────┬───────────────────────────────┬──────────────────────────────────┬──────────────┘
                 │                               │                                  │
                 ▼                               ▼                                  ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                             stream_fusion.costream SUBSYSTEM                                     │
│                                                                                                  │
│   ┌──────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                              MultiStreamCoordinator                                      │   │
│   │   - Concurrent Bounded Ingest Supervisors with Memory/Task Budgets                       │   │
│   │   - Channel Failure Isolation & Circuit Breakers (1 channel failure != session crash)    │   │
│   │   - Resource Cleanup via AsyncExitStack & Graceful Task Cancellation Shields             │   │
│   └────────────────────────────────────────────┬─────────────────────────────────────────────┘   │
│                                                │                                                 │
│                                                ▼                                                 │
│   ┌──────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                               CrossStreamSyncEngine                                      │   │
│   │   - Pairwise Cross-Correlation on Audio Triggers & Scene Cut Keyframes                   │   │
│   │   - Pairwise Offset Matrix Δt(i, j) & Canonical Reference Alignment                      │   │
│   │   - Normalized Epoch Clock Transformation: T_unified = T_local - Δt_channel               │   │
│   └────────────────────────────────────────────┬─────────────────────────────────────────────┘   │
│                                                │                                                 │
│                 ┌──────────────────────────────┴─────────────────────────────┐                   │
│                 ▼                                                            ▼                   │
│   ┌───────────────────────────┐                              ┌───────────────────────────────┐   │
│   │   CrossAudienceComparator │                              │    CoStreamDebateAnalyzer     │   │
│   │ - Bucket Alignment (Δt=2s)│                              │ - Diarized Turn-Taking Engine │   │
│   │ - Cross-Platform Agreement│                              │ - Inter-Stream Interrupts     │   │
│   │ - Audience Divergence     │                              │ - Talk-Time Ratio & Stance    │   │
│   │ - Meme Cascade Velocity   │                              │   Cross-Examination           │   │
│   └─────────────┬─────────────┘                              └───────────────┬───────────────┘   │
│                 │                                                            │                   │
│                 └──────────────────────────────┬─────────────────────────────┘                   │
│                                                ▼                                                 │
│   ┌──────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                               MultiAngleShortComposer                                    │   │
│   │   - Multi-Stream Highlight Climax Consensus (Simultaneous Crowd Burst Score)             │   │
│   │   - Spatial Multi-Camera Layout Packaging (Stacked Split, PiP, Grid)                     │   │
│   └──────────────────────────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                 │
                                                 ▼
             Universal StreamFusion Co-Stream Envelope & Analytical Manifests
```

### North Star Goals:
1. **Robust Concurrent Orchestration & Resource Isolation**: Coordinate arbitrary numbers of live streams ($K \ge 2$) across Twitch, Kick, and YouTube Live within strict resource boundaries (bounded memory per stream, bounded async queues, and graceful teardown). An error or disconnection on one channel must never crash or desynchronize remaining streams.
2. **Deterministic Cross-Stream Temporal Alignment**: Automatically calculate inter-stream latency offsets ($\Delta t_{i,j}$) using audio trigger cross-correlation, visual frame cut matching, or shared wall-clock anchors, transforming all chat messages, audio segments, and visual keyframes onto a single unified reference clock ($T_{\text{unified}}$).
3. **Cross-Platform Audience Reaction Comparison**: Quantify how different streaming communities react to the exact same moment. Compute the **Cross-Platform Agreement Index ($\mathcal{A}_{\text{cross}}$)**, detect **Audience Divergence / Revolt Alerts** (e.g., YouTube chat approves while Twitch chat revolts), and track **Meme Cascade Propagation** (which platform reacted first and how fast slang spread across platforms).
4. **Co-Host Conversational & Debate Dynamics**: Analyze multi-speaker turn-taking, talk-time ratio between co-streamers, speech interrupts, and mutual agreement/dissent on co-discussed topics.
5. **Multi-Angle Highlight Climax Packaging**: Detect multi-stream viral peaks and automatically assemble multi-angle short candidate packages with synchronized multi-camera layouts (Stacked, Picture-in-Picture, Grid).
6. **100% Backward Compatibility**: All 152 existing test cases across single-stream ingest, fusion, NLP, and multi-agent production remain 100% green.

---

## 2. Resource Management & Fault Isolation Architecture

In co-streaming environments, coordinating multiple high-bitrate video and chat pipelines simultaneously poses severe resource exhaustion hazards (socket leaks, unbound message buffers, memory leaks, and cascading task cancellations).

### 2.1 Bounded Per-Stream Resource Allocation
- **Bounded Message Buffers**: Each stream connector feeds an isolated circular message queue with a maximum depth ($N_{\text{max}} = 2000$ messages). If a downstream analyzer lags, older non-burst messages are dropped with dropped-frame telemetry logged.
- **Media Memory Cap**: Each channel's `CircularSegmentBuffer` enforces a strict memory ceiling ($\le 250\text{ MB}$ per stream) and automatic unlinking of expired `.mp4` / `.wav` segments.
- **Global Memory Ceiling**: The `MultiStreamCoordinator` monitors total session memory usage against a configurable session budget (default: $1024\text{ MB}$).

### 2.2 Fault Isolation & Circuit Breakers
- Each stream supervisor runs in an isolated `asyncio.Task` with individual error traps.
- If channel $S_k$ fails (e.g. Kick WebSocket drops or YouTube quota limits):
  1. Channel state transitions to `ERROR` or `RECONNECTING`.
  2. Circuit breaker records failure count with exponential backoff and jitter.
  3. The `CrossStreamSyncEngine` marks $S_k$ as temporarily degraded while continuing real-time synchronization across surviving channels.
- **Graceful Async Cleanup**: Uses `contextlib.AsyncExitStack` and explicit cancellation shields (`asyncio.shield`) during teardown, ensuring sockets, file handles, and worker loops are closed cleanly with zero unhandled exceptions.

---

## 3. Mathematical Formulations & Algorithms

### 3.1 Cross-Stream Latency Offset Matrix ($\Delta t_{i,j}$)
Let $S_0$ be the designated master reference stream (or earliest detected stream). For any stream $S_i$, let $x_i(t)$ represent its continuous acoustic energy or normalized chat velocity time series.

The pairwise cross-correlation between reference stream $S_0$ and stream $S_i$ over search window $[-\tau_{\text{max}}, +\tau_{\text{max}}]$ is defined as:
$$R_{0, i}(\tau) = \sum_{t} x_0(t) \cdot x_i(t + \tau)$$

The optimal latency delta $\Delta t_i$ is the lag maximizing cross-correlation:
$$\Delta t_i = \arg\max_{\tau \in [-\tau_{\text{max}}, \tau_{\text{max}}]} R_{0, i}(\tau)$$

Once calibrated, any local timestamp $t_i$ from stream $S_i$ is mapped to canonical time $T_{\text{unified}}$:
$$T_{\text{unified}} = t_i - \Delta t_i$$

### 3.2 Cross-Platform Audience Agreement Index ($\mathcal{A}_{\text{cross}}$)
For a synchronized time bucket $B = [T, T + \Delta T]$ (default $\Delta T = 2.0\text{s}$):
Let $p \in \mathcal{P}$ denote the participating platforms (e.g. Twitch, Kick, YouTube). For each platform $p$, let $s_p(B) \in [-1.0, +1.0]$ denote the mean audience sentiment polarity during bucket $B$.

The **Cross-Platform Agreement Index** $\mathcal{A}_{\text{cross}}(B)$ measures the directional consensus across platforms:
$$\mathcal{A}_{\text{cross}}(B) = 1.0 - \frac{1}{2} \cdot \left( \max_{p \in \mathcal{P}} s_p(B) - \min_{p \in \mathcal{P}} s_p(B) \right)$$

- $\mathcal{A}_{\text{cross}} \ge 0.80$: Universal Consensus (all platform chats react in unison).
- $0.50 \le \mathcal{A}_{\text{cross}} < 0.80$: Moderate Alignment (mixed reactions).
- $\mathcal{A}_{\text{cross}} < 0.50$: **Audience Divergence Alert** (sharp polarization across platforms).

### 3.3 Meme Cascade & Burst Propagation Velocity
When a novel slang token or emote $\omega$ spikes in chat:
Let $T_{\text{origin}}(\omega) = \min_{p} T_{\text{burst}}(p, \omega)$ be the timestamp of the first detected burst on origin platform $p_{\text{origin}}$.
For any subsequent platform $p'$, the propagation lag is:
$$\delta t(p', \omega) = T_{\text{burst}}(p', \omega) - T_{\text{origin}}(\omega)$$
The cascade propagation velocity is:
$$v_{\text{cascade}}(\omega) = \frac{1}{|\mathcal{P}| - 1} \sum_{p' \ne p_{\text{origin}}} \delta t(p', \omega)$$

---

## 4. Subsystem Module Layout

```
src/stream_fusion/costream/
├── __init__.py                 # Public package exports
├── coordinator.py              # MultiStreamCoordinator & session supervisor
├── sync_engine.py              # CrossStreamSyncEngine (latency offset matrix & clock alignment)
├── audience_comparator.py      # CrossAudienceComparator (sentiment alignment & divergence alerts)
├── debate_analyzer.py          # CoStreamDebateAnalyzer (turn-taking, talk-time, interrupts)
├── short_composer.py           # MultiAngleShortComposer (multi-camera climax highlights)
└── exceptions.py               # Typed exceptions (CoStreamError, StreamSyncError, etc.)
```

---

## 5. Acceptance Criteria & Definition of Done

1. **Multi-Stream Ingest & Lifecycle Management**:
   - `MultiStreamCoordinator` concurrently manages multiple stream connectors and ingestors.
   - Resource limits (memory budget, bounded queues) strictly enforced.
   - Resilient fault isolation: single stream crash does not affect other streams.
   - Graceful, clean shutdown with zero dangling tasks or unclosed sockets.
2. **Cross-Stream Synchronization**:
   - `CrossStreamSyncEngine` computes cross-stream offsets $\Delta t_{i,j}$ and aligns events to unified epoch.
   - Handles both automated cross-correlation and manual override offsets.
3. **Cross-Audience Reaction Comparison**:
   - Synchronizes sentiment curves across Twitch, Kick, and YouTube in discrete time buckets.
   - Generates Cross-Platform Agreement Index $\mathcal{A}_{\text{cross}}$ and flags audience divergence moments.
   - Tracks meme cascade propagation across platforms.
4. **Co-Host Debate & Interaction Analysis**:
   - Evaluates multi-channel transcripts and diarized turns.
   - Calculates talk-time ratio, interrupts, and agreement/disagreement scores.
5. **Multi-Angle Highlight Packaging**:
   - Identifies concurrent multimodal spikes across streams.
   - Produces `MultiAngleShortCandidate` packages with spatial layout presets (`STACKED_SPLIT`, `PICTURE_IN_PICTURE`, `SIDE_BY_SIDE`).
6. **CLI & JSON-RPC Integration**:
   - CLI commands `streamfusion costream align`, `streamfusion costream compare`, and `streamfusion costream start`.
   - JSON-RPC 2.0 methods added to `agent_rpc.py`.
7. **Testing & Code Quality**:
   - 100% green test suite across all 152 existing tests plus complete new test suite for Spec 23.
   - Clean, modern typing, error handling, and resource management.
