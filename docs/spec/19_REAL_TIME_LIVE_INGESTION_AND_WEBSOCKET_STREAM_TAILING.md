# StreamFusion: Real-Time Live Ingestion & WebSocket Stream Tailing (Spec 19)

## 1. North Star & Objectives
StreamFusion has historically operated on pre-recorded VOD files or downloaded time-slices. Spec 19 introduces native **Real-Time Live Ingestion & Stream Tailing**, transforming StreamFusion into a zero-latency, always-on stream monitoring and event processing engine.

### Core Objectives:
1. **Live HLS / RTMP Video Ingestion & Rolling Media Buffering**:
   - Ingest live HLS streams (`.m3u8` playlists with live `.ts` / `.fmp4` rolling segments) or RTMP streams.
   - Maintain a bounded circular rolling media buffer ($W_{\text{buf}} = 60\text{s} - 300\text{s}$) with automated temporal eviction to guarantee memory and disk constraints ($\le 250\text{ MB}$).
   - Dissect incoming segments into sliding micro-batches ($W_{\text{batch}} = 5\text{s} - 10\text{s}$) of synchronized audio and visual frames.

2. **Live IRC / WebSocket Chat Stream Tailing**:
   - High-throughput asynchronous WebSocket client for Twitch IRC (`wss://irc-ws.chat.twitch.tv:443`) and Kick chat endpoints.
   - Support anonymous connection (`NICK justinfan12345`, `JOIN #channel`) or authenticated tokens.
   - Parse Twitch IRC tags (`display-name`, `user-id`, `badges`, `emotes`, `color`, `tmi-sent-ts`) directly into StreamFusion `ChatMessage` models.
   - Stream incoming tokens directly into Spec 17 `RollingBurstDetector` for real-time viral chatter spike detection ($\ge 3.0\sigma$).

3. **Universal Live Event Broadcaster (WebSocket & Server-Sent Events)**:
   - Ultra-low latency event distribution bus dispatching typed `StreamFusionEnvelope` events (`CHAT_MESSAGE`, `BURST_DETECTED`, `FUSION_SLICE`, `HIGHLIGHT_MOMENT`, `SPONSOR_DETECTED`, `SHORT_PRODUCED`).
   - Support per-client topic/event filtering and replay buffers (retrieving the last $K$ events or events since a given UUID/timestamp).
   - Native dual transport: WebSocket for bidirectional agent control and HTTP Server-Sent Events (SSE) for simple web dashboards.

4. **Live Stream Coordinator**:
   - Orchestrates concurrent live video ingest, chat tailing, sliding micro-batch fusion analysis, and event broadcasting.
   - Robust lifecycle states (`STOPPED`, `STARTING`, `RUNNING`, `PAUSED`, `STOPPING`, `ERROR`) with telemetry health monitoring.

---

## 2. Architecture & Data Flow

```
                      LIVE SOURCE (Twitch / YouTube / Kick)
                                   │
               ┌───────────────────┴───────────────────┐
               ▼                                       ▼
      Live Video / Audio                      Live Chat Stream
       (HLS / RTMP / TS)                   (Twitch IRC / WebSocket)
               │                                       │
               ▼                                       ▼
    ┌──────────────────────┐               ┌──────────────────────┐
    │ CircularSegmentBuffer│               │    LiveChatTailer    │
    │  (Rolling 60-300s)   │               │   (IRC Tag Parser)   │
    └──────────┬───────────┘               └──────────┬───────────┘
               │                                      │
               │ (Sliding Micro-Batches 5-10s)        │ (Real-Time Events)
               ▼                                      ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                   LiveStreamCoordinator                     │
    │  - Audio Transcriber & Diarizer                             │
    │  - Visual Keyframing & OCR                                  │
    │  - Rolling Burst Detector & NLP                             │
    │  - Fusion Matrix & Highlight Detector                       │
    └──────────────────────────────┬──────────────────────────────┘
                                   │
                                   ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                    LiveEventBroadcaster                     │
    │              (WebSocket & Server-Sent Events)               │
    │  - Dispatches typed StreamFusionEnvelope[T]                 │
    │  - Replay buffer & Topic subscriptions                      │
    └──────────────────────────────┬──────────────────────────────┘
                                   │
            ┌──────────────────────┴──────────────────────┐
            ▼                                             ▼
    External Web Dashboards                   Autonomous Studio / Agents
     (Live HTML Scrubber)                      (Real-Time Short Clipper)
```

---

## 3. Data Contracts & Models

### 3.1 Live Stream Configuration (`LiveStreamConfig`)
- `channel_name: str`: Channel or streamer handle (e.g. `asmongold`).
- `platform: LivePlatform`: Enum (`TWITCH`, `KICK`, `YOUTUBE_LIVE`, `CUSTOM_HLS`).
- `stream_url: Optional[str]`: Custom HLS or RTMP URL if not auto-resolved.
- `buffer_duration_sec: float = 120.0`: Rolling buffer window depth.
- `microbatch_duration_sec: float = 10.0`: Window size for downstream sliding fusion.
- `ws_port: int = 8765`: WebSocket broadcaster port.
- `sse_port: int = 8766`: SSE HTTP broadcaster port.
- `anonymous_chat: bool = True`: Connect anonymously to public IRC.

### 3.2 Live Stream Status & Health (`LiveStreamStatus`, `LiveTailHealthMetrics`)
- `state: LiveState`: `STOPPED`, `STARTING`, `RUNNING`, `PAUSED`, `STOPPING`, `ERROR`.
- `uptime_sec: float`: Seconds since tailing started.
- `total_bytes_ingested: int`: Total media bytes buffered.
- `total_chat_messages: int`: Total live chat messages received.
- `active_subscribers: int`: Number of active WebSocket / SSE clients.
- `health_metrics: LiveTailHealthMetrics`:
  - `fps: float`: Ingest frame rate.
  - `chat_messages_per_sec: float`: Rolling chat message velocity.
  - `buffer_latency_sec: float`: Current lag behind live broadcast head.
  - `dropped_frames: int`: Count of dropped or corrupted frames.
  - `memory_mb: float`: Current process memory usage.

### 3.3 Client Subscription (`LiveClientSubscription`)
- `client_id: str`: Unique client identifier.
- `event_types: List[str]`: Subscribed event types (e.g. `["BURST_DETECTED", "HIGHLIGHT_MOMENT"]` or `["*"]`).
- `connected_at: str`: ISO 8601 UTC timestamp.

---

## 4. IRC & WebSocket Protocol Implementation

### Twitch IRC Parsing Spec
IRC messages follow RFC 1459 with IRCv3 tags:
```
@badge-info=;badges=broadcaster/1;color=#00FF7F;display-name=Asmongold;emotes=;first-msg=0;flags=;id=123456;mod=0;room-id=26261471;subscriber=0;tmi-sent-ts=1727500000000;turbo=0;user-id=26261471;user-type= :asmongold!asmongold@asmongold.tmi.twitch.tv PRIVMSG #asmongold :Hello chat
```
The parser extracts:
1. `tags`: Dictionary of key-value pairs (`display-name`, `color`, `emotes`, `user-id`, `tmi-sent-ts`).
2. `prefix`: Sender hostmask (`asmongold!asmongold@asmongold.tmi.twitch.tv`).
3. `command`: `PRIVMSG`, `PING`, `JOIN`, `PART`, etc.
4. `params`: Channel (`#asmongold`) and message payload (`Hello chat`).

Mapped directly to `ChatMessage`:
- `username`: tag `display-name` or prefix username.
- `content`: message payload.
- `timestamp_sec`: `(tmi-sent-ts - stream_start_ms) / 1000.0`.
- `badges`: parsed badges list.
- `emotes`: parsed emote spans.

---

## 5. Event Broadcaster & Replay Mechanics
- Dispatches universal `StreamFusionEnvelope[T]` instances.
- Replay ring buffer keeps the last $N = 250$ emitted envelopes in memory.
- WebSocket clients send subscription request:
  `{"action": "subscribe", "events": ["BURST_DETECTED", "SHORT_PRODUCED"], "replay_count": 10}`.
- SSE endpoint `/events?replay=20` streams continuous newline-delimited `data: {...}\n\n` frames.

---

## 6. CLI & Agent RPC Interface

### CLI:
- `streamfusion live tail <channel> [--platform twitch] [--buffer 120] [--ws-port 8765] [--sse-port 8766]`
- `streamfusion live status`

### Agent RPC 2.0:
- `streamfusion.startLiveTail`: `{ "channel": "asmongold", "platform": "TWITCH", "buffer_duration_sec": 120 }`
- `streamfusion.getLiveStatus`: `{ "stream_id": "live-asmongold-..." }`
- `streamfusion.stopLiveTail`: `{ "stream_id": "live-asmongold-..." }`
