# StreamFusion: Multi-Platform Live Stream & Chat Connectors Specification (Spec 22)

## 1. Overview & North Star Goals

StreamFusion's real-time multimodal engine must support cross-platform broadcasting environments where creators stream on Twitch, Kick, and YouTube Live simultaneously or individually. While each platform employs fundamentally different transport layers, wire encodings, and message schemas, the downstream intelligence pipeline (latency calibration, sentiment analysis, burst detection, knowledge graph grounding, and autonomous vertical shorts production) requires a uniform, high-fidelity stream of events.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   LIVE BROADCASTS                                      │
│        Twitch (IRC/EventSub)         Kick (Pusher WS)         YouTube Live (InnerTube) │
└──────────────────┬──────────────────────────┬──────────────────────────┬───────────────┘
                   │                          │                          │
                   ▼                          ▼                          ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        stream_fusion.connectors SUBSYSTEM                              │
│                                                                                        │
│   ┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────────┐   │
│   │ TwitchChatConnector │    │  KickChatConnector   │    │  YouTubeChatConnector   │   │
│   └──────────┬──────────┘    └──────────┬───────────┘    └────────────┬────────────┘   │
│              │                          │                             │                │
│              └──────────────────────────┼─────────────────────────────┘                │
│                                         ▼                                              │
│                        ┌─────────────────────────────────┐                             │
│                        │      BaseChatConnector (ABC)    │                             │
│                        │ - Async Connection State Machine│                             │
│                        │ - Exponential Backoff & Jitter  │                             │
│                        │ - Sliding-Window Velocity Metric│                             │
│                        │ - Rolling Burst Buffer & Fanout │                             │
│                        └────────────────┬────────────────┘                             │
│                                         │                                              │
│                                         ▼                                              │
│                        ┌─────────────────────────────────┐                             │
│                        │        MessageNormalizer        │                             │
│                        │ - Emote, Badge & Run Parser     │                             │
│                        │ - SuperChat / Bits Monetization │                             │
│                        │ - Stream Timestamp Normalizer   │                             │
│                        └────────────────┬────────────────┘                             │
└─────────────────────────────────────────┼──────────────────────────────────────────────┘
                                          │
                                          ▼
                         Canonical ChatMessage & Emotes
                                          │
                                          ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        CORE MULTIMODAL INTELLIGENCE PIPELINE                           │
│     RollingBurstDetector  ──►  LiveEventBroadcaster  ──►  ShortProductionStudio        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### North Star Goals:
1. **Universal Protocol Normalization**: Encapsulate all platform quirks (RFC 1459 IRC text lines, Pusher WebSocket double-encoded JSON, and YouTube InnerTube runs/renderers) behind a unified, polymorphic connector interface.
2. **Centralized Infrastructure, Zero Duplication**: Centralize connection lifecycle state machines, reconnection backoff with jitter, sliding-window message velocity tracking, and rolling burst buffers into reusable base classes.
3. **Rich Metadata Preservation**: Capture platform-specific monetization signals (YouTube Super Chats, Twitch Bits, Kick Gift Subscriptions) without compromising canonical `ChatMessage` compatibility.
4. **Resilience & Zero External Blockers**: Gracefully operate with zero required API keys using public streaming endpoints (`chat-downloader` for YouTube, Pusher WebSockets for Kick, IRC for Twitch) while providing clean extension hooks for authenticated developer APIs.
5. **100% Backward Compatibility**: Guarantee all 140 existing test cases continue to pass without modification.

---

## 2. Platform Protocol Deep-Dive & Wire Format Specifications

### 2.1 Twitch (IRC RFC 1459 + IRCv3 Tags)
- **Transport**: TLS TCP Socket (`irc.chat.twitch.tv:6697`) or WebSocket (`wss://irc-ws.chat.twitch.tv:443`).
- **Handshake Sequence**:
  1. `CAP REQ :twitch.tv/tags twitch.tv/commands`
  2. `PASS SCHMOOPIIE` (or OAuth token)
  3. `NICK justinfan<rand>`
  4. `JOIN #<channel>`
- **Keepalive**: Server sends `PING :tmi.twitch.tv`; Client responds `PONG :tmi.twitch.tv`.
- **Emotes**: Tag `emotes=<id>:<start>-<end>,<start>-<end>/...`.
- **Monetization**: `bits=<count>` tag in `PRIVMSG` or `USERNOTICE` events for subscriptions/resubs/gifts.

### 2.2 Kick (Pusher Channels Protocol over RFC 6455 WebSocket)
- **Transport**: RFC 6455 WebSocket (`wss://ws-us2.pusher.com/app/eb1d5f283081a78b932c?protocol=7&client=js&version=7.6.0&flash=false`).
- **Handshake Sequence**:
  1. WebSocket Connect.
  2. Receive `{"event":"pusher:connection_established","data":"{\"socket_id\":\"<id>\",\"activity_timeout\":120}"}`.
  3. Send `{"event":"pusher:subscribe","data":{"channel":"chatrooms.<chatroom_id>.v2"}}`.
  4. Receive `{"event":"pusher_internal:subscription_succeeded","channel":"chatrooms.<chatroom_id>.v2","data":"{}"}`.
- **Keepalive**: Server sends `{"event":"pusher:ping","data":{}}`; Client replies `{"event":"pusher:pong","data":{}}`.
- **Message Event**: `event == "App\\Events\\ChatMessageEvent"`. The `data` field contains a JSON-serialized string with:
  - `id`: Unique message ID (UUID).
  - `chatroom_id`: Channel chatroom integer.
  - `content`: Text body containing inline emote tokens formatted as `[emote:<id>:<name>]`.
  - `created_at`: Timestamp string (`YYYY-MM-DD HH:MM:SS` UTC).
  - `sender`: Dict with `id`, `username`, `slug`, `identity.badges` (list of badge objects with `type`, `text`, `count`).
- **Monetization & Gift Events**:
  - `App\Events\LuckyUsersWhoGotGiftSubscriptionsEvent`
  - `App\Events\GiftedSubscriptionsEvent`

### 2.3 YouTube Live (InnerTube Live Chat & Data API v3)
- **Transport**:
  - Unauthenticated Real-Time: HTTP/2 long-polling via InnerTube RPC (`chat-downloader` engine) using continuation tokens.
  - Authenticated: YouTube Data API v3 (`liveChatMessages.list`).
- **Polling Loop**:
  - Initiates request with initial live video ID / channel handle.
  - Receives batch of `actions` containing `addChatItemAction`.
  - Extracts next `continuation` token and awaits `timeoutMs` specified by server (typically 1000ms–4000ms).
- **Message Renderers**:
  - Standard Chat: `liveChatTextMessageRenderer`
  - Super Chat: `liveChatPaidMessageRenderer` (contains `purchaseAmountText`, `headerBackgroundColor`, `bodyBackgroundColor`).
  - Super Sticker: `liveChatPaidStickerRenderer`
  - Membership: `liveChatMembershipItemRenderer`
- **Emotes & Runs**:
  - Message content is structured as an array of `runs`:
    `runs: [{"text": "Hello "}, {"emoji": {"emojiId": "...", "shortcuts": [":smile:"]}}]`.
  - Must concatenate text runs while extracting custom emoji tokens.
- **Author Metadata**:
  - `author.id` (Channel ID starting with `UC`), `author.name`, `author.badges` (Owner, Moderator, Member).

---

### 3. Architecture & Class Hierarchy

```
src/stream_fusion/connectors/
├── __init__.py                 # Public exports
├── base.py                     # BaseChatConnector, PlatformCapabilities, ConnectorState
├── normalizer.py               # MessageNormalizer
├── registry.py                 # ConnectorRegistry
├── twitch/
│   ├── __init__.py
│   └── irc_connector.py        # TwitchChatConnector
├── kick/
│   ├── __init__.py
│   ├── pusher_connector.py     # KickChatConnector
│   ├── resolver.py             # KickChannelResolver
│   └── webhook_receiver.py     # KickWebhookReceiver
└── youtube/
    ├── __init__.py
    ├── chat_downloader.py      # YouTubeChatConnector (InnerTube via chat-downloader)
    └── data_api.py             # YouTubeDataApiConnector (Official v3 API)
```

### 3.1 Centralized Abstract Base Class (`BaseChatConnector`)
```python
class BaseChatConnector(ABC):
    def __init__(
        self,
        channel_name: str,
        platform: LivePlatform,
        capabilities: PlatformCapabilities,
        burst_detector: Optional[RollingBurstDetector] = None,
        stream_start_ms: Optional[float] = None,
        max_reconnect_attempts: int = 10,
        base_reconnect_delay: float = 2.0,
        max_reconnect_delay: float = 60.0,
    ):
        ...
    
    # State & Health
    @property
    def is_running(self) -> bool: ...
    @property
    def state(self) -> ConnectorState: ...
    @property
    def total_messages(self) -> int: ...
    def get_message_velocity(self, window_sec: float = 10.0) -> float: ...

    # Lifecycle Methods
    async def start(self) -> None: ...
    async def stop(self) -> None: ...

    # Callback Subscriptions
    def register_callback(self, cb: Callable[[ChatMessage], Any]) -> None: ...
    def register_burst_callback(self, cb: Callable[[Any], Any]) -> None: ...

    # Subclass hooks to implement:
    @abstractmethod
    async def _connect_and_listen(self) -> None: ...
    @abstractmethod
    async def _disconnect(self) -> None: ...
```

### 3.2 Canonical Message Normalizer (`MessageNormalizer`)
Transforms raw payloads from any platform into the unified `ChatMessage`:
- `normalize_twitch_irc(msg: TwitchIrcMessage) -> Optional[ChatMessage]`
- `normalize_kick_pusher(event_data: Dict[str, Any]) -> Optional[ChatMessage]`
- `normalize_youtube_item(item: Dict[str, Any]) -> Optional[ChatMessage]`

Standardized `metadata` keys:
- `platform`: `"TWITCH" | "KICK" | "YOUTUBE_LIVE"`
- `monetization`: `{"type": "SUPER_CHAT" | "BITS" | "GIFT_SUB", "amount": float, "currency": str, "tier": str}`
- `raw_badges`: List of raw platform badge identifiers

---

## 4. Acceptance Criteria & Verification Plan

1. **Connector Architecture**:
   - `BaseChatConnector` provides reliable reconnection with exponential backoff and jitter.
   - `PlatformCapabilities` specifies supported feature flags.
   - `ConnectorRegistry` correctly instantiates connectors for `TWITCH`, `KICK`, and `YOUTUBE_LIVE`.
2. **Kick Pusher Connector**:
   - Accurately parses Pusher handshake and subscribes to `chatrooms.<id>.v2`.
   - Answers `pusher:ping` with `pusher:pong`.
   - Extracts inline emotes matching `[emote:<id>:<name>]` and formats them into `ChatEmote`.
   - Resolves channel slugs to numeric chatroom IDs.
3. **YouTube Live Connector**:
   - Asynchronously wraps `chat-downloader` to process live streams non-blockingly.
   - Extracts concatenated runs, emojis, and Super Chat amounts.
4. **Twitch Connector Refactoring**:
   - Wraps existing `TwitchIrcParser` under the new `BaseChatConnector` interface with 100% backward compatibility.
5. **Coordinator & CLI Integration**:
   - `LiveStreamCoordinator` dynamically binds to any platform connector via `config.platform`.
   - `streamfusion live tail <channel> --platform [twitch|kick|youtube_live]` CLI flag works properly.
   - JSON-RPC `streamfusion.startLiveTail` handles multi-platform configuration.
6. **Test Suite Coverage**:
   - Complete unit test suite verifying each platform connector with deterministic offline mocks.
   - All 140 existing tests pass + new connector tests pass (100% green).
