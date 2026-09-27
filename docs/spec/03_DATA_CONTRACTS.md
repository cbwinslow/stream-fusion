# StreamFusion: Data Contracts & Schemas

All internal communication and serialization between pipeline stages MUST conform to these Pydantic schema contracts.

## 1. Chat Event Schema (`ChatMessage`)
```python
from pydantic import BaseModel, Field
from typing import List, Optional

class ChatBadge(BaseModel):
    name: str
    version: str

class ChatEmote(BaseModel):
    id: str
    name: str
    count: int = 1

class ChatMessage(BaseModel):
    message_id: str
    timestamp_offset: float = Field(..., description="Seconds from stream start")
    user_id: str
    author_name: str
    content: str
    emotes: List[ChatEmote] = Field(default_factory=list)
    badges: List[str] = Field(default_factory=list)
```

## 2. Audio Segment Schema (`AudioSegment`)
```python
from pydantic import BaseModel, Field
from typing import List, Optional

class WordTiming(BaseModel):
    word: str
    start: float
    end: float
    probability: float

class AudioSegment(BaseModel):
    segment_id: int
    start_sec: float
    end_sec: float
    speaker_label: str = Field(..., description="e.g. 'SPEAKER_STREAMER', 'SPEAKER_VIDEO_EXT', 'UNKNOWN'")
    transcript: str
    confidence: float
    words: Optional[List[WordTiming]] = None
```

## 3. Visual Frame Schema (`VisualKeyframe`)
```python
from pydantic import BaseModel, Field
from typing import List, Optional

class BoundingBox(BaseModel):
    label: str
    confidence: float
    box: List[float]  # [ymin, xmin, ymax, xmax] normalized

class VisualKeyframe(BaseModel):
    frame_index: int
    timestamp_sec: float
    scene_type: str = Field(..., description="'GAMEPLAY', 'REACT_VIDEO', 'FULLSCREEN_CAM', 'BROWSER', 'UNKNOWN'")
    screen_summary: str = Field(..., description="Dense caption of the on-screen event")
    ocr_text_blocks: List[str] = Field(default_factory=list)
    streamer_facial_expression: Optional[str] = None  # e.g., 'laughing', 'shocked', 'neutral'
    detected_objects: List[BoundingBox] = Field(default_factory=list)
```

## 4. Aligned Fusion Matrix Slice (`FusionSlice`)
The primary analytical unit bucketed by time interval (e.g., 2.0-second slice).

```python
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class FusionSlice(BaseModel):
    bucket_index: int
    start_sec: float
    end_sec: float
    
    # Audio State
    active_speakers: List[str]
    streamer_transcript: Optional[str] = None
    external_audio_transcript: Optional[str] = None
    
    # Visual State
    active_scene_type: str
    visual_description: str
    screen_ocr: List[str]
    
    # Chat State (Calibrated for latency)
    chat_message_count: int
    chat_velocity_per_sec: float
    dominant_emotes: Dict[str, int]
    chat_sentiment_polarity: float = Field(..., description="-1.0 (very negative) to +1.0 (very positive)")
    
    # Derived Signals
    is_spike_moment: bool = False
    agreement_score: Optional[float] = None  # Measure of chat agreeing with streamer statement

    # Advanced Multi-Modal Fields (Specs 08 - 14)
    speaker_identities: List[str] = Field(default_factory=list, description="Resolved creator IDs, e.g. ['STREAMER:theburntpeanut', 'CO_STREAMER:hutchmf']")
    chat_intents: Dict[str, float] = Field(default_factory=dict, description="Fine-grained emotion intents from Spec 08")
    active_sponsor_brand: Optional[str] = Field(None, description="Active sponsor detected in this window (Spec 09)")
    griefer_messages_flagged: int = Field(default=0, description="Count of bad-faith contrarian/griefer messages (Spec 12)")
    active_domain: Optional[str] = Field(None, description="Active web domain being browsed, e.g. 'x.com' (Spec 14)")
    read_along_text: Optional[str] = Field(None, description="On-screen text being read aloud by streamer (Spec 14)")
```

---

## 5. Master Contract Matrix (Specs 08–14)

| Domain | Specification | Primary Schemas | Core Responsibility |
|---|---|---|---|
| **Chat Intelligence** | Spec 08 | `ChatIntentDistribution`, `MemeBurstEvent`, `ChatterProfile` | Culture-aware intents, sliding Jaccard meme bursts, opinion leaders |
| **Sponsorship** | Spec 09 | `BrandProfile`, `SponsorSegment`, `SponsorImpactReport` | Audio + OCR brand matching, backlash index, Attention Score |
| **Resumption** | Spec 10 | `ChunkManifest` | Atomic slice caching, zero-redundant-compute resumption |
| **Voice Biometrics** | Spec 11 | `VoiceprintProfile`, `CoStreamInteraction`, `SpeakerMatchResult` | 192-d/512-d creator d-vectors, cross-channel co-streamer matching |
| **Chat Safety** | Spec 12 | `ChatterDetailedProfile`, `ChatterSafetyVerdict`, `BrigadeCluster` | Historical profiling, banter vs. griefing disambiguation, brigade alerts |
| **Knowledge Graph**| Spec 13 | `StreamerClaim`, `EntityStanceRecord`, `ClaimRevision` | Additive belief synthesis, ChromaDB/Qdrant vector index, VOD query |
| **Dense Vision** | Spec 14 | `ScreenWebContext`, `ScreenGameContext`, `ReadAlongSegment` | 1.0s dense frame parsing, X/Twitter cards, HUD extraction, read-along sync |

