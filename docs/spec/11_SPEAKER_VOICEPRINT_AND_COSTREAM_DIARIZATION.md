# StreamFusion: Speaker Voiceprint Library & Cross-Stream Diarization (Spec 11)

## 1. North Star & Objectives
In live streaming broadcasts, speech rarely consists solely of a single isolated voice. Streamers frequently collaborate on Discord, co-stream multiplayer games, host podcasts, and react to third-party videos.

Standard off-the-shelf diarization pipelines label speakers generically (`SPEAKER_00`, `SPEAKER_01`) on a per-VOD basis with no memory across broadcasts. This fails to identify that:
1. `SPEAKER_00` on Channel A is **Asmongold**.
2. A voice appearing on **TheBurntPeanut**'s stream is actually **HutchMF** playing alongside him in a Discord call.
3. Both creators have distinct individual channels, yet their voiceprints persist across broadcasts.

### Core Objectives
1. **Streamer Biometric Voiceprint Enrollment**: Extract and store normalized acoustic speaker embeddings (192-d or 512-d d-vectors via ECAPA-TDNN / PyAnnote) for registered creators.
2. **Cross-Stream Global Speaker Identification**: Map local diarization clusters (`SPEAKER_00`, `SPEAKER_01`) to real-world creator identities (`STREAMER:asmongold`, `CO_STREAMER:hutchmf`, `EXTERNAL_VIDEO`, `UNKNOWN`).
3. **Cross-Channel Network Graph**: Automatically discover collaborative co-streaming relationships, recording who plays with whom, frequency of interactions, and voice overlap statistics.

---

## 2. Voiceprint Biometric Engine

### 2.1 Embedding Representation
For any speech segment $s$, an acoustic encoder extracts an L2-normalized speaker embedding:
$$\vec{v} \in \mathbb{R}^D, \quad \|\vec{v}\|_2 = 1.0$$
where $D = 192$ (SpeechBrain ECAPA-TDNN) or $D = 512$ (PyAnnote Speaker Embedding).

### 2.2 Known Streamer Registry Schema (`VoiceprintProfile`)
```json
{
  "creator_id": "hutchmf",
  "display_name": "HutchMF",
  "primary_channel": "twitch.tv/hutchmf",
  "centroid_embedding": [0.021, -0.054, ..., 0.118],
  "sample_count": 24,
  "confidence_threshold": 0.78,
  "last_updated": "2026-09-27T15:30:00Z"
}
```

### 2.3 Incremental Centroid Online Update
When a new high-confidence sample $\vec{v}_{\text{new}}$ is verified for creator $C$:
$$\vec{\mu}_{\text{new}} = \frac{N \cdot \vec{\mu}_{\text{old}} + \vec{v}_{\text{new}}}{N + 1}, \quad \vec{\mu}_{\text{normalized}} = \frac{\vec{\mu}_{\text{new}}}{\|\vec{\mu}_{\text{new}}\|_2}$$
where $N$ is the historical sample count.

---

## 3. Cross-Stream Matching & Co-Stream Attribution Algorithm

### 3.1 Cosine Verification Formulation
For an unknown diarized speech segment embedding $\vec{v}_{\text{seg}}$ and registered creator centroid $\vec{\mu}_c$:
$$\text{Sim}(\vec{v}_{\text{seg}}, \vec{\mu}_c) = \vec{v}_{\text{seg}} \cdot \vec{\mu}_c = \sum_{k=1}^D v_k \mu_{c,k}$$

### 3.2 Decision Boundaries
| Cosine Similarity Range | Attribution Verdict | Description |
|---|---|---|
| $\text{Sim} \ge 0.82$ | `PRIMARY_STREAMER` / `CO_STREAMER` (Definite Match) | Positive identity confirmed; link to creator profile |
| $0.74 \le \text{Sim} < 0.82$ | `PROBABLE_MATCH` | Flagged for soft attribution; corroborated with chat mention checks |
| $\text{Sim} < 0.74$ | `EXTERNAL_OR_UNKNOWN` | External video, random in-game teammate, or un-enrolled voice |

### 3.3 Channel Context Grounding
1. If $\text{Sim}(\vec{v}, \vec{\mu}_{\text{channel\_owner}}) \ge 0.78$, label as `STREAMER:{owner_id}`.
2. If $\text{Sim}(\vec{v}, \vec{\mu}_{\text{other\_creator}}) \ge 0.78$, label as `CO_STREAMER:{other_creator_id}`.
3. If no match and visual frame has `scene_type == 'REACT_VIDEO'`, label as `EXTERNAL_VIDEO`.

---

## 4. Cross-Stream Collaborative Network

Every co-stream detection generates a network edge:
```json
{
  "host_creator": "theburntpeanut",
  "guest_creator": "hutchmf",
  "vod_id": "vod_994821",
  "interaction_start_sec": 120.0,
  "interaction_end_sec": 2400.0,
  "total_spoken_duration_sec": 640.2,
  "game_or_activity": "DayZ",
  "timestamp": "2026-09-27T15:30:00Z"
}
```

---

## 5. Acceptance Criteria & Deliverables
* **Module**: `src/stream_fusion/audio/voiceprint.py` (`VoiceprintLibrary`, `SpeakerEmbeddingExtractor`, `CoStreamDiarizer`).
* **Data Contracts**: `VoiceprintProfile`, `CoStreamInteraction`, `SpeakerMatchResult`.
* **CLI Command**: `streamfusion voice-enroll <creator_id> <sample_wav>` and `streamfusion costream-scan <vod_path>`.
* **Unit Tests**: `tests/test_voiceprint.py` validating enrollment, similarity calculation, and cross-stream matching.
