# StreamFusion: Maximum Data Extraction Architecture

To extract the **maximum possible semantic, behavioral, and emotional intelligence** from a stream VOD, StreamFusion moves beyond basic speech transcription and message counting into multi-dimensional feature extraction across four unified sensory modalities.

---

## 1. Modality 1: Audio & Prosodic Signal Extraction

Standard transcription only answers *"What was said?"*. StreamFusion extracts the full expressive state:

```
[Raw Audio (16kHz WAV)]
       │
       ├──► 1. Linguistic Layer (faster-whisper)
       │        ├── Word-level timestamps & phoneme alignment
       │        ├── Vocabulary entropy & speech rate (Words Per Minute)
       │        └── Streamer language sentiment (positive, sarcastic, critical)
       │
       ├──► 2. Speaker Diarization Layer (ReactionDiarizer)
       │        ├── Tag: STREAMER (proximity mic SM7B profile) vs EXTERNAL_VIDEO
       │        └── Overlap detection (Streamer interrupting or talking over video)
       │
       └──► 3. Acoustic Prosody & Paralinguistic Signals (NumPy / SciPy)
                ├── RMS Energy / Volume: Identifies shouting, laughter bursts, or whispering
                ├── Pitch / F0 Contour: Tracks emotional inflection, excitement, and agitation
                └── Dramatic Silence / Pauses: Tracks abrupt speech drop-offs right after video statements
```

---

## 2. Modality 2: Visual Screen & Streamer Facecam Extraction

Streams combine multiple sub-screens (e.g. Asmon's facecam in the corner, a YouTube video in a browser, and game UI):

```
[Keyframe Image (sampled at scene cuts)]
       │
       ├──► 1. Screen OCR & Text Extraction (Florence-2 <OCR>)
       │        ├── YouTube video title, channel name, view counts
       │        ├── Browser tab titles, Reddit post headers, Twitter/X usernames
       │        ├── Video player scrub-bar timestamps ("Paused at 04:15")
       │        └── In-game kill feeds, boss names, health bars, inventory text
       │
       ├──► 2. Streamer Facecam Dynamics
       │        ├── Face bounding box localization
       │        ├── Facial expression classification: [Laughing, Disbelief, Smirking, Angry, Neutral]
       │        ├── Gaze direction: Looking directly at camera (talking to chat) vs monitor (watching video)
       │        └── Physical posture: Leaning forward (high intensity), facepalm, head-shaking
       │
       └──► 3. Macro Scene Segmentation
                └── Class: [REACT_VIDEO, GAMEPLAY, FULLSCREEN_CAM, BROWSER_REDDIT, AFK]
```

---

## 3. Modality 3: Audience Chat & Behavioral Dynamics

Twitch chat is not uniform text; it is an organic emotional crowd sensor:

```
[Chat Replay Stream (TwitchDownloader)]
       │
       ├──► 1. Message Dynamics
       │        ├── Velocity: Messages per second (density)
       │        ├── Acceleration: Second-derivative burstiness (sudden collective shock)
       │        └── Chatter Entropy: Ratio of unique users to total messages (organic vs copy-pasta spam)
       │
       ├──► 2. Sub-Community Emote Lexicon (Twitch + 7TV + BTTV + FFZ)
       │        ├── Amusement / Hype: OMEGALUL, KEKW, ICANT, Pog, W, Clap
       │        ├── Disagreement / Skepticism: L, cap, NOPERS, cringe, weirdchamp
       │        ├── Shock / Dread: monkaW, Aware, DESPAIR, ???, HUH
       │        └── Agreement / Based: TRUE, NODDERS, based, Gigachad
       │
       └──► 3. Chatter Authority Hierarchy
                ├── Broadcaster / Moderator messages (editorial anchors)
                ├── VIP / Subscriber reactions (committed audience sentiment)
                └── General crowd velocity
```

---

## 4. Modality 4: Multimodal Grounding Matrix & Synthetic Metrics

When all streams are aligned with latency calibration ($T_{event} = T_{chat} - \Delta t$), StreamFusion computes three high-value composite metrics:

### Metric A: The Take Agreement Index ($\mathcal{A}$)
When the streamer pauses a video and delivers a statement:
$$\mathcal{A} = \frac{\sum \text{Agreement Tokens (TRUE, W, based)} - \sum \text{Dissent Tokens (L, cap, ?)}}{\text{Total Evaluated Tokens}}$$
* $\mathcal{A} > +0.5 \rightarrow$ High Audience Alignment ("Chat is with him").
* $\mathcal{A} < -0.2 \rightarrow$ Audience Revolt ("Chat is roasting him / calling cap").

### Metric B: Reaction Trigger Attribution
Automatically pairs:
* **The Trigger:** (e.g. YouTube video says *"crafting materials will be in the cash shop"*)
* **The Action:** (Streamer hits pause, leans forward, eyes wide)
* **The Take:** (Streamer says *"This is completely cooked"*)
* **The Crowd Verdict:** (Chat velocity spikes 6x with 85% `OMEGALUL / COOKED`)

### Metric C: Automated Viral Highlight Score ($\mathcal{H}$)
$$\mathcal{H}(t) = w_{vel} \cdot \left(\frac{\text{Velocity}(t)}{\overline{\text{Velocity}}}\right) + w_{prosody} \cdot \text{RMS}_{streamer}(t) + w_{face} \cdot \mathbb{I}(\text{Expression Change}) + w_{pause} \cdot \mathbb{I}(\text{Video Paused})$$

Segments exceeding the dynamic threshold are automatically extracted into a **YouTube Shorts Cut-List** with bounding box crop coordinates.
