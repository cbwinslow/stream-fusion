# StreamFusion: Pipeline Modules Engineering Specification

## 1. Module 1: Ingestion & Demuxing (`stream_fusion.ingest`)

### Component Responsibilities:
1. **Time-Range Download:** Wraps `yt-dlp` using `--download-sections "*START_TIME-END_TIME"` to stream or download only desired segments.
2. **Chat Replay Ingestion:** Calls `TwitchDownloaderCLI` or `chat-downloader` to fetch timestamped chat JSON.
3. **Demuxing:** Invokes `ffmpeg` to produce:
   * Mono 16kHz WAV (`audio.wav`) for transcription.
   * Frame sequence or MP4 (`video.mp4`) for visual processing.

---

## 2. Module 2: Voice Anchor & Reaction Diarization (`stream_fusion.audio`)

### The Problem:
A reaction streamer like Asmongold speaks into a Shure SM7B broadcast mic, but the VOD's audio track mixes his mic and the YouTube video's audio into a single stereo channel. Standard Whisper transcribes both without knowing who said what.

### The Algorithm:
1. **Transcription (`faster-whisper`):**
   * Produces timed segments: $[t_{start}, t_{end}, \text{transcript}]$.
2. **Diarization (`pyannote.audio`):**
   * Clusters speaker voice embeddings into distinct labels: `SPEAKER_00`, `SPEAKER_01`.
3. **Acoustic Anchor Profiling:**
   * Streamer audio has distinct physical acoustic properties:
     * High proximity effect (low-frequency boost at 100–250 Hz).
     * High signal-to-noise ratio (SNR) compared to external video audio.
     * Mono spatial centering.
   * StreamFusion compares each speaker's embedding to a calibrated **Streamer Profile**.
   * The speaker matching the profile is tagged `STREAMER`; all other speech is tagged `EXTERNAL_VIDEO`.

---

## 3. Module 3: Adaptive Scene Keyframing & Vision (`stream_fusion.vision`)

### The Problem:
Running a Vision-Language Model on 60 FPS video requires processing 216,000 images per hour, which is too slow and wasteful for static stream layouts.

### The Algorithm:
1. **Scene Transition Detection (`PySceneDetect`):**
   * Uses ContentDetector (threshold: 27.0).
   * Identifies when:
     * A video is started or paused.
     * Streamer switches from game to browser or fullscreen camera.
     * In-between static scenes, samples at a fallback rate of 1 frame every 3.0 seconds.
2. **Multi-Task Prompting with Microsoft Florence-2:**
   * Florence-2 supports task-specific prefix prompts:
     * `<MORE_DETAILED_CAPTION>` $\rightarrow$ Produces dense description of stream layout, active application, and streamer expression.
     * `<OCR>` $\rightarrow$ Extracts video title, browser URL, and chat text on screen.
3. **Structured Extraction:**
   * Extracts visual state into `VisualKeyframe` objects.

---

## 4. Module 4: Dynamic Broadcast Latency Calibration (`stream_fusion.chat`)

### The Problem:
Twitch broadcast latency is not constant; it varies between 2.5 and 8.0 seconds depending on ingest server, viewer buffer, and ISP routing. Hardcoding a static delay causes misalignment between video events and chat spikes.

### The Cross-Correlation Algorithm:
1. **Signal 1 ($S_{trigger}(t)$):** Audio energy / sudden speech bursts from the streamer (e.g., streamer laughing or shouting).
2. **Signal 2 ($S_{chat}(t)$):** Normalized chat message count per second.
3. **Cross-Correlation:**
   $$\Delta t = \arg\max_{\tau \in [2.0, 9.0]} \int S_{trigger}(t) \cdot S_{chat}(t + \tau) \, dt$$
4. The peak lag $\Delta t$ is computed via `scipy.signal.correlate`.
5. All chat timestamps are then aligned: $T_{aligned} = T_{chat} - \Delta t$.

---

## 5. Module 5: Multimodal Temporal Fusion Engine (`stream_fusion.fusion`)

### The Fusion Matrix:
Discretizes the timeline into uniform time buckets $\Delta B$ (default 2.0s):

$$B_k = [k \cdot \Delta B, \, (k + 1) \cdot \Delta B)$$

For each bucket $B_k$, the engine binds:
* $A_{streamer}(B_k)$: Words spoken by streamer.
* $A_{video}(B_k)$: Words spoken in external video.
* $V(B_k)$: Current scene type, dense caption, and visible OCR text.
* $C(B_k)$: Calibrated chat message velocity, top emotes, and sentiment polarity $P \in [-1.0, 1.0]$.

### Highlight / Spike Detection Formula:
A slice is flagged as a **Highlight Candidate** if:

$$\text{Score}(B_k) = w_1 \cdot \left(\frac{\text{Velocity}(B_k)}{\overline{\text{Velocity}}}\right) + w_2 \cdot |\text{Sentiment}(B_k)| + w_3 \cdot \mathbb{I}(\text{Streamer Spoke})$$

Exceeds the dynamic threshold $\Theta_{highlight} = 3.5 \times \text{baseline}$.
