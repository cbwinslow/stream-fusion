# StreamFusion: Advanced Feature Expansions Specification

## 1. Dynamic Rolling Latency Delta ($\Delta t(t)$) & Manual Override

### The Problem:
On an 8-hour stream, Twitch broadcast latency is **not static**. Due to TCP re-transmits, viewer video player buffering, and dynamic transcoder adjustments, the broadcast lag $\Delta t$ can drift between 3.2s and 7.8s over the course of a broadcast. A single static offset causes temporal misalignment in later hours.

### The Specification:
1. **CLI Adjustable Controls:**
   * `--auto-latency`: Enables dynamic cross-correlation auto-calibration (default).
   * `--latency-offset <float>`: Manually forces a fixed offset (e.g. `--latency-offset 4.2`), bypassing auto-tuning.
   * `--latency-search-range <min,max>`: Bounds the search window (default: `2.0,9.0`).
2. **Rolling Window Cross-Correlation ($\Delta t(t)$):**
   * Computes discrete cross-correlation in rolling 5-minute sliding windows with 2.5-minute overlap.
   * Produces a continuous, smoothed latency curve $\Delta t(t)$ using linear spline interpolation.
   * Every chat message is calibrated against its exact temporal anchor:
     $$T_{\text{aligned}} = T_{\text{chat}} - \Delta t(T_{\text{chat}})$$

---

## 2. Stream Chunking & Checkpoint-Resume Engine

### The Problem:
An 8-hour 1080p stream is 15–25 GB. If processing fails at hour 7 due to an OS reboot or disk pressure, re-running from scratch wastes hours of compute.

### The Specification:
1. **Time-Slice Chunking:**
   * Streams are partitioned into uniform virtual chunks (default: 10 minutes / 600s).
   * Ingestion demuxes audio and video chunks incrementally using `yt-dlp --download-sections`.
2. **Stateful Checkpointing:**
   * Intermediate representations are saved per chunk:
     * `cache/{stream_id}/chunk_001_audio.parquet`
     * `cache/{stream_id}/chunk_001_vision.parquet`
     * `cache/{stream_id}/chunk_001_chat.parquet`
   * On failure or restart, StreamFusion queries the chunk cache and resumes from the last uncompleted chunk.
3. **Progressive Streaming HTML Reports:**
   * HTML report updates progressively as each chunk finishes, allowing editors to review early stream segments while later segments are still processing.

---

## 3. Deep Visual Multi-Task Parsing (Object Detection & Dense Region Captioning)

### The Specification:
Leveraging Microsoft Florence-2's unified sequence-to-sequence multi-task capabilities:

```
                                  FLORENCE-2 PROMPT SUITE
                                             │
      ┌──────────────────────┬───────────────┴───────────────┬──────────────────────┐
      ▼                      ▼                               ▼                      ▼
<MORE_DETAILED_CAPTION>    <OCR>                           <OD>           <DENSE_REGION_CAPTION>
Dense macro scene      Screen text, titles,      Detects: face, screen,   Detailed breakdown
description            browser tabs, game UI     monitor, chair, mug      of sub-regions
```

1. **Automatic Facecam Localization via `<OD>`:**
   * Queries Florence-2 with prompt `<OD>`.
   * Filters for `label == "person" | "face"`.
   * The bounding box with the highest temporal motion variance is classified as the **Streamer Facecam Crop**.
   * Eliminates hardcoded crop coordinates in the 9:16 vertical short clipper.
2. **In-Game HUD & Region Parsing:**
   * Isolates game health bars, mini-maps, and kill feeds from background gameplay visuals.

---

## 4. Advanced Chat NLP & Open-Domain Entity Recognition

### The Specification:
1. **Linguistic & Cultural Sentiment Classification:**
   * Categorizes chat messages into fine-grained emotional intents:
     * `AMUSEMENT` (LULW, KEKW, ICANT, dying of laughter)
     * `HYPE` (Pog, POGGERS, let's go, W)
     * `DISBELIEF / SKEPTICISM` (cap, fake, ???, HUH, no shot)
     * `DISGUST / CRINGE` (cringe, weirdchamp, Aware, DESPAIR)
     * `AGREEMENT / VALIDATION` (TRUE, based, facts, real, Gigachad)
2. **Copy-Pasta & Meme Burst Tracker:**
   * Computes Jaccard similarity between incoming chat messages across 10-second windows.
   * Detects spontaneous viral copy-pastas and meme propagation in the community.
3. **Chatter Graph & Influencer Detection:**
   * Identifies "Opinion Leader Chatters": viewers whose messages consistently trigger waves of copy-pastas or streamer vocal call-outs.

---

## 5. Sponsor & Brand Performance Quantifier

### The Specification:
1. **Brand Mention Detection:**
   * Monitors spoken audio and screen OCR for registered brand names (e.g. *Starforge Systems*, *Madrinas*, *Dr. Squatch*, *Elgato*, *Razer*).
2. **Sponsor Impact Window ($t_{\text{sponsor}} \pm 60s$):**
   * Calculates:
     * Chat mention frequency of brand tokens.
     * Sentiment polarity during the sponsor segment.
     * Net audience sentiment delta relative to stream baseline.
3. **Automated Sponsor Report Card:**
   * Exports an executive PDF/HTML report for brand sponsors showing quantitative viewer engagement metrics.

---

## 6. Word-by-Word Animated Karaoke Captions for Shorts

### The Specification:
1. **Word-Level Timing Alignment:**
   * Uses `faster-whisper`'s `WordTiming` timestamps (`word`, `start_sec`, `end_sec`).
2. **Animated ASS / SubRip Subtitle Generation:**
   * Generates Advanced SubStation Alpha (`.ass`) subtitle files with:
     * High-contrast bold yellow/white typography (Impact or Montserrat).
     * Word-by-word karaoke highlight animations.
3. **FFmpeg Filter Injection:**
   * Injects subtitles directly into `VerticalHighlightClipper`:
     `-vf "subtitles=clip.ass:force_style='Fontname=Arial,Fontsize=18,PrimaryColour=&H00FFFF'"`
