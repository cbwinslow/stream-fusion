# StreamFusion: Dense Frame Extraction, Screen OCR & Web/Social Post Grounding (Spec 14)

## 1. North Star & Objectives
Livestreams contain dense visual information: social media posts on X.com (Twitter), Reddit threads, news articles, patch notes, video games with complex HUDs, and web browser windows. Audio transcription alone only captures what the streamer says aloud, omitting critical context such as who wrote the tweet, what comments were visible, or what website was being browsed.

### Core Objectives
1. **Dense Continuous Keyframe Extraction (1.0s Interval)**:
   - Deep multimodal analysis applied uniformly to every sampled second of broadcast video, ensuring zero lost visual context.
2. **Social Media & Web Context Parsing**:
   - Detect and structure X.com/Twitter post cards (author handle, display name, tweet body text, embedded quote tweets).
   - Detect browser URL bars, domains (`x.com`, `reddit.com/r/livestreamfail`, `ign.com`, `youtube.com`), and article headlines.
3. **Gaming HUD & On-Screen UI Extraction**:
   - Localize and parse gaming HUD elements: minimaps, health/mana resource gauges, kill feeds, equipment inventories, and match timers.
4. **Read-Along Audio-Visual Alignment**:
   - Synchronize streamer speech with on-screen text bounding boxes to establish exactly which paragraph, headline, or tweet the streamer is reading aloud in real time.

---

## 2. Social Post & Web Page Extraction Engine

### 2.1 Structured Web Element Contract (`ScreenWebContext`)
```json
{
  "timestamp_sec": 34.0,
  "browser_detected": true,
  "detected_url": "https://x.com/Zackrawrr/status/18392019283",
  "domain": "x.com",
  "active_tab_title": "Zack (@Zackrawrr) on X: 'Servers are finally back up...'",
  "social_post_cards": [
    {
      "platform": "X_TWITTER",
      "author_handle": "@Zackrawrr",
      "author_name": "Zack",
      "post_text": "Servers are finally back up. Jumping on to test the new raid boss.",
      "bounding_box": [0.22, 0.30, 0.65, 0.78],
      "has_embedded_media": false
    }
  ],
  "article_headlines": []
}
```

### 2.2 Game HUD & State Contract (`ScreenGameContext`)
```json
{
  "timestamp_sec": 88.0,
  "game_title_hint": "World of Warcraft",
  "hud_elements": {
    "minimap_box": [0.02, 0.82, 0.28, 0.98],
    "health_percentage": 0.34,
    "mana_percentage": 0.82,
    "killfeed_entries": [
      "Asmongold defeated Mythic Raider (Critical Strike)"
    ],
    "inventory_open": false
  }
}
```

---

## 3. Read-Along Audio-Visual Gaze Alignment

When a streamer reads a tweet, patch note, or Reddit comment on stream:

### 3.1 Mathematical Formulations
Let $W_{\text{speech}} = [w_1, w_2, \dots, w_K]$ be the sequence of spoken words from Whisper with word timings in interval $[t_a, t_b]$.
Let $T_{\text{screen}} = [t_1, t_2, \dots, t_M]$ be the OCR text tokens detected in the corresponding visual keyframe bounding box $B_{\text{post}}$.

The **Read-Along Alignment Score** is:
$$\text{ReadAlongAlignment}(B_{\text{post}}) = \frac{|W_{\text{speech}} \cap T_{\text{screen}}|}{|W_{\text{speech}}|}$$

### 3.2 Sub-Window ROI Scaling Normalization
Streamers rarely display web pages across 100% of the canvas; they commonly position browser windows at 50%–70% scale with their facecam and chat sidebar occupying remaining quadrants. Running full-frame OCR directly on downscaled text causes character recognition degradation.
1. **Window Boundary Localizer**: Detects browser window bounding box $[y_1, x_1, y_2, x_2]$ using Florence-2 `<OD>` (`window`, `display_screen`).
2. **Crop & Super-Resolution Crop Rescaling**: Crops the region of interest (ROI) and applies bicubic upscaling ($2\times$) if character height $< 14\text{ px}$.
3. **Card-Level Text Segmentation**: Distinguishes tweet author handle (`@handle`) from post body, preventing noisy author names from blending into sentence semantics.

---

## 4. Dense Pipeline Architecture

```
Raw Frame (1.0s)
  │
  ├──► Florence-2 (OD / Dense Captioning / Text OCR)
  │      ├── Facecam Bounding Box
  │      ├── Browser Window Bounding Box
  │      └── Full OCR Text & Bounding Boxes
  │
  ├──► Specialized Layout Detectors
  │      ├── Social Card Parser (X/Twitter post bounding boxes)
  │      ├── Browser URL / Domain Bar Parser
  │      └── Game HUD Gauge & Minimap Parser
  │
  └──► Temporal Sync Engine
         └── Correlates Spoken Speech Words with Screen Text Bounding Boxes
```

---

## 5. Acceptance Criteria & Deliverables
* **Module**: `src/stream_fusion/vision/dense_extractor.py` (`DenseFrameExtractor`, `SocialCardParser`, `ReadAlongAligner`).
* **Data Contracts**: `ScreenWebContext`, `ScreenGameContext`, `ReadAlongSegment`.
* **Export Integration**: Store dense frame metadata in Parquet matrix (`dense_frame_metadata` column) and display in HTML Grounding Report.
* **CLI Command**: `streamfusion inspect-frame <image_or_video> --time <seconds>`.
* **Unit Tests**: `tests/test_dense_extractor.py` verifying tweet parsing, URL bar detection, and read-along word-to-box alignment.
