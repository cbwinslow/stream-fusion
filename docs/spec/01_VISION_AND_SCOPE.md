# StreamFusion: Vision & Scope Specification

## 1. Executive Summary
**StreamFusion** is an open-source, modular multimodal analysis framework designed to ingest livestream video recordings (VODs) alongside synchronized chat replay streams (Twitch, YouTube). 

By synchronizing and temporally aligning:
1. **Audio Transcription & Speaker Diarization** (separating streamer mic speech from external react video/game audio)
2. **Visual Scene & Screen Understanding** (OCR for titles/text + VLM scene description of games/web/react windows)
3. **Audience Chat Replay & Emote Dynamics** (message velocity, sentiment scoring, emote distributions, calibrated for broadcast latency)

StreamFusion constructs a unified **Multimodal Temporal Matrix** that explains *what happened on screen*, *what was said*, and *how the audience reacted*.

---

## 2. Target Personas & Use Cases

### A. Stream Editors & Content Creators
* **The Problem:** Editors spend 4–8 hours manually scrubbing full streams to find highlights, react segments, and funny chat moments.
* **The Solution:** Automated detection of react segments, take vs. chat agreement index, and automated cut-point generation for YouTube Shorts and TikToks.

### B. Game Studios & Marketing Agencies
* **The Problem:** Agencies manually watch sponsored streams or gameplay trailers to write subjective sentiment reports.
* **The Solution:** Objective second-by-second sentiment and attention audit during gameplay or reveal trailers.

### C. Multimodal AI & Dataset Researchers
* **The Problem:** Scarcity of real-world datasets that pair continuous video/speech with dense, real-time human emotional feedback.
* **The Solution:** Extraction of time-aligned triples: `(Visual State, Audio Dialogue) -> Audience Reaction Distribution`.

---

## 3. Core Principles & Philosophy
* **Offline-First & Local-First:** Optimized for consumer hardware (e.g. single 12GB RTX 3060) via sequential batch inference rather than expensive cloud servers.
* **Modular Provider Architecture:** Swappable backends for ASR (Faster-Whisper, WhisperX, remote APIs), Vision (Florence-2, Moondream, Qwen2-VL, Gemini), and Storage (Local, SMB, S3/MinIO).
* **Strict Schema Contracts:** All intermediate and final data structures adhere to strict Pydantic schemas, serializable to Parquet, JSONL, or SQLite.
* **Broadcast Latency Calibration:** Native support for modeling and compensating the 3–8 second viewer-to-streamer latency offset.
