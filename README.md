# StreamFusion 🎙️📹💬

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Architecture: Spec-Driven](https://img.shields.io/badge/Architecture-Spec--Driven-green.svg)](docs/spec/01_VISION_AND_SCOPE.md)

**StreamFusion** is an open-source, modular multimodal analysis framework for livestream VODs and chat replay streams (Twitch, YouTube).

It reconstructs what happened on stream by synchronizing **Speaker-Diarized Audio**, **Screen & Visual Context**, and **Audience Chat Reactions** into a unified temporal matrix.

```
                  ┌─► Audio Extraction ──► WhisperX / Faster-Whisper (Diarization) ──┐
                  │                                                                  │
Video VOD (yt-dlp)┼─► Scene/Keyframe ─────► Florence-2 / Qwen2-VL / Screen OCR ──────┼─► Time-Series Fusion Matrix
                  │                                                                  │   (Pandas / Parquet / HTML)
                  └─► Chat Replay (JSON) ─► Latency Offset & Emote Sentiment ────────┘
```

---

## Key Features

1. **Reaction Diarization:** Distinguishes between the streamer speaking directly into their microphone vs. external audio from videos/games playing on stream.
2. **Visual Screen Understanding:** Uses compact Vision-Language Models (Florence-2 / Moondream2) and OCR to describe active windows, games, and streamer facial expressions.
3. **Calibrated Chat Grounding:** Accounts for the 3–8 second broadcast latency delay, accurately pairing audience emote spikes (`OMEGALUL`, `Pog`, `W`, `L`, `???`) with the precise triggering visual/audio event.
4. **Hardware Optimized for Consumer GPUs:** Sequentially executes stages to run comfortably on a single **NVIDIA RTX 3060 (12GB)** without out-of-memory errors.
5. **Homelab Ready:** Decoupled storage and compute so raw video files can reside on a homelab NAS over ZeroTier or SMB, while compute is executed on a dedicated GPU worker.

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/your-username/stream-fusion.git
cd stream-fusion
pip install -e .
```

To install GPU acceleration and audio/vision dependencies:
```bash
pip install -e ".[audio,vision,ingest]"
```

### 2. Basic CLI Usage

#### Process a local or remote stream VOD:
```bash
# Process a 3-minute sample segment with chat alignment
streamfusion analyze "https://www.twitch.tv/videos/12345678" --duration 180 --out ./output
```

#### Run on a pre-downloaded video and chat replay:
```bash
streamfusion fuse --video sample_vod.mp4 --chat sample_chat.json --out ./report.html
```

---

## Specifications & Documentation

The project is built following **Spec-Driven Development (SDD)**:

* [01. Vision & Scope](docs/spec/01_VISION_AND_SCOPE.md)
* [02. System Architecture](docs/spec/02_ARCHITECTURE.md)
* [03. Data Contracts & Schemas](docs/spec/03_DATA_CONTRACTS.md)
* [04. Homelab & Deployment Guide](docs/spec/04_HOMELAB_AND_DEPLOYMENT.md)

---

## License

MIT License. See [LICENSE](LICENSE) for details.
