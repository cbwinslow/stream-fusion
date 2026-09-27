# StreamFusion: Homelab & Deployment Architecture

## 1. Hardware Strategy: Homelab vs Workstation

```
+-------------------------------------------------------------+
|                     ZeroTier Virtual LAN                    |
+-------------------------------------------------------------+
               │                                      │
               ▼                                      ▼
+-----------------------------+        +------------------------------+
|     Homelab Server (NAS)    |        |  Windows Workstation (Local) |
|                             |        |                              |
| Role: Storage & Central DB  |        | Role: GPU Compute Worker     |
|                             |        |                              |
| • 10-50TB Array (ZFS/RAID)  |        | • NVIDIA RTX 3060 (12GB)     |
| • SMB / NFS / MinIO (S3)    |◄───────┤ • WhisperX / Faster-Whisper  |
| • PostgreSQL / SQLite Sync  |  Sync  | • Florence-2 / Local VLM     |
| • K40 / K80 (Legacy role)   |        | • ZeroTier Remote IP Access  |
+-----------------------------+        +------------------------------+
```

---

## 2. Realistic Role for K40 / K80 GPUs
* **Technical Constraints:**
  * NVIDIA Kepler architecture (Compute Capability 3.5 / 3.7).
  * Dropped by PyTorch >= 1.11, CUDA 11.5+, CUDA 12, FlashAttention, and modern Hugging Face libraries.
  * No Tensor Cores; lacks FP16 / BF16 acceleration and modern 4-bit/8-bit quantization.
* **Appropriate Tasks for Homelab Server:**
  1. **Video Storage & Caching:** Full stream VODs take 5–15 GB each. Storing archive VODs on the homelab prevents filling your workstation NVMe.
  2. **Job Queue & Ingestion:** The homelab can run lightweight background scripts (using `yt-dlp` and `chat-downloader`) to fetch upcoming streams while your workstation is busy or offline.
  3. **Central Analytics Database:** Hosts SQLite/PostgreSQL and Parquet data lakes accessible over the network.
* **Workstation (RTX 3060):**
  * Dedicated compute worker. Pulls raw media from homelab over ZeroTier, runs high-speed inference, and uploads processed metadata back to the homelab.

---

## 3. ZeroTier Integration & Remote Access

### Network Topology
1. Both machines join the private ZeroTier network (e.g. `10.147.17.x`).
2. StreamFusion configuration allows setting remote storage paths:
   ```yaml
   storage:
     backend: "local"  # or "smb", "s3"
     media_dir: "\\\\10.147.17.5\\streams\\vods"
     output_dir: "\\\\10.147.17.5\\streams\\output"
   ```
3. Optional Ollama API endpoint:
   If Ollama is hosted on either machine, StreamFusion's VLM module can make standard HTTP calls across ZeroTier (`http://10.147.17.5:11434/api/generate`).
