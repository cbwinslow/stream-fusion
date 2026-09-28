# StreamFusion: Self-Expanding Adaptive Slang & Meme Engine (Spec 17)

## 1. North Star & System Objectives
Live streaming communities generate new memes, 7TV/BTTV emotes, and linguistic slang daily (e.g. `catJAM`, `KEKW`, `DESPAIR`, `Bedge`, `skull emoji`, `ICANT`). Hardcoded dictionaries inevitably decay and fail to understand emerging internet subcultures. 

The goal of **Spec 17** is to construct an unsupervised, self-expanding adaptive slang and meme engine that:
1. Detects emerging slang and viral emotes in real time using statistical burst detection ($\ge 3\sigma$ spike over rolling background distributions).
2. Contextually infers the emotional polarity, intent category, and valence by analyzing co-occurring chat messages and acoustic prosody (streamer laughter/screaming).
3. Discovers novel semantic clusters for conversational intents that do not fit the canonical 7 intents.
4. Maintains an evolving `adaptive_lexicon.json` with half-life temporal confidence decay, automatically promoting stable terms and pruning ephemeral fads.

---

## 2. Mathematical Formulations & Statistical Engine

### 2.1 Unsupervised Rolling Burst Velocity ($\ge 3\sigma$)
Let chat messages arrive over stream time $t$. We partition the stream into rolling temporal windows $W_k = [t_k, t_k + \Delta t]$ (e.g. $\Delta t = 10\text{s}$).

For each candidate token / emote $w$, we track:
- **Instantaneous Velocity**: $V_k(w) = \frac{N_k(w)}{\Delta t}$, where $N_k(w)$ is the count of $w$ in window $W_k$.
- **Background Rolling Mean**: $\mu_{k}(w) = \alpha V_k(w) + (1 - \alpha) \mu_{k-1}(w)$.
- **Background Variance**: $\sigma_k^2(w) = \alpha (V_k(w) - \mu_k(w))^2 + (1 - \alpha) \sigma_{k-1}^2(w)$.
- **Standardized Burst Z-Score**:
  $$Z_k(w) = \frac{V_k(w) - \mu_{k-1}(w)}{\sigma_{k-1}(w) + \epsilon}$$

A token $w$ triggers a **Burst Event** if:
$$Z_k(w) \ge 3.0 \quad \text{and} \quad N_k(w) \ge N_{\text{min}} \quad \text{and} \quad U_k(w) \ge U_{\text{min}}$$
where $U_k(w)$ is the number of distinct chatters propagating the token.

### 2.2 Contextual Intent & Valence Inference
When token $w$ bursts in window $W_k$, we collect the co-occurring messages $M_k(w) = \{ m \in W_k \mid w \in m \}$.
For each message $m_i \in M_k(w)$:
1. Strip the target token $w$ to observe surrounding context $m_i \setminus \{w\}$.
2. Evaluate existing canonical intent scores $S_{\text{canon}}(m_i \setminus \{w\})$.
3. Co-occurring intent probability:
   $$P(I \mid w) = \frac{\sum_{m_i} S_{\text{canon}}(m_i \setminus \{w\})[I]}{\sum_{I'} \sum_{m_i} S_{\text{canon}}(m_i \setminus \{w\})[I']}$$
4. Inferred Valence:
   $$V(w) = \sum_{I} P(I \mid w) \cdot \text{Valence}(I)$$

### 2.3 Acoustic Prosody Grounding
If the streamer's vocal energy or pitch spikes during the burst window:
$$\text{AcousticBoost}(w) = \max_{s \in \text{Audio}(W_k)} \left( \frac{\text{RMS}(s) - \mu_{\text{RMS}}}{\sigma_{\text{RMS}}} \right)$$
If $\text{AcousticBoost} > 1.5$ and co-occurring chat expresses amusement/laughter, the classification confidence increases:
$$\text{Confidence}(w) = \min(1.0, \text{Confidence}_{\text{chat}} + 0.15 \cdot \mathbf{1}_{\{\text{AcousticBoost} > 1.5\}})$$

### 2.4 Exponential Half-Life Temporal Decay
Memes fade over time. For an adaptive lexicon entry with last observation $t_{\text{last}}$ and query time $t$:
$$C(t) = C_0 \cdot 2^{-\frac{t - t_{\text{last}}}{t_{1/2}}}$$
where $t_{1/2}$ is the half-life parameter (default: 14 days).
- **Promotion Threshold**: $C(t) \ge 0.80$ with $\ge 20$ occurrences $\rightarrow$ `PROMOTED` / `ACTIVE`.
- **Pruning Threshold**: $C(t) < 0.20$ after $\ge 30$ days without activity $\rightarrow$ `PRUNED`.

---

## 3. Architecture & Data Contracts

```
Chat Stream (Messages + Emotes)
  │
  ├──► Token Extraction & Normalization
  │      └── Strip punctuation, collapse repeated chars (loooool -> lol)
  │
  ├──► Rolling Window Frequency Accumulator
  │      └── Mean (μ), StdDev (σ), Z-Score Evaluation
  │
  ├──► Z-Score Filter (Z >= 3.0) ──► Candidate Identified
  │                                    │
  ├──► Acoustic Context Collector ◄────┤
  │      └── AudioProsodyAnalyzer      │
  │                                    ▼
  ├──► Contextual Auto-Tagger ─────────► Inferred Intent & Valence
  │                                    │
  ├──► Novel Cluster Detection (Centroids)
  │                                    │
  └──► Persistent AdaptiveLexicon Store (adaptive_lexicon.json)
         ├── Confidence Update & Temporal Decay
         └── Integrated with ChatNLPAnalyzer
```

### 3.1 Data Schema: `AdaptiveTermEntry`
```json
{
  "term": "catjam",
  "inferred_intent": "HYPE",
  "valence": 0.85,
  "confidence": 0.92,
  "first_seen_timestamp": "2026-09-28T07:00:00Z",
  "last_seen_timestamp": "2026-09-28T07:20:00Z",
  "occurrence_count": 142,
  "unique_authors_count": 68,
  "peak_z_score": 5.4,
  "decay_half_life_days": 14.0,
  "status": "ACTIVE"
}
```

---

## 4. CLI & Agent Integration
- `streamfusion slang scan --chat <chat.json> [--audio <audio.wav>] [--out <adaptive_lexicon.json>]`: Scans chat stream for emerging slang and updates lexicon.
- `streamfusion slang list [--lexicon <path>]`: Displays active, candidate, and promoted slang terms.
- `streamfusion slang prune [--lexicon <path>]`: Prunes decayed slang terms.
- JSON-RPC agent method: `streamfusion.querySlang` for AI agents to query the latest vernacular.
