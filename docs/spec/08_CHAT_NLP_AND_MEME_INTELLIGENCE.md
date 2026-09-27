# StreamFusion: Chat NLP, Meme Burst & Community Intelligence Specification (Spec 08)

## 1. Overview & Objectives
Twitch and live streaming chats are not standard natural language; they are high-velocity, emote-dense, semiotic subcultures. Standard off-the-shelf sentiment analyzers (e.g. VADER or generic BERT) misclassify sarcastic chat phrases (e.g. "SURELY", "Aware", "KEKW") as neutral or positive.

This specification details:
1. **Fine-Grained Emotional Intent Classification** tuned for streaming culture (`AMUSEMENT`, `HYPE`, `DISBELIEF`, `DISGUST`, `AGREEMENT`, `QUESTION`, `HEART_WARM`).
2. **Copy-Pasta & Meme Burst Tracker**: Sliding-window Jaccard similarity and token n-gram burst detection to capture emergent community memes.
3. **Chatter Graph & Influencer Ranking**: Quantifying "Opinion Leader" chatters whose messages trigger chat copy-pastas or streamer verbal acknowledgments.

---

## 2. Emotional Intent Taxonomy & Classification Architecture

### 2.1 Intent Classes
| Class | Core Emotes & Triggers | Valence / Polarity | Semantics |
|---|---|---|---|
| `AMUSEMENT` | LUL, LULW, KEKW, OMEGALUL, ICANT, lmao, dying | +0.8 | Streamer or video did something hilarious |
| `HYPE` | Pog, PogBones, POGGERS, letsgo, W, CLIP IT | +0.9 | High excitement, clutch play, epic moment |
| `DISBELIEF` | HUH, cap, fake, no shot, ???, Ain'tNoway, SURELY | -0.3 | Skepticism, doubt, unexpected plot twist |
| `DISGUST_CRINGE` | cringe, weirdchamp, Aware, DESPAIR, WutFace, DansGame | -0.7 | Awkwardness, gross scene, painful cringe |
| `AGREEMENT` | TRUE, based, facts, real, Gigachad, +1, so true | +0.6 | Chat consensus agreeing with streamer's take |
| `HEART_WARM` | BibleThump, FeelsStrongMan, widepeepoHappy, <3, wholesome | +0.9 | Emotional connection, genuine respect/nostalgia |
| `QUESTION` | ?, why, what, who, how, explain | 0.0 | Chat seeking clarification or lore |

### 2.2 Rule-Enhanced Hybrid Intent Classifier
A fast zero-GPU or low-latency hybrid classifier combining:
- Normalized token pattern matching (accounting for elongated words like `looool`, `poggg`)
- Emote dictionary lookups
- Weighted intent vector returning probabilities for each intent class.

---

## 3. Copy-Pasta & Viral Meme Burst Tracker

### 3.1 Mathematical Definition
For any two chat messages $m_i, m_j$ within a temporal window $W$ (e.g. $\tau = 10\text{s}$):
$$\text{Jaccard}(m_i, m_j) = \frac{|T(m_i) \cap T(m_j)|}{|T(m_i) \cup T(m_j)|}$$
where $T(m)$ is the set of word tokens in message $m$.

### 3.2 Burst Detection Algorithm
1. **Window Segmentation:** Partition messages into sliding 10-second windows with 5-second overlap.
2. **Spam Density Clustering:**
   - If Jaccard similarity $\ge 0.65$ between $\ge K$ distinct authors ($K \ge 5$), cluster into a **Meme Propagation Event**.
3. **Meme Metrics:**
   - `origin_message`: First observed occurrence.
   - `burst_start_sec`, `burst_peak_sec`, `burst_end_sec`.
   - `propagation_velocity`: Messages per second of this specific meme.
   - `unique_spreaders`: Count of distinct chatters echoing the copy-pasta.
   - `representative_text`: Canonical string representing the meme.

---

## 4. Chatter Graph & "Opinion Leader" Scoring

### 4.1 Influence Score Formula
$$\text{Influence}(u) = \alpha \cdot \text{FirstMemeOriginCount}(u) + \beta \cdot \text{StreamerResponseCount}(u) + \gamma \cdot \log(1 + \text{TotalMessages}(u))$$

Where:
- $\text{FirstMemeOriginCount}(u)$: Number of times chatter $u$ sent the earliest message of a subsequent copy-pasta burst.
- $\text{StreamerResponseCount}(u)$: Occurrences where the streamer transcribed audio matches chatter $u$'s message within a 3–8 second window.
- Weights: $\alpha = 0.5, \beta = 0.4, \gamma = 0.1$.

---

## 5. Deliverables & Acceptance Criteria
- Module: `src/stream_fusion/chat/nlp.py` (`ChatNLPAnalyzer`, `MemeBurstTracker`, `ChatterInfluenceScorer`).
- Data contracts: `MemeBurstEvent`, `ChatterProfile`, `ChatIntentDistribution`.
- Unit tests: `tests/test_chat_nlp.py` with 100% test coverage.
