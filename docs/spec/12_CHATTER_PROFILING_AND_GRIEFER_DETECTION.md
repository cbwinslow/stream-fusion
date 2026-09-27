# StreamFusion: Chatter Profiling, Banter vs. Griefing & Community Safety (Spec 12)

## 1. North Star & Objectives
Live streaming communities possess unique communicative norms. Viewers and streamers engage in playful roasting, self-deprecating humor, dark comedy, and competitive banter. Traditional moderation tools fail because they treat all negative sentiment as abuse, banning loyal community members while failing to detect organized, bad-faith griefers who subtly poison chat environments.

### Core Objectives
1. **Persistent Historical Chatter Profiling**: Track all chat activity across multiple streams in a dedicated relational schema (Postgres / SQLite), aggregating message volume, vocabulary patterns, intent distributions, and interaction histories.
2. **Contextual Disambiguation (Banter vs. Griefing)**:
   - Differentiate friendly roasting and gaming banter from malicious harassment, griefing, and hate speech using stream state context.
3. **Quantifiable Chatter Fingerprinting**:
   - Compute mathematical indices for each user: Contrarian Score, Hostility Index, Banter Reciprocity, and Community Alignment.
4. **Automated Griefer & Brigade Alerts**:
   - Surface accounts exhibiting persistent contrarian hostility, chat disruption, or coordinated brigade characteristics.

---

## 2. Contextual Disambiguation Taxonomy

| Message Classification | Contextual Trigger | Example Tokens | Intent & Action |
|---|---|---|---|
| `GOOD_NATURED_BANTER` | Streamer failure, gaming death, funny misplay, co-occurring with chat amusement burst | `OMEGALUL`, `TRASH`, `YOU THREW`, `LMAO`, `washed` | Natural stream engagement. No penalty. |
| `COMMUNITY_ROAST` | Streamer voluntarily invites roasting (e.g. fashion check, messy room, food take) | `cringe`, `bald`, `greasy`, `what are those`, `Aware` | Consensual streamer-chat dynamic. No penalty. |
| `BAD_FAITH_GRIEFING` | Serious discussion, emotional topic, or unprovoked persistent insults designed to provoke an argument | `loser`, `fake`, `unsubbed`, `nobody cares`, `irrelevant` | Flagged: High Contrarian / Disruptive. |
| `MALICIOUS_HARASSMENT` | Explicit hate speech, doxxing threats, targeted slurs, or persistent personal attacks | Direct slurs, personal doxxing, harassment | Immediate Critical Alert / Moderation Ban. |
| `COORDINATED_BRIGADE` | Group of accounts with recent first-seen timestamps echoing identical disruptive narratives | Coordinated talking points, raid spam | Cluster Flagged for Brigade Review. |

---

## 3. Mathematical Behavioral Formulations

### 3.1 Contrarian Index ($C_{\text{contrarian}}$)
Measures the proportion of messages where a chatter takes an adversarial stance against the streamer:
$$C_{\text{contrarian}}(u) = \frac{\sum_{m \in M(u)} \mathbb{I}(\text{Stance}(m) = \text{OPPOSES\_STREAMER})}{|M(u)|}$$

Where:
* $C_{\text{contrarian}}(u) \in [0.0, 1.0]$.
* $C_{\text{contrarian}} > 0.80$ over $N \ge 20$ messages indicates chronic bad-faith contrarianism.

### 3.2 Hostility Index ($H(u)$)
$$H(u) = \frac{\sum_{m \in M(u)} \text{HostilityWeight}(m)}{|M(u)|}$$

### 3.3 Contextual Severity Weighting ($W_{\text{context}}$)
Negative words spoken during high-amusement or gaming-death states receive discounted toxicity:
$$W_{\text{actual}}(m) = W_{\text{raw}}(m) \times (1.0 - 0.7 \cdot \mathbb{I}(\text{SceneState} \in \{\text{GAMEPLAY\_DEATH}, \text{AMUSEMENT\_SPIKE}\}))$$

### 3.4 Banter Reciprocity Score ($R_{\text{banter}}$)
Measures the chatter's participation in positive community events (meme bursts, hype moments):
$$R_{\text{banter}}(u) = \frac{\text{Count}(m \in \text{MemeBursts}) + \text{Count}(\text{Intent} \in \{\text{HYPE}, \text{AMUSEMENT}, \text{HEART\_WARM}\})}{|M(u)|}$$

### 3.5 Griefer Probability ($P_{\text{griefer}}$)
$$P_{\text{griefer}}(u) = \sigma\left(w_1 \cdot C_{\text{contrarian}} + w_2 \cdot H(u) - w_3 \cdot R_{\text{banter}} - w_4 \cdot \log(1 + \text{TenureDays})\right)$$

### 3.6 Automated Bot & System Account Exclusion
Automated channel bots frequently output high-frequency command responses (`!uptime`, `!discord`, `!specs`, `!merch`) and automated timer notifications.
* **Whitelisted Bot Registry**: `nightbot`, `streamelements`, `moobot`, `fossabot`, `soundalerts`, `wizebot`.
* **Automated Heuristic**: Any account sending $> 90\%$ messages starting with `!` or matching regex `^(Welcome to the stream|Follow the channel|Join the discord)` is assigned `is_automated_bot = True` and excluded from chatter moderation profiling.

### 3.7 Coordinated Brigade Temporal Clustering
Coordinated hate raids and political brigades involve accounts entering chat within a narrow time window $\tau_{\text{arrival}} \le 120\text{s}$ exhibiting:
1. High token Jaccard similarity ($\ge 0.70$) on hostile talking points.
2. Zero prior message history on the channel (`TenureDays == 0`).
3. Clustered activation during controversial monologue moments.
When $\ge 5$ such accounts fire simultaneously, the system emits a `BRIGADE_ALERT` payload linking all associated `user_id`s.

---

## 4. Historical Chatter Store Schema (Postgres / SQLite)

```sql
CREATE TABLE chatters (
    user_id VARCHAR(64) PRIMARY KEY,
    username VARCHAR(128) NOT NULL,
    first_seen_at TIMESTAMP NOT NULL,
    last_seen_at TIMESTAMP NOT NULL,
    total_messages INT DEFAULT 0,
    contrarian_index FLOAT DEFAULT 0.0,
    hostility_index FLOAT DEFAULT 0.0,
    banter_reciprocity FLOAT DEFAULT 0.0,
    griefer_score FLOAT DEFAULT 0.0,
    flagged_status VARCHAR(32) DEFAULT 'CLEAN' -- 'CLEAN', 'WATCHLIST', 'GRIEFER', 'BRIGADE'
);

CREATE TABLE chatter_messages (
    message_id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(64) REFERENCES chatters(user_id),
    vod_id VARCHAR(64) NOT NULL,
    timestamp_offset FLOAT NOT NULL,
    content TEXT NOT NULL,
    intent VARCHAR(32) NOT NULL,
    contextual_label VARCHAR(32) NOT NULL, -- 'BANTER', 'ROAST', 'GRIEFING', 'HARASSMENT'
    is_spike_context BOOLEAN DEFAULT FALSE
);
```

---

## 5. Acceptance Criteria & Deliverables
* **Module**: `src/stream_fusion/chat/profiler.py` (`ChatterProfileStore`, `BanterClassifier`, `GrieferDetector`).
* **Data Contracts**: `ChatterDetailedProfile`, `ChatterSafetyVerdict`, `BrigadeCluster`.
* **CLI Command**: `streamfusion chatter-inspect <username_or_id>` and `streamfusion flag-griefers --chat <chat_path>`.
* **Unit Tests**: `tests/test_chatter_profiler.py` evaluating banter vs. griefing classification on gaming failure vs. serious monologue.
