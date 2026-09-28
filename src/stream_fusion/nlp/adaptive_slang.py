"""Self-Expanding Adaptive Slang & Meme Engine (Spec 17).

Provides statistical burst detection (Z >= 3.0), contextual sentiment auto-tagging,
prosodic acoustic boosting, novel intent clustering, and persistent temporal decay.
"""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

from stream_fusion.models.schemas import ChatMessage, AudioSegment
from stream_fusion.chat.nlp import (
    INTENT_LEXICON,
    INTENT_WEIGHTS,
    normalize_repeated_chars,
    tokenize_for_similarity,
)


class SlangCandidate(BaseModel):
    """Emerging slang or emote candidate detected via statistical burst."""
    term: str
    burst_velocity: float = Field(..., description="Messages per second during burst")
    z_score: float = Field(..., description="Standard deviations above rolling background")
    total_occurrences: int
    unique_authors: int
    window_start_sec: float
    window_end_sec: float
    co_occurring_intents: Dict[str, float] = Field(default_factory=dict)
    inferred_intent: str = "NEUTRAL"
    inferred_valence: float = 0.0
    acoustic_energy_boost: float = 0.0
    confidence: float = 0.5


class AdaptiveTermEntry(BaseModel):
    """Persistent lexicon record with exponential half-life temporal confidence decay."""
    term: str
    inferred_intent: str
    valence: float = 0.0
    confidence: float = 0.5
    first_seen_timestamp: str
    last_seen_timestamp: str
    occurrence_count: int = 1
    unique_authors_count: int = 1
    peak_z_score: float = 3.0
    decay_half_life_days: float = 14.0
    status: str = "ACTIVE"  # "CANDIDATE", "ACTIVE", "PROMOTED", "DECAYED"

    def compute_decayed_confidence(self, as_of: Optional[datetime] = None) -> float:
        """Calculates current confidence following C(t) = C_0 * 2^(-dt / half_life)."""
        now = as_of or datetime.now(timezone.utc)
        try:
            last_dt = datetime.fromisoformat(self.last_seen_timestamp.replace("Z", "+00:00"))
        except Exception:
            return self.confidence

        elapsed_days = max(0.0, (now - last_dt).total_seconds() / 86400.0)
        decay_factor = math.pow(2.0, -elapsed_days / max(1.0, self.decay_half_life_days))
        return round(self.confidence * decay_factor, 3)


class SemanticCluster(BaseModel):
    """Novel semantic cluster for slang not fitting canonical intents."""
    cluster_id: str
    centroid_terms: List[str]
    representative_label: str
    mean_valence: float
    member_count: int


class RollingBurstDetector:
    """Detects word and emote arrival spikes >= 3.0 sigma over rolling windows."""

    def __init__(
        self,
        window_sec: float = 10.0,
        z_threshold: float = 3.0,
        min_occurrences: int = 4,
        min_distinct_authors: int = 3,
        alpha_ema: float = 0.25,
    ):
        self.window_sec = window_sec
        self.z_threshold = z_threshold
        self.min_occurrences = min_occurrences
        self.min_distinct_authors = min_distinct_authors
        self.alpha_ema = alpha_ema

        # Baseline stats per normalized token
        self._mean_velocity: Dict[str, float] = defaultdict(float)
        self._var_velocity: Dict[str, float] = defaultdict(lambda: 0.05)

    def extract_tokens(self, text: str) -> List[str]:
        """Extracts and normalizes potential slang tokens/emotes from text."""
        cleaned = re.sub(r"[^\w<3]", " ", text)
        raw_tokens = cleaned.split()
        normalized = []
        for t in raw_tokens:
            norm = normalize_repeated_chars(t.strip().lower())
            # Skip pure numbers and single common letters
            if len(norm) > 1 and not norm.isdigit():
                normalized.append(norm)
        return normalized

    def detect_bursts(self, messages: List[ChatMessage]) -> List[SlangCandidate]:
        """Scans messages using rolling temporal windows and identifies burst candidates."""
        if not messages:
            return []

        sorted_msgs = sorted(messages, key=lambda m: m.timestamp_offset)
        max_time = sorted_msgs[-1].timestamp_offset
        current_time = sorted_msgs[0].timestamp_offset

        candidates: List[SlangCandidate] = []
        seen_candidate_terms: Set[str] = set()

        msg_idx = 0
        n_msgs = len(sorted_msgs)

        while current_time <= max_time:
            window_end = current_time + self.window_sec
            window_msgs: List[ChatMessage] = []

            # Gather messages in current window
            while msg_idx < n_msgs and sorted_msgs[msg_idx].timestamp_offset < window_end:
                if sorted_msgs[msg_idx].timestamp_offset >= current_time:
                    window_msgs.append(sorted_msgs[msg_idx])
                msg_idx += 1

            if window_msgs:
                token_counts: Dict[str, int] = Counter()
                token_authors: Dict[str, Set[str]] = defaultdict(set)

                for msg in window_msgs:
                    tokens = self.extract_tokens(msg.content)
                    for tok in set(tokens):  # count once per message
                        token_counts[tok] += 1
                        token_authors[tok].add(msg.author_name)

                # Evaluate Z-score for each token
                for tok, count in token_counts.items():
                    n_authors = len(token_authors[tok])
                    if count >= self.min_occurrences and n_authors >= self.min_distinct_authors:
                        inst_velocity = count / self.window_sec
                        prev_mean = self._mean_velocity[tok]
                        prev_std = math.sqrt(max(0.001, self._var_velocity[tok]))

                        # Z-Score formulation
                        z = (inst_velocity - prev_mean) / (prev_std + 0.05)

                        # Update EMA
                        self._mean_velocity[tok] = (
                            self.alpha_ema * inst_velocity + (1 - self.alpha_ema) * prev_mean
                        )
                        self._var_velocity[tok] = (
                            self.alpha_ema * ((inst_velocity - self._mean_velocity[tok]) ** 2)
                            + (1 - self.alpha_ema) * self._var_velocity[tok]
                        )

                        if z >= self.z_threshold and tok not in seen_candidate_terms:
                            seen_candidate_terms.add(tok)
                            candidates.append(
                                SlangCandidate(
                                    term=tok,
                                    burst_velocity=round(inst_velocity, 2),
                                    z_score=round(z, 2),
                                    total_occurrences=count,
                                    unique_authors=n_authors,
                                    window_start_sec=round(current_time, 2),
                                    window_end_sec=round(window_end, 2),
                                )
                            )

            # Advance window by half window_sec (50% overlap)
            current_time += self.window_sec / 2.0
            # Rewind msg_idx to start of next window
            while msg_idx > 0 and sorted_msgs[msg_idx - 1].timestamp_offset >= current_time:
                msg_idx -= 1

        return candidates


class ContextualAutoTagger:
    """Infers intent and emotional valence of candidate slang from co-occurring text and audio."""

    def __init__(self, canonical_lexicon: Optional[Dict[str, Set[str]]] = None):
        self.lexicon = canonical_lexicon or INTENT_LEXICON

    def tag_candidate(
        self,
        candidate: SlangCandidate,
        all_messages: List[ChatMessage],
        audio_segments: Optional[List[AudioSegment]] = None,
    ) -> SlangCandidate:
        """Tags candidate with inferred intent, valence, and acoustic boost."""
        # Find messages in the burst window containing the candidate term
        co_occurring_scores: Dict[str, float] = defaultdict(float)
        matched_count = 0

        for msg in all_messages:
            if candidate.window_start_sec <= msg.timestamp_offset <= candidate.window_end_sec + 2.0:
                content_lower = msg.content.lower()
                if candidate.term in content_lower:
                    matched_count += 1
                    # Evaluate other tokens in this message
                    words = [w.strip() for w in re.findall(r"\w+|<3", content_lower)]
                    for w in words:
                        if w == candidate.term:
                            continue
                        for intent, kw_set in self.lexicon.items():
                            if w in kw_set:
                                co_occurring_scores[intent] += 1.0

        # Determine dominant intent
        total_intent_hits = sum(co_occurring_scores.values())
        norm_intents = {}
        if total_intent_hits > 0:
            norm_intents = {
                k: round(v / total_intent_hits, 3) for k, v in co_occurring_scores.items()
            }
            dominant_intent = max(norm_intents.items(), key=lambda x: x[1])[0]
        else:
            dominant_intent = "NEUTRAL"
            norm_intents = {"NEUTRAL": 1.0}

        # Calculate valence
        inferred_valence = 0.0
        for intent, weight in norm_intents.items():
            inferred_valence += weight * INTENT_WEIGHTS.get(intent, 0.0)

        # Acoustic prosody boost
        acoustic_boost = 0.0
        if audio_segments:
            for seg in audio_segments:
                if (
                    candidate.window_start_sec - 2.0 <= seg.start_sec <= candidate.window_end_sec + 2.0
                    or candidate.window_start_sec - 2.0 <= seg.end_sec <= candidate.window_end_sec + 2.0
                ):
                    # If streamer is laughing or loud in transcript
                    t_lower = seg.transcript.lower()
                    if any(laugh in t_lower for laugh in ["haha", "laugh", "lmao", "no way"]):
                        acoustic_boost = max(acoustic_boost, 2.0)
                    elif seg.confidence > 0.8:
                        acoustic_boost = max(acoustic_boost, 1.0)

        # Confidence calculation
        base_conf = 0.5 + min(0.35, candidate.z_score * 0.05)
        if acoustic_boost >= 1.5:
            base_conf = min(0.95, base_conf + 0.15)

        candidate.co_occurring_intents = norm_intents
        candidate.inferred_intent = dominant_intent
        candidate.inferred_valence = round(inferred_valence, 3)
        candidate.acoustic_energy_boost = round(acoustic_boost, 2)
        candidate.confidence = round(base_conf, 3)

        return candidate


class NovelClusterDetector:
    """Clusters emerging terms that don't match canonical intents using character n-gram centroids."""

    def __init__(self, n_gram_size: int = 3, similarity_threshold: float = 0.40):
        self.n_gram_size = n_gram_size
        self.similarity_threshold = similarity_threshold

    def _get_ngrams(self, term: str) -> Set[str]:
        padded = f"^{term}$"
        return {
            padded[i : i + self.n_gram_size]
            for i in range(len(padded) - self.n_gram_size + 1)
        }

    def _jaccard(self, set_a: Set[str], set_b: Set[str]) -> float:
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)

    def discover_clusters(self, candidates: List[SlangCandidate]) -> List[SemanticCluster]:
        """Clusters novel candidates with low canonical confidence into semantic groups."""
        novel_candidates = [
            c for c in candidates if c.inferred_intent == "NEUTRAL" or c.confidence < 0.60
        ]
        if not novel_candidates:
            return []

        clusters: List[List[SlangCandidate]] = []
        ngram_cache = {c.term: self._get_ngrams(c.term) for c in novel_candidates}

        for cand in novel_candidates:
            c_ngrams = ngram_cache[cand.term]
            assigned = False
            for group in clusters:
                # Check similarity against group centroid (first term or average)
                group_rep = group[0].term
                sim = self._jaccard(c_ngrams, ngram_cache[group_rep])
                if sim >= self.similarity_threshold:
                    group.append(cand)
                    assigned = True
                    break
            if not assigned:
                clusters.append([cand])

        results = []
        for idx, group in enumerate(clusters):
            terms = [c.term for c in group]
            avg_valence = round(sum(c.inferred_valence for c in group) / len(group), 3)
            results.append(
                SemanticCluster(
                    cluster_id=f"novel_cluster_{idx + 1}",
                    centroid_terms=terms,
                    representative_label=terms[0],
                    mean_valence=avg_valence,
                    member_count=len(group),
                )
            )

        return results


class AdaptiveLexiconStore:
    """Persistent catalog (adaptive_lexicon.json) with temporal decay and intent mapping."""

    def __init__(self, db_path: Path = Path("adaptive_lexicon.json")):
        self.db_path = Path(db_path)
        self.entries: Dict[str, AdaptiveTermEntry] = {}
        self.load()

    def load(self, path: Optional[Path] = None) -> None:
        """Loads lexicon entries from disk."""
        target = path or self.db_path
        if target.exists():
            try:
                with open(target, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                    self.entries = {
                        k: AdaptiveTermEntry.model_validate(v) for k, v in raw.items()
                    }
            except Exception:
                self.entries = {}

    def save(self, path: Optional[Path] = None) -> None:
        """Serializes lexicon entries to disk."""
        target = path or self.db_path
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = {k: v.model_dump() for k, v in self.entries.items()}
        with open(target, "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2)

    def add_or_update(
        self, candidate: SlangCandidate, timestamp_iso: Optional[str] = None
    ) -> AdaptiveTermEntry:
        """Inserts or updates a candidate in the adaptive lexicon."""
        now_iso = timestamp_iso or datetime.now(timezone.utc).isoformat()
        term = candidate.term.lower()

        if term in self.entries:
            entry = self.entries[term]
            entry.occurrence_count += candidate.total_occurrences
            entry.unique_authors_count += candidate.unique_authors
            entry.last_seen_timestamp = now_iso
            entry.peak_z_score = max(entry.peak_z_score, candidate.z_score)
            # Refine confidence and valence
            entry.confidence = min(0.99, round(entry.confidence * 0.4 + candidate.confidence * 0.6, 3))
            entry.valence = round(entry.valence * 0.3 + candidate.inferred_valence * 0.7, 3)
            if candidate.inferred_intent != "NEUTRAL":
                entry.inferred_intent = candidate.inferred_intent

            if entry.confidence >= 0.80 and entry.occurrence_count >= 15:
                entry.status = "PROMOTED"
            else:
                entry.status = "ACTIVE"
        else:
            status = "PROMOTED" if candidate.confidence >= 0.85 else "ACTIVE"
            entry = AdaptiveTermEntry(
                term=term,
                inferred_intent=candidate.inferred_intent,
                valence=candidate.inferred_valence,
                confidence=candidate.confidence,
                first_seen_timestamp=now_iso,
                last_seen_timestamp=now_iso,
                occurrence_count=candidate.total_occurrences,
                unique_authors_count=candidate.unique_authors,
                peak_z_score=candidate.z_score,
                status=status,
            )
            self.entries[term] = entry

        return entry

    def get_active_lexicon(self) -> Dict[str, Set[str]]:
        """Returns intent mapping for active/promoted slang to plug into ChatNLPAnalyzer."""
        mapping: Dict[str, Set[str]] = defaultdict(set)
        for term, entry in self.entries.items():
            if entry.status in ("ACTIVE", "PROMOTED") and entry.inferred_intent != "NEUTRAL":
                mapping[entry.inferred_intent].add(term)
        return dict(mapping)

    def apply_temporal_decay(self, as_of: Optional[datetime] = None) -> List[str]:
        """Applies half-life temporal decay and marks inactive entries as DECAYED."""
        decayed = []
        for term, entry in self.entries.items():
            cur_conf = entry.compute_decayed_confidence(as_of=as_of)
            if cur_conf < 0.25 and entry.status != "DECAYED":
                entry.status = "DECAYED"
                decayed.append(term)
        return decayed

    def prune_decayed(self, threshold: float = 0.20) -> int:
        """Permanently removes terms with decayed confidence below threshold."""
        to_prune = [
            t
            for t, e in self.entries.items()
            if e.compute_decayed_confidence() < threshold and e.status == "DECAYED"
        ]
        for t in to_prune:
            del self.entries[t]
        return len(to_prune)


class AdaptiveSlangEngine:
    """High-level orchestrator connecting burst detection, auto-tagging, clustering, and storage."""

    def __init__(
        self,
        lexicon_store: Optional[AdaptiveLexiconStore] = None,
        burst_detector: Optional[RollingBurstDetector] = None,
        auto_tagger: Optional[ContextualAutoTagger] = None,
        cluster_detector: Optional[NovelClusterDetector] = None,
    ):
        self.store = lexicon_store or AdaptiveLexiconStore()
        self.detector = burst_detector or RollingBurstDetector()
        self.tagger = auto_tagger or ContextualAutoTagger()
        self.clusterer = cluster_detector or NovelClusterDetector()

    def process_chat_stream(
        self,
        messages: List[ChatMessage],
        audio_segments: Optional[List[AudioSegment]] = None,
        persist: bool = True,
    ) -> Tuple[List[SlangCandidate], List[SemanticCluster]]:
        """Processes messages to find bursts, auto-tag intents, cluster, and update store."""
        raw_candidates = self.detector.detect_bursts(messages)

        tagged_candidates: List[SlangCandidate] = []
        for cand in raw_candidates:
            tagged = self.tagger.tag_candidate(
                cand, all_messages=messages, audio_segments=audio_segments
            )
            tagged_candidates.append(tagged)
            if persist:
                self.store.add_or_update(tagged)

        clusters = self.clusterer.discover_clusters(tagged_candidates)

        if persist:
            self.store.save()

        return tagged_candidates, clusters
