"""Cross-Broadcast Temporal Stance Shift Tracker and Opinion Synthesizer (Spec 20)."""

from collections import defaultdict
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid

from stream_fusion.models.schemas import (
    EntityOpinionSynthesis,
    StancePolarity,
    StanceShiftRecord,
    StreamerClaim,
)

logger = logging.getLogger(__name__)

STANCE_VALUES = {
    "POSITIVE": 1.0,
    "APPROVAL": 1.0,
    "FAVORABLE": 1.0,
    "NEUTRAL": 0.0,
    "MIXED": 0.0,
    "NEGATIVE": -1.0,
    "DISAPPROVAL": -1.0,
    "CRITICAL": -1.0,
}


class StreamEntityObservation:
    """A single recorded stance observation for an entity in a stream."""

    def __init__(
        self,
        entity_name: str,
        stance: str,
        stream_id: str,
        quote: str,
        timestamp_sec: float = 0.0,
        recorded_at: Optional[datetime] = None,
    ):
        self.entity_name = entity_name.strip()
        self.stance = stance.upper()
        self.stream_id = stream_id
        self.quote = quote
        self.timestamp_sec = timestamp_sec
        self.recorded_at = recorded_at or datetime.now(timezone.utc)

    @property
    def numeric_value(self) -> float:
        return STANCE_VALUES.get(self.stance, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_name": self.entity_name,
            "stance": self.stance,
            "numeric_value": self.numeric_value,
            "stream_id": self.stream_id,
            "quote": self.quote,
            "timestamp_sec": self.timestamp_sec,
            "recorded_at": self.recorded_at.isoformat(),
        }


class TemporalStanceShiftTracker:
    """Tracks streamer stance trajectories across multiple broadcasts and detects reversals."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path
        # Mapping entity_name (lower) -> List[StreamEntityObservation]
        self._history: Dict[str, List[StreamEntityObservation]] = defaultdict(list)
        if self.storage_path and self.storage_path.exists():
            self.load()

    def record_claim(self, claim: StreamerClaim, stream_id: str) -> None:
        """Records an entity observation directly from an extracted claim."""
        if not claim.subject:
            return
        stance_str = "NEUTRAL"
        if hasattr(claim, "stance") and claim.stance:
            stance_str = claim.stance.value if hasattr(claim.stance, "value") else str(claim.stance)

        self.record_observation(
            entity_name=claim.subject,
            stance=stance_str,
            stream_id=stream_id,
            quote=claim.statement,
            timestamp_sec=claim.start_sec if hasattr(claim, "start_sec") else 0.0,
        )

    def record_observation(
        self,
        entity_name: str,
        stance: str,
        stream_id: str,
        quote: str,
        timestamp_sec: float = 0.0,
    ) -> None:
        """Appends a new observation to an entity's timeline."""
        obs = StreamEntityObservation(
            entity_name=entity_name,
            stance=stance,
            stream_id=stream_id,
            quote=quote,
            timestamp_sec=timestamp_sec,
        )
        self._history[entity_name.lower()].append(obs)
        if self.storage_path:
            self.save()

    def get_timeline(self, entity_name: str) -> List[StreamEntityObservation]:
        """Returns ordered timeline of observations for an entity."""
        return sorted(self._history.get(entity_name.lower(), []), key=lambda o: o.recorded_at)

    def detect_shifts(self, entity_name: str) -> List[StanceShiftRecord]:
        """Identifies all stance shifts and reversals between sequential observations."""
        timeline = self.get_timeline(entity_name)
        if len(timeline) < 2:
            return []

        shifts: List[StanceShiftRecord] = []
        for i in range(1, len(timeline)):
            prev = timeline[i - 1]
            curr = timeline[i]

            prev_val = prev.numeric_value
            curr_val = curr.numeric_value
            delta = curr_val - prev_val

            if abs(delta) >= 0.5:
                # Sign flip check (e.g. +1.0 to -1.0 or -1.0 to +1.0)
                is_reversal = (prev_val > 0 and curr_val < 0) or (prev_val < 0 and curr_val > 0)

                shifts.append(
                    StanceShiftRecord(
                        entity_name=curr.entity_name,
                        previous_stance=prev.stance,
                        new_stance=curr.stance,
                        shift_delta=round(delta, 2),
                        previous_stream_id=prev.stream_id,
                        new_stream_id=curr.stream_id,
                        evidence_quote_before=prev.quote,
                        evidence_quote_after=curr.quote,
                        is_reversal=is_reversal,
                        detected_at=curr.recorded_at,
                    )
                )

        return shifts

    def compute_volatility(self, entity_name: str) -> float:
        """Calculates normalized stance volatility index (0.0 to 1.0)."""
        timeline = self.get_timeline(entity_name)
        if len(timeline) < 2:
            return 0.0

        total_variation = 0.0
        for i in range(1, len(timeline)):
            total_variation += abs(timeline[i].numeric_value - timeline[i - 1].numeric_value)

        # Max possible variation is 2.0 per step (from +1 to -1)
        max_possible = 2.0 * (len(timeline) - 1)
        if max_possible == 0.0:
            return 0.0
        return min(1.0, round(total_variation / max_possible, 3))

    def save(self) -> None:
        """Serializes observation history to disk."""
        if not self.storage_path:
            return
        data = {}
        for ent, obs_list in self._history.items():
            data[ent] = [o.to_dict() for o in obs_list]
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self) -> None:
        """Deserializes observation history from disk."""
        if not self.storage_path or not self.storage_path.exists():
            return
        try:
            data = json.loads(self.storage_path.read_text(encoding="utf-8"))
            for ent, items in data.items():
                self._history[ent] = [
                    StreamEntityObservation(
                        entity_name=it["entity_name"],
                        stance=it["stance"],
                        stream_id=it["stream_id"],
                        quote=it["quote"],
                        timestamp_sec=it.get("timestamp_sec", 0.0),
                        recorded_at=datetime.fromisoformat(it["recorded_at"]) if "recorded_at" in it else None,
                    )
                    for it in items
                ]
        except Exception as e:
            logger.warning(f"Failed to load stance shift history: {e}")


class CrossStreamOpinionSynthesizer:
    """Synthesizes historical creator stances, flags contradictions, and generates briefs."""

    def __init__(self, tracker: TemporalStanceShiftTracker):
        self.tracker = tracker

    def synthesize(self, entity_name: str) -> EntityOpinionSynthesis:
        """Generates a complete longitudinal opinion synthesis for an entity."""
        timeline = self.tracker.get_timeline(entity_name)
        shifts = self.tracker.detect_shifts(entity_name)
        volatility = self.tracker.compute_volatility(entity_name)

        if not timeline:
            return EntityOpinionSynthesis(
                entity_name=entity_name,
                overall_consensus_stance="UNKNOWN",
                total_claims_count=0,
                contradiction_count=0,
                volatility_score=0.0,
                summary=f"No recorded statements found for '{entity_name}'.",
            )

        # Calculate consensus stance
        mean_val = sum(o.numeric_value for o in timeline) / len(timeline)
        if mean_val >= 0.35:
            consensus = "POSITIVE"
        elif mean_val <= -0.35:
            consensus = "NEGATIVE"
        else:
            consensus = "MIXED / NUANCED"

        # Detect contradictions (reversals)
        contradictions = []
        for s in shifts:
            if s.is_reversal:
                contradictions.append({
                    "shift_id": s.shift_id,
                    "from_stream": s.previous_stream_id,
                    "to_stream": s.new_stream_id,
                    "from_stance": s.previous_stance,
                    "to_stance": s.new_stance,
                    "earlier_quote": s.evidence_quote_before,
                    "later_quote": s.evidence_quote_after,
                })

        # Generate summary
        summary_lines = [
            f"Creator stance analysis for '{entity_name}':",
            f"- Overall Consensus: {consensus} (Mean Score: {mean_val:+.2f})",
            f"- Recorded Observations: {len(timeline)} across multiple streams",
            f"- Contradictions / Stance Reversals: {len(contradictions)}",
            f"- Stance Volatility Index: {volatility:.2f} / 1.00",
        ]
        if contradictions:
            summary_lines.append(f"Notable reversal: Shifted from {contradictions[0]['from_stance']} to {contradictions[0]['to_stance']} between streams.")

        return EntityOpinionSynthesis(
            entity_name=entity_name,
            overall_consensus_stance=consensus,
            total_claims_count=len(timeline),
            contradiction_count=len(contradictions),
            volatility_score=volatility,
            stance_timeline=[o.to_dict() for o in timeline],
            contradictions=contradictions,
            summary="\n".join(summary_lines),
        )
