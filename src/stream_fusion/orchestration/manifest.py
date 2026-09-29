"""Full-Spectrum Manifest Builder and Serializer (Spec 24)."""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from stream_fusion.models.schemas import (
    FullSpectrumManifest,
    FullSpectrumStageSummary,
    StageExecutionStatus,
    StreamAnalysisResult,
)

logger = logging.getLogger(__name__)


class FullSpectrumManifestBuilder:
    """Builds and serializes a comprehensive FullSpectrumManifest linking all multimodal subsystem outputs."""

    def __init__(self, stream_id: str, output_directory: Path):
        self.stream_id = stream_id
        self.output_directory = Path(output_directory)
        self.manifest = FullSpectrumManifest(
            manifest_id=f"fsm-{uuid.uuid4().hex[:8]}",
            stream_id=stream_id,
            output_directory=str(self.output_directory.resolve()),
        )

    def record_stage(
        self,
        stage_name: str,
        status: StageExecutionStatus,
        duration_sec: float,
        output_summary: str = "",
        error_message: Optional[str] = None,
    ) -> None:
        """Records telemetry and completion status for a pipeline stage."""
        summary = FullSpectrumStageSummary(
            stage_name=stage_name,
            status=status,
            duration_sec=round(duration_sec, 3),
            output_summary=output_summary,
            error_message=error_message,
        )
        self.manifest.stages[stage_name] = summary

    def populate_from_analysis(
        self,
        analysis: StreamAnalysisResult,
        effective_media_sec: float,
        total_audio_segments: Optional[int] = None,
        total_keyframes: Optional[int] = None,
        total_chat_messages: Optional[int] = None,
        calibrated_delay_sec: Optional[float] = None,
    ) -> None:
        """Fills core analytical metrics from StreamAnalysisResult."""
        self.manifest.effective_media_duration_sec = round(effective_media_sec, 2)

        if total_audio_segments is not None:
            self.manifest.total_audio_segments = total_audio_segments
        elif hasattr(analysis, "audio_segments"):
            self.manifest.total_audio_segments = len(getattr(analysis, "audio_segments"))

        if total_keyframes is not None:
            self.manifest.total_keyframes = total_keyframes
        elif hasattr(analysis, "visual_keyframes"):
            self.manifest.total_keyframes = len(getattr(analysis, "visual_keyframes"))

        if total_chat_messages is not None:
            self.manifest.total_chat_messages = total_chat_messages
        elif analysis.total_chat_messages:
            self.manifest.total_chat_messages = analysis.total_chat_messages
        elif hasattr(analysis, "chat_buckets"):
            self.manifest.total_chat_messages = sum(getattr(b, "message_count", 0) for b in getattr(analysis, "chat_buckets"))

        slices = getattr(analysis, "slices", []) or getattr(analysis, "fusion_slices", [])
        self.manifest.total_fusion_slices = len(slices)

        if calibrated_delay_sec is not None:
            self.manifest.calibrated_broadcast_delay_sec = round(calibrated_delay_sec, 3)
        elif hasattr(analysis, "metadata") and isinstance(analysis.metadata, dict):
            self.manifest.calibrated_broadcast_delay_sec = round(analysis.metadata.get("latency_offset_sec", 0.0), 3)

        if slices:
            agreement_scores = [s.agreement_score for s in slices if s.agreement_score is not None]
            if agreement_scores:
                self.manifest.take_agreement_mean = round(float(sum(agreement_scores) / len(agreement_scores)), 3)


    def save(self, file_path: Optional[Path] = None) -> Path:
        """Saves manifest to disk as formatted JSON."""
        target = file_path or (self.output_directory / f"{self.stream_id}_full_spectrum_manifest.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.manifest.model_dump(mode="json"), f, indent=2)
        logger.info("Saved FullSpectrumManifest to %s", target)
        return target
