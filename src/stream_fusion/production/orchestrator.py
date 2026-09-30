"""Orchestrator for Autonomous Multi-Agent Short Production (Spec 21).

Coordinates the multi-agent editorial committee (Director -> Editor -> Policy -> Copywriter -> Publisher).
"""

from pathlib import Path
from typing import List, Optional

from stream_fusion.models.schemas import (
    AdaptiveTermEntry,
    AuditStatus,
    AudioSegment,
    ChatMessage,
    DynamicMomentThresholds,
    FusionSlice,
    ShortCandidate,
    ShortProductionPackage,
    SponsorSegment,
    StreamAnalysisResult,
    StreamerClaim,
    VisualKeyframe,
    WordTiming,
)
from stream_fusion.production.checker import PolicyAgent
from stream_fusion.production.copywriter import CopywriterAgent
from stream_fusion.production.director import DirectorAgent
from stream_fusion.production.editor import EditorAgent
from stream_fusion.production.publisher import PublisherAgent
from stream_fusion.schema.envelope import StreamFusionEnvelope


class ShortProductionOrchestrator:
    """Orchestrates the entire multi-agent committee pipeline to produce vertical shorts."""

    def __init__(
        self,
        director: Optional[DirectorAgent] = None,
        editor: Optional[EditorAgent] = None,
        checker: Optional[PolicyAgent] = None,
        copywriter: Optional[CopywriterAgent] = None,
        publisher: Optional[PublisherAgent] = None,
    ):
        self.director = director or DirectorAgent()
        self.editor = editor or EditorAgent()
        self.checker = checker or PolicyAgent()
        self.copywriter = copywriter or CopywriterAgent()
        self.publisher = publisher or PublisherAgent()

    def produce_shorts(
        self,
        analysis: StreamAnalysisResult,
        output_dir: Path,
        video_path: Optional[Path] = None,
        fusion_slices: Optional[List[FusionSlice]] = None,
        keyframes: Optional[List[VisualKeyframe]] = None,
        audio_segments: Optional[List[AudioSegment]] = None,
        chat_messages: Optional[List[ChatMessage]] = None,
        claims: Optional[List[StreamerClaim]] = None,
        sponsor_segments: Optional[List[SponsorSegment]] = None,
        adaptive_terms: Optional[List[AdaptiveTermEntry]] = None,
        top_k: Optional[int] = 3,
        min_highlight_score: float = 0.4,
        allow_flagged: bool = True,
        dry_run: bool = False,
        dynamic: bool = False,
        thresholds: Optional[DynamicMomentThresholds] = None,
    ) -> List[tuple[ShortProductionPackage, StreamFusionEnvelope[ShortProductionPackage]]]:
        """Runs the multi-agent production committee end-to-end."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Director selects candidates
        candidates = self.director.select_candidates(
            analysis=analysis,
            fusion_slices=fusion_slices,
            audio_segments=audio_segments,
            chat_messages=chat_messages,
            top_k=top_k,
            min_highlight_score=min_highlight_score,
            dynamic=dynamic,
            thresholds=thresholds,
        )

        packages: List[tuple[ShortProductionPackage, StreamFusionEnvelope[ShortProductionPackage]]] = []

        # 2. Iterate through each candidate short
        for candidate in candidates:
            # Step A: Editor designs framing and styling
            cut_plan = self.editor.create_cut_plan(
                candidate=candidate,
                keyframes=keyframes,
            )

            # Step B: Policy agent audits content safety, claims, and sponsors
            audit_report = self.checker.audit_candidate(
                candidate=candidate,
                audio_segments=audio_segments,
                chat_messages=chat_messages,
                claims=claims,
                sponsor_segments=sponsor_segments,
            )

            # If rejected by policy, skip unless explicitly permitted
            if audit_report.audit_status == AuditStatus.REJECTED and not allow_flagged:
                continue

            # Step C: Copywriter generates viral copy and virality scorecard
            copy_bundle, virality = self.copywriter.generate_copy(
                candidate=candidate,
                audit=audit_report,
                adaptive_terms=adaptive_terms,
            )

            # Step D: Publisher renders video & extracts thumbnail if video exists
            rendered_video = None
            rendered_thumb = None

            if video_path and video_path.exists():
                # Extract word timings for subtitles in window
                words_in_window: List[WordTiming] = []
                if audio_segments:
                    for seg in audio_segments:
                        if seg.words and not (seg.end_sec < candidate.start_sec or seg.start_sec > candidate.end_sec):
                            words_in_window.extend(seg.words)

                try:
                    rendered_video = self.publisher.render_short(
                        candidate=candidate,
                        plan=cut_plan,
                        video_path=video_path,
                        output_dir=output_dir,
                        words=words_in_window or None,
                        dry_run=dry_run,
                    )
                except Exception:
                    # Non-fatal if ffmpeg is missing or dry_run
                    rendered_video = None

                try:
                    rendered_thumb = self.publisher.extract_thumbnail(
                        candidate=candidate,
                        video_path=video_path,
                        output_dir=output_dir,
                        dry_run=dry_run,
                    )
                except Exception:
                    rendered_thumb = None

            # Step E: Publisher packages final artifacts and emits envelope
            pkg, env = self.publisher.package(
                candidate=candidate,
                plan=cut_plan,
                audit=audit_report,
                copy=copy_bundle,
                virality=virality,
                output_dir=output_dir,
                video_path=rendered_video,
                thumbnail_path=rendered_thumb,
            )
            packages.append((pkg, env))

        return packages
