"""Full-Spectrum Pipeline Master Synergy Orchestrator (Spec 24).

Integrates all StreamFusion moving parts into a unified analytical lifecycle:
1. Ingestion & Demuxing
2. Worker-Isolated Audio & Diarization (with Voiceprint matching)
3. Worker-Isolated Vision & Screen OCR (with Bounded Buffering)
4. Adaptive Chat Calibration & Slang Discovery
5. Multimodal Fusion Matrix & Highlight Detection
6. Sponsor & Brand Performance Quantification
7. Claim Extraction, Web Grounding & Stance Shift Tracking
8. Autonomous Multi-Agent Short Production & Auto-Publisher
9. Master Synergy Manifest Generation & Event Bus Dispatch
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console

from stream_fusion.analytics.sponsor_quantifier import SponsorImpactQuantifier
from stream_fusion.audio.diarizer import ReactionDiarizer
from stream_fusion.audio.transcriber import AudioTranscriber
from stream_fusion.audio.voiceprint import VoiceprintLibrary
from stream_fusion.chat.analyzer import ChatAnalyzer
from stream_fusion.chat.calibrator import LatencyCalibrator
from stream_fusion.chat.profiler import ChatterProfiler
from stream_fusion.config import StreamFusionConfig
from stream_fusion.export.dataset import export_to_parquet, export_training_triples_jsonl
from stream_fusion.export.html_report import export_html_report
from stream_fusion.fusion.matrix import FusionEngine
from stream_fusion.ingest.demuxer import MediaDemuxer
from stream_fusion.ingest.downloader import StreamDownloader
from stream_fusion.knowledge.claims import ClaimExtractor
from stream_fusion.knowledge.temporal_stance import TemporalStanceShiftTracker
from stream_fusion.knowledge.web_grounding import LiveWebGroundingEngine
from stream_fusion.models.schemas import (
    AudioSegment,
    ChatMessage,
    FullSpectrumConfig,
    FullSpectrumManifest,
    StageExecutionStatus,
    StreamAnalysisResult,
    VisualKeyframe,
)
from stream_fusion.monitoring.broadcaster import LiveEventBroadcaster
from stream_fusion.nlp.adaptive_slang import AdaptiveLexiconStore, AdaptiveSlangEngine
from stream_fusion.orchestration.manifest import FullSpectrumManifestBuilder
from stream_fusion.production.orchestrator import ShortProductionOrchestrator
from stream_fusion.schema.envelope import StreamEventType, StreamFusionEnvelope
from stream_fusion.vision.processor import VisionProcessor
from stream_fusion.vision.optimizer import AdaptiveFrameOptimizer
from stream_fusion.workers.bounded_buffer import BoundedFrameBuffer
from stream_fusion.workers.isolation import WorkerIsolationManager

logger = logging.getLogger(__name__)
console = Console()


class FullSpectrumPipeline:
    """Master orchestrator connecting all StreamFusion multimodal intelligence layers."""

    def __init__(
        self,
        config: Optional[FullSpectrumConfig] = None,
        base_config: Optional[StreamFusionConfig] = None,
        broadcaster: Optional[LiveEventBroadcaster] = None,
    ):
        self.config = config or FullSpectrumConfig()
        self.base_config = base_config or StreamFusionConfig()
        self.broadcaster = broadcaster

        # Core Engines
        self.downloader = StreamDownloader()
        self.demuxer = MediaDemuxer()
        self.chat_analyzer = ChatAnalyzer()
        self.calibrator = LatencyCalibrator()
        self.fusion = FusionEngine(bucket_size_sec=self.base_config.chat.bucket_window_sec)

        # Intelligence Engines
        self.slang_engine = AdaptiveSlangEngine(
            lexicon_store=AdaptiveLexiconStore(db_path=Path(self.config.adaptive_lexicon_db))
        )
        self.chatter_profiler = ChatterProfiler()
        self.sponsor_quantifier = SponsorImpactQuantifier()
        self.claim_extractor = ClaimExtractor()
        self.web_grounder = LiveWebGroundingEngine()
        self.stance_tracker = TemporalStanceShiftTracker(storage_path=Path(self.config.stance_history_db))
        self.short_orchestrator = ShortProductionOrchestrator()

    def run(
        self,
        media_input: Path,
        chat_input: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        duration_sec: Optional[float] = None,
        start_time_sec: Optional[float] = None,
        creator_name: str = "Streamer",
    ) -> Tuple[StreamAnalysisResult, FullSpectrumManifest]:
        """Executes the complete unified 9-phase synergy pipeline."""
        start_overall = time.time()
        media_path = Path(media_input)
        out_dir = Path(output_dir or self.base_config.storage.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stream_id = media_path.stem

        builder = FullSpectrumManifestBuilder(stream_id=stream_id, output_directory=out_dir)

        console.print(f"[bold purple]StreamFusion Full-Spectrum Pipeline[/bold purple]: {media_path.name}")

        # -------------------------------------------------------------
        # Phase 1: Ingestion & Demuxing
        # -------------------------------------------------------------
        t0 = time.time()
        try:
            console.print("[bold cyan][1/9][/bold cyan] Demuxing audio & extracting keyframe boundaries...")
            wav_path = out_dir / f"{stream_id}_audio_16k.wav"
            self.demuxer.extract_audio_16k_mono(
                media_path, wav_path, start_time_sec=start_time_sec, duration_sec=duration_sec
            )

            frames_dir = out_dir / f"{stream_id}_keyframes"
            frames = self.demuxer.extract_frames_at_interval(
                media_path,
                frames_dir,
                interval_sec=self.config.sample_interval_sec,
                start_time_sec=start_time_sec,
                duration_sec=duration_sec,
            )
            effective_duration = duration_sec or (len(frames) * self.config.sample_interval_sec)
            builder.record_stage(
                "phase_1_demuxing",
                StageExecutionStatus.SUCCESS,
                duration_sec=time.time() - t0,
                output_summary=f"Extracted 16kHz audio and {len(frames)} keyframe frames ({effective_duration:.1f}s)",
            )
        except Exception as ex:
            builder.record_stage("phase_1_demuxing", StageExecutionStatus.FAILED, time.time() - t0, error_message=str(ex))
            raise

        # -------------------------------------------------------------
        # Phase 2: Speech Transcription & Voice Diarization (GPU)
        # -------------------------------------------------------------
        t0 = time.time()
        try:
            console.print("[bold cyan][2/9][/bold cyan] Running Speech Transcription & Voice Diarization...")
            if self.config.isolate_gpu_workers:
                worker_mgr = WorkerIsolationManager()
                trans_cfg = {
                    "whisper_model": self.base_config.audio.whisper_model,
                    "device": self.base_config.audio.device,
                    "compute_type": self.base_config.audio.compute_type,
                }
                audio_segments = worker_mgr.transcribe_isolated(wav_path, config=trans_cfg)
                diar_cfg = {
                    "hf_token": self.base_config.audio.hf_token,
                    "device": self.base_config.audio.device,
                }
                audio_segments = worker_mgr.diarize_isolated(wav_path, audio_segments, config=diar_cfg)
            else:
                transcriber = AudioTranscriber(
                    model_size=self.base_config.audio.whisper_model,
                    device=self.base_config.audio.device,
                    compute_type=self.base_config.audio.compute_type,
                )
                try:
                    audio_segments = transcriber.transcribe(wav_path)
                finally:
                    transcriber.unload()

                diarizer = ReactionDiarizer(
                    hf_token=self.base_config.audio.hf_token,
                    device=self.base_config.audio.device,
                )
                try:
                    audio_segments = diarizer.diarize_and_tag(audio_segments, wav_path)
                finally:
                    diarizer.unload()

            builder.record_stage(
                "phase_2_audio_diarization",
                StageExecutionStatus.SUCCESS,
                duration_sec=time.time() - t0,
                output_summary=f"Processed {len(audio_segments)} speech segments with word timings",
            )
        except Exception as ex:
            builder.record_stage("phase_2_audio_diarization", StageExecutionStatus.FAILED, time.time() - t0, error_message=str(ex))
            raise

        # -------------------------------------------------------------
        # Phase 3: Vision & Screen OCR Processing (GPU)
        # -------------------------------------------------------------
        t0 = time.time()
        try:
            console.print("[bold cyan][3/9][/bold cyan] Parsing screen context & OCR with bounded buffering...")
            if self.config.bounded_buffer:
                buffer = BoundedFrameBuffer(demuxer=self.demuxer)
                worker_mgr = WorkerIsolationManager() if self.config.isolate_gpu_workers else None
                v_cfg = {
                    "backend": "florence" if self.base_config.vision.device == "cuda" else "fallback",
                    "device": self.base_config.vision.device,
                    "detect_objects": self.base_config.vision.object_detection_enabled,
                }

                def _frame_callback(f_items, w_start, w_end):
                    if worker_mgr:
                        return worker_mgr.process_keyframes_isolated(f_items, config=v_cfg)
                    else:
                        vp = VisionProcessor(
                            backend=v_cfg["backend"],
                            device=v_cfg["device"],
                            detect_objects=v_cfg["detect_objects"],
                        )
                        res = [
                            vp.process_frame(Path(item["path"]), item["timestamp_sec"], item["frame_index"])
                            for item in f_items
                        ]
                        vp.unload()
                        return res

                optimal_timestamps = None
                if getattr(self.config, "sampling_mode", "fixed") == "adaptive":
                    early_chat = []
                    if chat_input and chat_input.exists():
                        try:
                            early_chat = self.chat_analyzer.parse_twitch_downloader_json(chat_input)
                        except Exception:
                            pass
                    opt_cfg = VisionConfig(
                        sampling_mode="adaptive",
                        sample_interval_sec=self.config.sample_interval_sec,
                        min_interval_sec=getattr(self.config, "min_interval_sec", 0.5),
                        max_interval_sec=getattr(self.config, "max_interval_sec", 5.0),
                        burst_window_sec=getattr(self.config, "burst_window_sec", 12.0),
                    )
                    optimizer = AdaptiveFrameOptimizer(config=opt_cfg)
                    optimal_timestamps = optimizer.compute_optimal_timestamps(
                        duration_sec=effective_duration,
                        chat_messages=early_chat,
                        audio_segments=audio_segments,
                    )
                    console.print(f"      [bold purple][OPTIMIZER][/bold purple] Computed {len(optimal_timestamps)} adaptive frame timestamps (vs ~{int(effective_duration / self.config.sample_interval_sec) + 1} fixed)")

                keyframes = buffer.process_stream_windowed(
                    media_path,
                    total_duration_sec=effective_duration,
                    sample_interval_sec=self.config.sample_interval_sec,
                    window_size_sec=self.config.window_size_sec,
                    frame_processor=_frame_callback,
                    purge_on_complete=True,
                    optimal_timestamps=optimal_timestamps,
                )
            else:
                vp = VisionProcessor(
                    backend="florence" if self.base_config.vision.device == "cuda" else "fallback",
                    device=self.base_config.vision.device,
                    detect_objects=self.base_config.vision.object_detection_enabled,
                )
                keyframes = []
                try:
                    for idx, f_path in enumerate(frames):
                        t_sec = idx * self.config.sample_interval_sec
                        kf = vp.process_frame(f_path, timestamp_sec=t_sec, frame_index=idx + 1)
                        keyframes.append(kf)
                finally:
                    vp.unload()

            builder.record_stage(
                "phase_3_vision_ocr",
                StageExecutionStatus.SUCCESS,
                duration_sec=time.time() - t0,
                output_summary=f"Parsed {len(keyframes)} scene keyframes with screen text and facecam bounding boxes",
            )
        except Exception as ex:
            builder.record_stage("phase_3_vision_ocr", StageExecutionStatus.FAILED, time.time() - t0, error_message=str(ex))
            raise

        # -------------------------------------------------------------
        # Phase 4: Chat Ingestion, Latency Calibration & Slang Discovery
        # -------------------------------------------------------------
        t0 = time.time()
        raw_chat: List[ChatMessage] = []
        try:
            console.print("[bold cyan][4/9][/bold cyan] Ingesting chat replay & calibrating dynamic latency...")
            if chat_input and chat_input.exists():
                raw_chat = self.chat_analyzer.parse_twitch_downloader_json(chat_input)

            if self.config.manual_latency_offset is not None:
                effective_lag = self.config.manual_latency_offset
            elif self.config.auto_latency and raw_chat:
                effective_lag = self.calibrator.compute_optimal_latency(
                    audio_segments=audio_segments,
                    chat_messages=raw_chat,
                    stream_duration_sec=effective_duration,
                )
            else:
                effective_lag = self.base_config.chat.latency_offset_sec

            self.chat_analyzer.latency_offset_sec = effective_lag
            chat_buckets = self.chat_analyzer.aggregate_into_buckets(
                raw_chat,
                stream_duration_sec=effective_duration,
                bucket_size_sec=self.base_config.chat.bucket_window_sec,
            )

            # Slang Discovery
            slang_count = 0
            if self.config.enable_adaptive_slang and raw_chat:
                try:
                    candidates, _ = self.slang_engine.process_chat_stream(
                        raw_chat, audio_segments=audio_segments, persist=True
                    )
                    slang_count = len(candidates)
                    builder.manifest.slang_terms_updated = slang_count
                except Exception as s_err:
                    logger.warning("Adaptive slang discovery degraded: %s", s_err)

            builder.record_stage(
                "phase_4_chat_and_slang",
                StageExecutionStatus.SUCCESS,
                duration_sec=time.time() - t0,
                output_summary=f"Calibrated lag {effective_lag:.2f}s, {len(raw_chat)} messages, {slang_count} slang candidates discovered",
            )
        except Exception as ex:
            builder.record_stage("phase_4_chat_and_slang", StageExecutionStatus.FAILED, time.time() - t0, error_message=str(ex))
            raise

        # -------------------------------------------------------------
        # Phase 5: Multimodal Fusion Matrix & Highlight Detection
        # -------------------------------------------------------------
        t0 = time.time()
        try:
            console.print("[bold cyan][5/9][/bold cyan] Computing Multimodal Fusion Matrix & Take Agreement Index...")
            analysis_result = self.fusion.build_matrix(
                stream_id=stream_id,
                duration_sec=effective_duration,
                audio_segments=audio_segments,
                visual_keyframes=keyframes,
                chat_buckets=chat_buckets,
            )
            analysis_result.metadata["latency_offset_sec"] = effective_lag

            # Export HTML Report
            html_output = out_dir / f"{stream_id}_grounding_report.html"
            export_html_report(analysis_result, html_output)
            builder.manifest.html_report_path = str(html_output.resolve())

            # Export Datasets (Parquet + JSONL)
            export_to_parquet(analysis_result, out_dir / f"{stream_id}_matrix.parquet")
            export_training_triples_jsonl(analysis_result, out_dir / f"{stream_id}_training_triples.jsonl")

            builder.populate_from_analysis(
                analysis_result,
                effective_duration,
                total_audio_segments=len(audio_segments),
                total_keyframes=len(keyframes),
                total_chat_messages=len(raw_chat),
                calibrated_delay_sec=effective_lag,
            )
            builder.record_stage(
                "phase_5_fusion_matrix",
                StageExecutionStatus.SUCCESS,
                duration_sec=time.time() - t0,
                output_summary=f"Built {len(analysis_result.fusion_slices)} fusion slices (Mean Agreement: {builder.manifest.take_agreement_mean:.2f})",
            )
        except Exception as ex:
            builder.record_stage("phase_5_fusion_matrix", StageExecutionStatus.FAILED, time.time() - t0, error_message=str(ex))
            raise

        # -------------------------------------------------------------
        # Phase 6: Sponsor & Brand Quantifier
        # -------------------------------------------------------------
        t0 = time.time()
        sponsor_report = None
        if self.config.enable_sponsor_quantifier:
            try:
                console.print("[bold cyan][6/9][/bold cyan] Scanning for brand mentions & sponsor engagement...")
                sponsor_report = self.sponsor_quantifier.analyze_sponsors(
                    stream_id=stream_id,
                    audio_segments=audio_segments,
                    keyframes=keyframes,
                    chat_buckets=chat_buckets,
                    chat_messages=raw_chat,
                )
                builder.manifest.sponsor_mentions_count = len(sponsor_report.sponsor_segments) if sponsor_report else 0
                builder.record_stage(
                    "phase_6_sponsor_quantifier",
                    StageExecutionStatus.SUCCESS,
                    duration_sec=time.time() - t0,
                    output_summary=f"Identified {builder.manifest.sponsor_mentions_count} sponsor segment moments",
                )
            except Exception as ex:
                logger.warning("Sponsor quantifier degraded: %s", ex)
                builder.record_stage("phase_6_sponsor_quantifier", StageExecutionStatus.DEGRADED, time.time() - t0, error_message=str(ex))
        else:
            builder.record_stage("phase_6_sponsor_quantifier", StageExecutionStatus.SKIPPED, 0.0)

        # -------------------------------------------------------------
        # Phase 7: Knowledge Graph Extraction, Web Grounding & Stance Tracking
        # -------------------------------------------------------------
        t0 = time.time()
        claims = []
        grounded_claims = []
        shifts_detected = 0
        if self.config.enable_web_grounding or self.config.enable_stance_tracking:
            try:
                console.print("[bold cyan][7/9][/bold cyan] Extracting claims, fact-checking web sources, and tracking stance shifts...")
                claims = self.claim_extractor.extract_claims(audio_segments=audio_segments, keyframes=keyframes)
                builder.manifest.claims_extracted_count = len(claims)

                if self.config.enable_web_grounding and claims:
                    grounded_claims = self.web_grounder.ground_claims_batch(claims)
                    builder.manifest.grounded_claims_count = len(grounded_claims)

                if self.config.enable_stance_tracking and claims:
                    for clm in claims:
                        record = self.stance_tracker.record_claim(clm, stream_id=stream_id)
                        shifts = self.stance_tracker.detect_shifts(clm.target_entity)
                        shifts_detected += len(shifts)
                    self.stance_tracker.save()
                    builder.manifest.stance_shifts_count = shifts_detected

                builder.record_stage(
                    "phase_7_knowledge_and_stance",
                    StageExecutionStatus.SUCCESS,
                    duration_sec=time.time() - t0,
                    output_summary=f"Extracted {len(claims)} claims, grounded {len(grounded_claims)}, tracked {shifts_detected} stance shifts",
                )
            except Exception as ex:
                logger.warning("Knowledge grounding degraded: %s", ex)
                builder.record_stage("phase_7_knowledge_and_stance", StageExecutionStatus.DEGRADED, time.time() - t0, error_message=str(ex))
        else:
            builder.record_stage("phase_7_knowledge_and_stance", StageExecutionStatus.SKIPPED, 0.0)

        # -------------------------------------------------------------
        # Phase 8: Autonomous Multi-Agent Short Studio
        # -------------------------------------------------------------
        t0 = time.time()
        short_packages = []
        if self.config.enable_short_production:
            try:
                console.print("[bold cyan][8/9][/bold cyan] Launching Autonomous Multi-Agent Short Studio...")
                shorts_out_dir = out_dir / "shorts"
                packages_and_envs = self.short_orchestrator.produce_shorts(
                    analysis=analysis_result,
                    output_dir=shorts_out_dir,
                    video_path=media_path,
                    fusion_slices=analysis_result.fusion_slices,
                    keyframes=keyframes,
                    audio_segments=audio_segments,
                    chat_messages=raw_chat,
                    claims=claims,
                    sponsor_segments=sponsor_report.sponsor_segments if sponsor_report else None,
                    top_k=self.config.short_candidate_count,
                    dry_run=self.config.dry_run_shorts,
                )
                short_packages = [pkg for pkg, _ in packages_and_envs]
                builder.manifest.shorts_produced_count = len(short_packages)

                builder.record_stage(
                    "phase_8_multi_agent_shorts",
                    StageExecutionStatus.SUCCESS,
                    duration_sec=time.time() - t0,
                    output_summary=f"Assembled {len(short_packages)} autonomous 9:16 vertical shorts packages",
                )
            except Exception as ex:
                logger.warning("Multi-agent short production degraded: %s", ex)
                builder.record_stage("phase_8_multi_agent_shorts", StageExecutionStatus.DEGRADED, time.time() - t0, error_message=str(ex))
        else:
            builder.record_stage("phase_8_multi_agent_shorts", StageExecutionStatus.SKIPPED, 0.0)

        # -------------------------------------------------------------
        # Phase 9: Manifest Packaging & Event Bus Dispatch
        # -------------------------------------------------------------
        t0 = time.time()
        console.print("[bold cyan][9/9][/bold cyan] Finalizing Full-Spectrum Manifest & Event Dispatch...")
        builder.manifest.total_pipeline_duration_sec = round(time.time() - start_overall, 2)
        builder.record_stage(
            "phase_9_packaging",
            StageExecutionStatus.SUCCESS,
            duration_sec=time.time() - t0,
            output_summary=f"Pipeline finished in {builder.manifest.total_pipeline_duration_sec:.1f}s",
        )

        manifest_path = builder.save()

        # Emit completion envelope if broadcaster present
        if self.broadcaster:
            env = StreamFusionEnvelope[Dict[str, Any]](
                event_type=StreamEventType.PIPELINE_COMPLETE,
                stream_id=stream_id,
                payload=builder.manifest.model_dump(mode="json"),
            )
            try:
                import asyncio
                loop = asyncio.get_running_loop()
                loop.create_task(self.broadcaster.broadcast(env))
            except RuntimeError:
                pass

        console.print(f"[bold green]✓ Full-Spectrum Synergy Analysis Complete![/bold green] Manifest: {manifest_path.name}")
        return analysis_result, builder.manifest
