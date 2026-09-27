"""StreamFusion end-to-end multimodal pipeline orchestrator."""

from pathlib import Path
from typing import List, Optional
from rich.console import Console

from stream_fusion.models.schemas import StreamAnalysisResult
from stream_fusion.config import StreamFusionConfig
from stream_fusion.ingest.downloader import StreamDownloader
from stream_fusion.ingest.demuxer import MediaDemuxer
from stream_fusion.audio.transcriber import AudioTranscriber
from stream_fusion.audio.diarizer import ReactionDiarizer
from stream_fusion.vision.processor import VisionProcessor
from stream_fusion.chat.analyzer import ChatAnalyzer
from stream_fusion.chat.calibrator import LatencyCalibrator
from stream_fusion.fusion.matrix import FusionEngine
from stream_fusion.export.html_report import export_html_report
from stream_fusion.export.dataset import export_to_parquet, export_training_triples_jsonl
from stream_fusion.export.clipper import VerticalHighlightClipper
from contextlib import nullcontext
from stream_fusion.checkpoint.manager import CheckpointManager
from stream_fusion.monitoring.telemetry import TelemetryCollector

console = Console()


class StreamPipeline:
    """Orchestrates end-to-end VOD and chat replay analysis."""

    def __init__(self, config: Optional[StreamFusionConfig] = None):
        self.config = config or StreamFusionConfig()
        self.downloader = StreamDownloader()
        self.demuxer = MediaDemuxer()
        self.chat_analyzer = ChatAnalyzer()
        self.calibrator = LatencyCalibrator()
        self.fusion = FusionEngine(bucket_size_sec=self.config.chat.bucket_window_sec)

    def run(
        self,
        media_input: Path,
        chat_input: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        duration_sec: Optional[float] = None,
        latency_offset: Optional[float] = None,
        auto_calibrate_latency: Optional[bool] = None,
        cache_dir: Optional[Path] = None,
        chunk_index: Optional[int] = None,
        telemetry: Optional[TelemetryCollector] = None,
    ) -> StreamAnalysisResult:
        """Executes the complete phased sequential pipeline."""
        out_path = output_dir or self.config.storage.output_dir
        out_path.mkdir(parents=True, exist_ok=True)
        stream_id = media_input.stem
        checkpoint_mgr = (
            CheckpointManager(cache_dir=cache_dir, stream_id=stream_id)
            if cache_dir
            else None
        )
        c_idx = chunk_index if chunk_index is not None else 0

        # -------------------------------------------------------------
        # Phase 1: Ingestion & Demuxing
        # -------------------------------------------------------------
        with (telemetry.stage("phase_1_demuxing") if telemetry else nullcontext()):
            console.print(f"[bold cyan][1/5][/bold cyan] Demuxing media: {media_input.name}")
            wav_path = out_path / f"{stream_id}_audio_16k.wav"
            self.demuxer.extract_audio_16k_mono(media_input, wav_path)

            frames_dir = out_path / f"{stream_id}_keyframes"
            frames = self.demuxer.extract_frames_at_interval(
                media_input, frames_dir, interval_sec=self.config.vision.sample_interval_sec
            )
            effective_duration = duration_sec or (len(frames) * self.config.vision.sample_interval_sec)

        # -------------------------------------------------------------
        # Phase 2: Speech Transcription & Reaction Diarization (GPU)
        # -------------------------------------------------------------
        with (telemetry.stage("phase_2_speech_transcription") if telemetry else nullcontext()):
            console.print("[bold cyan][2/5][/bold cyan] Running Speech Transcription & Voice Diarization...")
            cached_audio = checkpoint_mgr.load_chunk_audio(c_idx) if checkpoint_mgr else None
            if cached_audio is not None:
                audio_segments = cached_audio
                console.print(f"      [green][RESUMED][/green] Loaded {len(audio_segments)} audio segments from checkpoint")
            else:
                transcriber = AudioTranscriber(
                    model_size=self.config.audio.whisper_model,
                    device=self.config.audio.device,
                    compute_type=self.config.audio.compute_type,
                )
                audio_segments = transcriber.transcribe(wav_path)
                transcriber.unload()

                diarizer = ReactionDiarizer(
                    hf_token=self.config.audio.hf_token, device=self.config.audio.device
                )
                audio_segments = diarizer.diarize_and_tag(audio_segments, wav_path)
                diarizer.unload()
                if checkpoint_mgr:
                    checkpoint_mgr.save_chunk_audio(c_idx, audio_segments)
                console.print(f"      [green][OK][/green] Processed {len(audio_segments)} diarized audio segments")

        # -------------------------------------------------------------
        # Phase 3: Vision & Screen OCR Processing (GPU)
        # -------------------------------------------------------------
        with (telemetry.stage("phase_3_vision_ocr") if telemetry else nullcontext()):
            console.print("[bold cyan][3/5][/bold cyan] Analyzing screen context & OCR on keyframes...")
            cached_vision = checkpoint_mgr.load_chunk_vision(c_idx) if checkpoint_mgr else None
            if cached_vision is not None:
                keyframes = cached_vision
                console.print(f"      [green][RESUMED][/green] Loaded {len(keyframes)} visual scene keyframes from checkpoint")
            else:
                vision_processor = VisionProcessor(
                    backend="florence" if self.config.vision.device == "cuda" else "fallback",
                    device=self.config.vision.device,
                    detect_objects=self.config.vision.object_detection_enabled,
                )
                keyframes = []
                for idx, f_path in enumerate(frames):
                    t_sec = idx * self.config.vision.sample_interval_sec
                    kf = vision_processor.process_frame(f_path, timestamp_sec=t_sec, frame_index=idx + 1)
                    keyframes.append(kf)
                vision_processor.unload()
                if checkpoint_mgr:
                    checkpoint_mgr.save_chunk_vision(c_idx, keyframes)
                console.print(f"      [green][OK][/green] Parsed {len(keyframes)} visual scene keyframes")

        # -------------------------------------------------------------
        # Phase 4: Chat Ingestion & Dynamic Latency Calibration
        # -------------------------------------------------------------
        with (telemetry.stage("phase_4_chat_calibration") if telemetry else nullcontext()):
            console.print("[bold cyan][4/5][/bold cyan] Ingesting chat replay & calculating stream delay...")
            raw_chat = []
            if chat_input and chat_input.exists():
                raw_chat = self.chat_analyzer.parse_twitch_downloader_json(chat_input)

            use_auto_calib = (
                auto_calibrate_latency
                if auto_calibrate_latency is not None
                else self.config.chat.auto_calibrate_latency
            )

            if latency_offset is not None:
                effective_lag = latency_offset
                console.print(f"      [green][OK][/green] Using manual broadcast delay: [bold yellow]{effective_lag:.2f}s[/bold yellow]")
            elif use_auto_calib:
                effective_lag = self.calibrator.compute_optimal_latency(
                    audio_segments=audio_segments,
                    chat_messages=raw_chat,
                    stream_duration_sec=effective_duration,
                )
                console.print(f"      [green][OK][/green] Calibrated broadcast delay: [bold yellow]{effective_lag:.2f}s[/bold yellow]")
            else:
                effective_lag = self.config.chat.latency_offset_sec
                console.print(f"      [green][OK][/green] Using config broadcast delay: [bold yellow]{effective_lag:.2f}s[/bold yellow]")

            self.chat_analyzer.latency_offset_sec = effective_lag
            chat_buckets = self.chat_analyzer.aggregate_into_buckets(
                raw_chat,
                stream_duration_sec=effective_duration,
                bucket_size_sec=self.config.chat.bucket_window_sec,
            )
            if checkpoint_mgr:
                checkpoint_mgr.save_chunk_chat(c_idx, chat_buckets)

        # -------------------------------------------------------------
        # Phase 5: Temporal Fusion Matrix & Highlight Detection
        # -------------------------------------------------------------
        with (telemetry.stage("phase_5_fusion_matrix") if telemetry else nullcontext()):
            console.print("[bold cyan][5/5][/bold cyan] Merging Multimodal Temporal Matrix...")
            result = self.fusion.build_matrix(
                stream_id=stream_id,
                duration_sec=effective_duration,
                audio_segments=audio_segments,
                visual_keyframes=keyframes,
                chat_buckets=chat_buckets,
            )

            if checkpoint_mgr:
                checkpoint_mgr.save_final_result(result)
                checkpoint_mgr.mark_chunk_completed(c_idx)

            # Export HTML Dashboard
            html_output = out_path / f"{stream_id}_grounding_report.html"
            export_html_report(result, html_output)
            console.print(f"[bold green][OK] Grounding Report Generated:[/bold green] {html_output.resolve()}")

            # Export Columnar Parquet
            parquet_output = out_path / f"{stream_id}_matrix.parquet"
            export_to_parquet(result, parquet_output)
            console.print(f"[bold green][OK] Columnar Parquet Matrix:[/bold green] {parquet_output.resolve()}")

            # Export Hugging Face Training Triples
            jsonl_output = out_path / f"{stream_id}_training_triples.jsonl"
            export_training_triples_jsonl(result, jsonl_output)
            console.print(f"[bold green][OK] Multimodal Training Triples:[/bold green] {jsonl_output.resolve()}")

            # Auto-render top highlights as 9:16 vertical shorts with karaoke subtitles and facecam localization
            if result.highlights:
                console.print(f"[bold purple]Rendering {len(result.highlights[:2])} auto-detected highlights as 9:16 vertical shorts...[/bold purple]")
                clipper = VerticalHighlightClipper()
                shorts_dir = out_path / "shorts"
                for h_idx, hl in enumerate(result.highlights[:2]):
                    short_out = shorts_dir / f"{stream_id}_short_{h_idx+1}.mp4"
                    clip_start = hl.get("clip_start_sec", hl["timestamp_sec"])
                    clip_end = hl.get("clip_end_sec", hl["timestamp_sec"] + 10.0)

                    # Extract word timings for karaoke subtitles
                    hl_words = []
                    for seg in audio_segments:
                        if seg.words and (seg.start_sec <= clip_end and seg.end_sec >= clip_start):
                            for w in seg.words:
                                if clip_start <= w.start <= clip_end:
                                    hl_words.append(w)

                    # Find facecam box from keyframe near this highlight
                    facecam_box = None
                    for kf in keyframes:
                        if abs(kf.timestamp_sec - hl["timestamp_sec"]) <= 3.0 and kf.detected_objects:
                            facecam_box = VisionProcessor.locate_streamer_facecam(kf.detected_objects)
                            if facecam_box:
                                break

                    try:
                        clipper.export_highlight_short(
                            video_path=media_input,
                            start_sec=clip_start,
                            end_sec=clip_end,
                            output_path=short_out,
                            words=hl_words if hl_words else None,
                            facecam_box=facecam_box,
                            burn_subtitles=bool(hl_words),
                        )
                        console.print(f"      [bold green][OK] Highlight Short #{h_idx+1}:[/bold green] {short_out.resolve()}")
                    except Exception as e:
                        console.print(f"      [yellow]Shorts export skipped: {e}[/yellow]")

        if telemetry:
            telemetry.record_quality_metric("total_chat_messages", result.total_chat_messages)
            telemetry.record_quality_metric("slices_count", len(result.slices))
            telemetry.record_quality_metric("highlights_count", len(result.highlights))
            telemetry.record_quality_metric("keyframes_count", len(keyframes))
            telemetry.record_quality_metric("audio_segments_count", len(audio_segments))

        return result

    def run_chunked(
        self,
        media_input: Path,
        chunk_duration_sec: float = 1800.0,
        chat_input: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        total_duration_sec: Optional[float] = None,
        latency_offset: Optional[float] = None,
        cache_dir: Optional[Path] = None,
    ) -> List[StreamAnalysisResult]:
        """Processes a long VOD stream in sequential chunks."""
        out_path = output_dir or self.config.storage.output_dir
        out_path.mkdir(parents=True, exist_ok=True)
        stream_id = media_input.stem
        
        # Estimate duration if not provided
        est_duration = total_duration_sec or 3600.0
        num_chunks = max(1, int((est_duration + chunk_duration_sec - 1) // chunk_duration_sec))
        console.print(f"[bold cyan]Chunked Processing:[/bold cyan] {num_chunks} chunks of {chunk_duration_sec:.0f}s each")

        checkpoint_mgr = (
            CheckpointManager(
                cache_dir=cache_dir,
                stream_id=stream_id,
                chunk_duration_sec=chunk_duration_sec,
                total_chunks_expected=num_chunks,
            )
            if cache_dir
            else None
        )

        results = []
        for i in range(num_chunks):
            chunk_start = i * chunk_duration_sec
            chunk_out = out_path / f"chunk_{i:03d}"
            console.print(f"[bold yellow]Processing Chunk {i+1}/{num_chunks} [{chunk_start:.0f}s - {chunk_start+chunk_duration_sec:.0f}s][/bold yellow]")

            if checkpoint_mgr and checkpoint_mgr.is_chunk_completed(i):
                console.print(f"      [green][SKIP][/green] Chunk {i+1}/{num_chunks} already completed. Resuming...")
                cached_res = checkpoint_mgr.load_final_result()
                if cached_res:
                    results.append(cached_res)
                continue

            res = self.run(
                media_input=media_input,
                chat_input=chat_input,
                output_dir=chunk_out,
                duration_sec=chunk_duration_sec,
                latency_offset=latency_offset,
                cache_dir=cache_dir,
                chunk_index=i,
            )
            results.append(res)
        return results
