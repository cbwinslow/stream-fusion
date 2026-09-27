"""StreamFusion end-to-end multimodal pipeline orchestrator."""

from pathlib import Path
from typing import Optional
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
    ) -> StreamAnalysisResult:
        """Executes the complete phased sequential pipeline."""
        out_path = output_dir or self.config.storage.output_dir
        out_path.mkdir(parents=True, exist_ok=True)
        stream_id = media_input.stem

        # -------------------------------------------------------------
        # Phase 1: Ingestion & Demuxing
        # -------------------------------------------------------------
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
        console.print("[bold cyan][2/5][/bold cyan] Running Speech Transcription & Voice Diarization...")
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
        console.print(f"      [green][OK][/green] Processed {len(audio_segments)} diarized audio segments")

        # -------------------------------------------------------------
        # Phase 3: Vision & Screen OCR Processing (GPU)
        # -------------------------------------------------------------
        console.print("[bold cyan][3/5][/bold cyan] Analyzing screen context & OCR on keyframes...")
        vision_processor = VisionProcessor(
            backend="florence" if self.config.vision.device == "cuda" else "fallback",
            device=self.config.vision.device,
        )
        keyframes = []
        for idx, f_path in enumerate(frames):
            t_sec = idx * self.config.vision.sample_interval_sec
            kf = vision_processor.process_frame(f_path, timestamp_sec=t_sec, frame_index=idx + 1)
            keyframes.append(kf)
        vision_processor.unload()
        console.print(f"      [green][OK][/green] Parsed {len(keyframes)} visual scene keyframes")

        # -------------------------------------------------------------
        # Phase 4: Chat Ingestion & Dynamic Latency Calibration
        # -------------------------------------------------------------
        console.print("[bold cyan][4/5][/bold cyan] Ingesting chat replay & calculating stream delay...")
        raw_chat = []
        if chat_input and chat_input.exists():
            raw_chat = self.chat_analyzer.parse_twitch_downloader_json(chat_input)

        # Calculate exact stream latency delay offset
        optimal_lag = self.calibrator.compute_optimal_latency(
            audio_segments=audio_segments,
            chat_messages=raw_chat,
            stream_duration_sec=effective_duration,
        )
        console.print(f"      [green][OK][/green] Calibrated broadcast delay: [bold yellow]{optimal_lag:.2f}s[/bold yellow]")

        self.chat_analyzer.latency_offset_sec = optimal_lag
        chat_buckets = self.chat_analyzer.aggregate_into_buckets(
            raw_chat,
            stream_duration_sec=effective_duration,
            bucket_size_sec=self.config.chat.bucket_window_sec,
        )

        # -------------------------------------------------------------
        # Phase 5: Temporal Fusion Matrix & Highlight Detection
        # -------------------------------------------------------------
        console.print("[bold cyan][5/5][/bold cyan] Merging Multimodal Temporal Matrix...")
        result = self.fusion.build_matrix(
            stream_id=stream_id,
            duration_sec=effective_duration,
            audio_segments=audio_segments,
            visual_keyframes=keyframes,
            chat_buckets=chat_buckets,
        )

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

        # Auto-render top highlights as 9:16 vertical shorts
        if result.highlights:
            console.print(f"[bold purple]Rendering {len(result.highlights[:2])} auto-detected highlights as 9:16 vertical shorts...[/bold purple]")
            from stream_fusion.export.clipper import VerticalHighlightClipper
            clipper = VerticalHighlightClipper()
            shorts_dir = out_path / "shorts"
            for h_idx, hl in enumerate(result.highlights[:2]):
                short_out = shorts_dir / f"{stream_id}_short_{h_idx+1}.mp4"
                try:
                    clipper.export_highlight_short(
                        video_path=media_input,
                        start_sec=hl.get("clip_start_sec", hl["timestamp_sec"]),
                        end_sec=hl.get("clip_end_sec", hl["timestamp_sec"] + 10.0),
                        output_path=short_out,
                    )
                    console.print(f"      [bold green][OK] Highlight Short #{h_idx+1}:[/bold green] {short_out.resolve()}")
                except Exception as e:
                    console.print(f"      [yellow]Shorts export skipped: {e}[/yellow]")

        return result
