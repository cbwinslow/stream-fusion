"""Run Real Stream Production Benchmark with Adaptive Sampling (Spec 30).

Executes a full end-to-end pipeline run on real Asmongold broadcast footage (asmon_sample_60s.mp4)
and Twitch chat replay (sample_asmon_chat.json), comparing fixed vs adaptive frame density
and recording a ProductionBenchmarkReport.
"""

import json
from pathlib import Path
import sys
import time

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rich.console import Console
from rich.table import Table

from stream_fusion.config import StreamFusionConfig
from stream_fusion.models.schemas import ProductionBenchmarkReport
from stream_fusion.monitoring.profiler import ProductionRunProfiler
from stream_fusion.pipeline import StreamPipeline
from stream_fusion.vision.optimizer import AdaptiveFrameOptimizer

console = Console(force_terminal=True, legacy_windows=False)


def run_benchmark():
    root = Path(__file__).resolve().parent.parent
    video_path = root / "asmon_sample_60s.mp4"
    chat_path = root / "sample_asmon_chat.json"
    output_dir = root / "benchmark_production_run"
    cache_dir = root / "benchmark_cache"

    if not video_path.exists():
        console.print(f"[bold red]Error: Video file not found: {video_path}[/bold red]")
        return
    if not chat_path.exists():
        console.print(f"[bold red]Error: Chat file not found: {chat_path}[/bold red]")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    console.print("=" * 80, style="bold cyan")
    console.print("  [STREAMFUSION] REAL PRODUCTION BENCHMARK (SPEC 30 ADAPTIVE DENSITY)", style="bold green")
    console.print(f"  VOD Input:  {video_path.name} ({video_path.stat().st_size / (1024*1024):.1f} MB)", style="white")
    console.print(f"  Chat Input: {chat_path.name} ({chat_path.stat().st_size / (1024*1024):.1f} MB)", style="white")
    console.print("=" * 80, style="bold cyan")

    # 1. Configure for Adaptive Density Run
    config = StreamFusionConfig()
    config.audio.whisper_model = "tiny"
    config.audio.device = "cpu"
    config.audio.compute_type = "int8"
    config.vision.device = "cpu"
    config.vision.sampling_mode = "adaptive"
    config.vision.min_interval_sec = 0.5   # 2 FPS during bursts
    config.vision.max_interval_sec = 4.0   # 0.25 FPS during idle
    config.vision.burst_window_sec = 8.0
    config.execution.bounded_buffering = True
    config.execution.window_size_sec = 20.0

    pipeline = StreamPipeline(config=config)

    t_start = time.time()
    console.print("\n[bold yellow]>>> Starting Full Pipeline Execution...[/bold yellow]")

    # Analyze first 30 seconds of real broadcast
    duration_to_test = 30.0
    result = pipeline.run(
        media_input=video_path,
        chat_input=chat_path,
        output_dir=output_dir,
        duration_sec=duration_to_test,
        cache_dir=cache_dir,
        chunk_index=0,
    )
    total_run_time = time.time() - t_start

    # 2. Collect Profiler Telemetry
    profiler = ProductionRunProfiler(stream_id=result.stream_id, duration_sec=result.duration_sec)

    from stream_fusion.checkpoint.manager import CheckpointManager
    cp_mgr = CheckpointManager(cache_dir=cache_dir, stream_id=result.stream_id)
    cached_kfs = cp_mgr.load_chunk_vision(0)
    if cached_kfs:
        analyzed_timestamps = [round(kf.timestamp_sec, 3) for kf in cached_kfs]
    else:
        analyzed_timestamps = [round(s.start_sec, 3) for s in result.slices]
    analyzed_timestamps = sorted(list(set(analyzed_timestamps)))

    profiler.record_sampling_run(
        analyzed_timestamps=analyzed_timestamps,
        sampling_mode="adaptive",
        sample_interval_sec=2.0,
    )

    report: ProductionBenchmarkReport = profiler.generate_report()
    report_file = output_dir / "production_benchmark_report.json"
    profiler.export_json(report_file)

    # 3. Present Results Table
    console.print("\n" + "=" * 80, style="bold cyan")
    console.print("  [BENCHMARK] PRODUCTION BENCHMARK REPORT RESULTS", style="bold green")
    console.print("=" * 80, style="bold cyan")

    table = Table(title="Efficiency & Compute Reduction", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold yellow")
    table.add_column("Context / Baseline", style="dim white")

    table.add_row("Stream Duration Analyzed", f"{report.duration_sec}s", "Real broadcast sample")
    table.add_row("Total Potential 60 FPS Frames", f"{report.total_potential_frames:,}", "Raw demuxed frames at 60 FPS")
    table.add_row("Fixed 2.0s Sampling Frames", f"{report.fixed_sampling_frames:,}", "Legacy fixed interval mode")
    table.add_row("Actual Analyzed Slices", f"{report.actual_analyzed_frames:,}", "Dynamically densified around speech/chat")
    table.add_row("Frames Saved vs Fixed", f"{report.frames_saved:,}", f"{report.reduction_pct}% frame reduction")
    table.add_row("Est. GPU Compute Time Saved", f"~{report.estimated_gpu_time_saved_sec:.1f}s", "Based on ~80ms GPU VLM inference/frame")
    table.add_row("Burst Zone Highlight Coverage", f"{report.burst_zone_coverage_pct}%", "Climax moments fully captured")
    table.add_row("Peak Process RAM", f"{report.peak_ram_mb:.1f} MB", "Bounded memory footprint")
    table.add_row("Total Benchmark Runtime", f"{total_run_time:.2f}s", "Demux + Whisper + Chat + Fusion + Parquet")

    console.print(table)

    console.print(f"\n[bold green][OK] Report JSON exported to:[/bold green] {report_file}")

    # Check outputs
    parquet_file = output_dir / f"{result.stream_id}_matrix.parquet"
    html_file = output_dir / f"{result.stream_id}_grounding_report.html"
    if parquet_file.exists():
        console.print(f"[bold green][OK] Parquet Multimodal Matrix:[/bold green] {parquet_file} ({parquet_file.stat().st_size} bytes)")
    if html_file.exists():
        console.print(f"[bold green][OK] HTML Grounding Report:[/bold green] {html_file} ({html_file.stat().st_size} bytes)")


if __name__ == "__main__":
    run_benchmark()
