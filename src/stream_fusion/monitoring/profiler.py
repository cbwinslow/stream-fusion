"""Production Run Profiler and Telemetry Benchmark (Spec 30).

Computes end-to-end efficiency metrics, frame reduction ratios, and climax coverage
for production-scale full stream analysis runs.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from stream_fusion.models.schemas import ProductionBenchmarkReport
from stream_fusion.monitoring.telemetry import SystemResourceProbe


class ProductionRunProfiler:
    """Collects and synthesizes performance benchmarks for production VOD runs."""

    def __init__(self, stream_id: str, duration_sec: float):
        self.stream_id = stream_id
        self.duration_sec = max(0.1, duration_sec)
        self.fps: float = 60.0
        self.sampling_mode: str = "adaptive"
        self.sample_interval_sec: float = 2.0
        self.analyzed_timestamps: List[float] = []
        self.burst_intervals: List[Tuple[float, float]] = []
        self.peak_ram_mb: float = 0.0
        self.peak_vram_mb: float = 0.0

    def record_sampling_run(
        self,
        analyzed_timestamps: List[float],
        sampling_mode: str = "adaptive",
        sample_interval_sec: float = 2.0,
        burst_intervals: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        """Records the timestamps extracted and analyzed during the run."""
        self.analyzed_timestamps = sorted(analyzed_timestamps)
        self.sampling_mode = sampling_mode
        self.sample_interval_sec = sample_interval_sec
        self.burst_intervals = burst_intervals or []

        # Probe current peak memory usage
        ram, vram = SystemResourceProbe.get_memory_usage_mb()
        self.peak_ram_mb = max(self.peak_ram_mb, ram)
        self.peak_vram_mb = max(self.peak_vram_mb, vram)

    def calculate_burst_coverage(self) -> float:
        """Calculates percentage of detected burst intervals covered by at least one keyframe."""
        if not self.burst_intervals:
            return 100.0

        covered = 0
        for s, e in self.burst_intervals:
            if any(s <= t <= e for t in self.analyzed_timestamps):
                covered += 1

        return round((covered / len(self.burst_intervals)) * 100.0, 2)

    def generate_report(self) -> ProductionBenchmarkReport:
        """Synthesizes all run telemetry into a structured ProductionBenchmarkReport."""
        total_potential = int(self.duration_sec * self.fps)
        fixed_step = max(0.1, self.sample_interval_sec)
        fixed_frames = int(self.duration_sec / fixed_step) + 1
        actual_frames = len(self.analyzed_timestamps)

        frames_saved = max(0, fixed_frames - actual_frames)
        reduction_pct = (
            round((frames_saved / fixed_frames) * 100.0, 2)
            if fixed_frames > 0
            else 0.0
        )
        # Average inference cost model: ~80ms GPU inference per frame (Florence-2 / OCR)
        est_gpu_saved = round(frames_saved * 0.08, 2)
        coverage = self.calculate_burst_coverage()

        return ProductionBenchmarkReport(
            stream_id=self.stream_id,
            duration_sec=round(self.duration_sec, 2),
            total_potential_frames=total_potential,
            fixed_sampling_frames=fixed_frames,
            actual_analyzed_frames=actual_frames,
            sampling_mode=self.sampling_mode,
            frames_saved=frames_saved,
            reduction_pct=reduction_pct,
            estimated_gpu_time_saved_sec=est_gpu_saved,
            burst_zone_coverage_pct=coverage,
            peak_ram_mb=round(self.peak_ram_mb, 2),
            peak_vram_mb=round(self.peak_vram_mb, 2),
        )

    def export_json(self, output_path: Path) -> Path:
        """Exports the benchmark report as JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report = self.generate_report()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, indent=2)
        return output_path

    def format_markdown_summary(self) -> str:
        """Generates an executive markdown table of performance telemetry."""
        rep = self.generate_report()
        return (
            f"### Production Run Benchmark: `{rep.stream_id}`\n"
            f"- **Stream Duration**: {rep.duration_sec}s\n"
            f"- **Sampling Mode**: `{rep.sampling_mode}`\n"
            f"- **Total 60fps Frames**: {rep.total_potential_frames:,}\n"
            f"- **Frames in Fixed Mode**: {rep.fixed_sampling_frames:,}\n"
            f"- **Actual Frames Analyzed**: {rep.actual_analyzed_frames:,}\n"
            f"- **Frames Saved**: **{rep.frames_saved:,}** ({rep.reduction_pct}% reduction)\n"
            f"- **Est. GPU Time Saved**: ~{rep.estimated_gpu_time_saved_sec:.1f}s\n"
            f"- **Burst Zone Coverage**: {rep.burst_zone_coverage_pct}%\n"
            f"- **Peak RAM / VRAM**: {rep.peak_ram_mb} MB / {rep.peak_vram_mb} MB\n"
        )
