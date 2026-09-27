"""Telemetry and System Resource Monitoring for StreamFusion (Spec 15)."""

from datetime import datetime, timezone
import os
import time
from typing import Any, Dict, Optional, Tuple

try:
    import psutil  # type: ignore
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class SystemResourceProbe:
    """Probes system memory (RAM) and GPU accelerator memory (VRAM)."""

    @staticmethod
    def get_memory_usage_mb() -> Tuple[float, float]:
        """Returns (process_ram_mb, vram_mb)."""
        ram_mb = 0.0
        if HAS_PSUTIL:
            try:
                proc = psutil.Process(os.getpid())
                ram_mb = float(proc.memory_info().rss) / (1024.0 * 1024.0)
            except Exception:
                pass

        vram_mb = 0.0
        if HAS_TORCH and torch.cuda.is_available():
            try:
                vram_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
            except Exception:
                pass

        return round(ram_mb, 2), round(vram_mb, 2)


class StageTimer:
    """Scoped context manager measuring execution time, peak memory, and exceptions."""

    def __init__(self, stage_name: str, collector: Optional["TelemetryCollector"] = None):
        self.stage_name = stage_name
        self.collector = collector
        self.start_wall: float = 0.0
        self.start_iso: str = ""
        self.end_iso: str = ""
        self.duration_sec: float = 0.0
        self.start_ram: float = 0.0
        self.start_vram: float = 0.0
        self.peak_ram: float = 0.0
        self.peak_vram: float = 0.0
        self.status: str = "SUCCESS"
        self.error_message: Optional[str] = None

    def __enter__(self) -> "StageTimer":
        self.start_wall = time.perf_counter()
        self.start_iso = datetime.now(timezone.utc).isoformat()
        self.start_ram, self.start_vram = SystemResourceProbe.get_memory_usage_mb()
        self.peak_ram = self.start_ram
        self.peak_vram = self.start_vram
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.duration_sec = round(time.perf_counter() - self.start_wall, 3)
        self.end_iso = datetime.now(timezone.utc).isoformat()

        end_ram, end_vram = SystemResourceProbe.get_memory_usage_mb()
        self.peak_ram = max(self.peak_ram, end_ram)
        self.peak_vram = max(self.peak_vram, end_vram)

        if exc_type is not None:
            self.status = "ERROR"
            self.error_message = str(exc_val)
        else:
            self.status = "SUCCESS"

        if self.collector:
            self.collector.record_stage(
                stage_name=self.stage_name,
                duration_sec=self.duration_sec,
                start_iso=self.start_iso,
                end_iso=self.end_iso,
                ram_mb=self.peak_ram,
                vram_mb=self.peak_vram,
                status=self.status,
                error_message=self.error_message,
            )


class TelemetryCollector:
    """Aggregates stage timings, resource usage, and quality metrics across a run."""

    def __init__(self, run_id: Optional[str] = None):
        self.run_id = run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        self.stages: Dict[str, Dict[str, Any]] = {}
        self.quality_metrics: Dict[str, Any] = {}
        self.start_time = time.perf_counter()

    def stage(self, stage_name: str) -> StageTimer:
        """Returns a context manager stage timer."""
        return StageTimer(stage_name, collector=self)

    def record_stage(
        self,
        stage_name: str,
        duration_sec: float,
        start_iso: str,
        end_iso: str,
        ram_mb: float,
        vram_mb: float,
        status: str = "SUCCESS",
        error_message: Optional[str] = None,
        throughput_name: Optional[str] = None,
        throughput_value: Optional[float] = None,
    ) -> None:
        """Records telemetry for a single stage."""
        self.stages[stage_name] = {
            "stage_name": stage_name,
            "duration_sec": duration_sec,
            "start_time_iso": start_iso,
            "end_time_iso": end_iso,
            "ram_mb_peak": ram_mb,
            "vram_mb_peak": vram_mb,
            "status": status,
            "error_message": error_message,
            "throughput_metric_name": throughput_name,
            "throughput_value": throughput_value,
        }

    def record_quality_metric(self, name: str, value: Any) -> None:
        """Records analytical accuracy or output yield metrics."""
        self.quality_metrics[name] = value

    def get_total_duration(self) -> float:
        """Returns total elapsed seconds since collector initialization."""
        return round(time.perf_counter() - self.start_time, 3)
