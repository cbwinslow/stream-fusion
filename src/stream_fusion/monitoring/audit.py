"""Persistent Audit Registry, Run Tracking, and Regression Detection (Spec 15)."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import subprocess
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class StageTelemetry(BaseModel):
    stage_name: str
    duration_sec: float
    start_time_iso: str
    end_time_iso: str
    ram_mb_peak: float
    vram_mb_peak: float
    status: str = "SUCCESS"
    error_message: Optional[str] = None
    throughput_metric_name: Optional[str] = None
    throughput_value: Optional[float] = None


class RegressionAlert(BaseModel):
    metric_name: str
    stage_name: Optional[str] = None
    current_value: float
    baseline_value: float
    relative_change: float
    severity: str  # "WARNING", "CRITICAL"
    description: str


class AuditRunRecord(BaseModel):
    run_id: str
    git_commit: str = "unknown"
    git_dirty: bool = False
    timestamp: str = ""
    stream_id: str
    media_duration_sec: float
    total_pipeline_duration_sec: float
    overall_real_time_factor: float
    stages: Dict[str, StageTelemetry] = Field(default_factory=dict)
    quality_metrics: Dict[str, Any] = Field(default_factory=dict)
    regressions_detected: List[RegressionAlert] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    status: str = "SUCCESS"


def get_git_info() -> Tuple[str, bool]:
    """Retrieves current git commit hash and dirty status."""
    commit = "unknown"
    dirty = False
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        status_out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        dirty = bool(status_out)
    except Exception:
        pass
    return commit, dirty


class RegressionComparator:
    """Detects performance, memory, and analytical quality regressions against a baseline."""

    def compare_runs(
        self,
        current: AuditRunRecord,
        baseline: AuditRunRecord,
        speed_threshold: float = 0.25,
        memory_threshold: float = 0.30,
        quality_drop_threshold: float = -0.15,
    ) -> Tuple[List[RegressionAlert], List[str]]:
        """Compares current run with baseline and flags anomalies."""
        alerts: List[RegressionAlert] = []
        recommendations: List[str] = []

        # 1. Total Pipeline Duration
        if baseline.total_pipeline_duration_sec > 0:
            rel_total = (current.total_pipeline_duration_sec - baseline.total_pipeline_duration_sec) / baseline.total_pipeline_duration_sec
            if rel_total > speed_threshold:
                alerts.append(
                    RegressionAlert(
                        metric_name="total_pipeline_duration_sec",
                        current_value=current.total_pipeline_duration_sec,
                        baseline_value=baseline.total_pipeline_duration_sec,
                        relative_change=round(rel_total, 3),
                        severity="CRITICAL" if rel_total > 0.40 else "WARNING",
                        description=f"Total execution slowed by {rel_total:+.1%} compared to baseline ({current.total_pipeline_duration_sec:.2f}s vs {baseline.total_pipeline_duration_sec:.2f}s)",
                    )
                )
                recommendations.append("Investigate stage bottlenecks or GPU compute contention.")

        # 2. Stage-by-Stage Duration & Memory
        for st_name, cur_st in current.stages.items():
            base_st = baseline.stages.get(st_name)
            if not base_st:
                continue

            # Stage duration
            if base_st.duration_sec > 0:
                rel_dur = (cur_st.duration_sec - base_st.duration_sec) / base_st.duration_sec
                if rel_dur > speed_threshold:
                    alerts.append(
                        RegressionAlert(
                            metric_name="stage_duration_sec",
                            stage_name=st_name,
                            current_value=cur_st.duration_sec,
                            baseline_value=base_st.duration_sec,
                            relative_change=round(rel_dur, 3),
                            severity="WARNING",
                            description=f"Stage '{st_name}' slowed by {rel_dur:+.1%} ({cur_st.duration_sec:.2f}s vs {base_st.duration_sec:.2f}s)",
                        )
                    )
                    recommendations.append(f"Profile stage '{st_name}' for batching efficiency or I/O latency.")

            # Stage peak RAM
            if base_st.ram_mb_peak > 0:
                rel_ram = (cur_st.ram_mb_peak - base_st.ram_mb_peak) / base_st.ram_mb_peak
                if rel_ram > memory_threshold:
                    alerts.append(
                        RegressionAlert(
                            metric_name="ram_mb_peak",
                            stage_name=st_name,
                            current_value=cur_st.ram_mb_peak,
                            baseline_value=base_st.ram_mb_peak,
                            relative_change=round(rel_ram, 3),
                            severity="WARNING",
                            description=f"Stage '{st_name}' RAM increased by {rel_ram:+.1%} ({cur_st.ram_mb_peak:.1f}MB vs {base_st.ram_mb_peak:.1f}MB)",
                        )
                    )
                    recommendations.append(f"Check for retained tensor references or buffer leaks in stage '{st_name}'.")

        # 3. Quality Metrics
        for q_name, cur_val in current.quality_metrics.items():
            base_val = baseline.quality_metrics.get(q_name)
            if base_val is not None and isinstance(base_val, (int, float)) and isinstance(cur_val, (int, float)):
                if base_val > 0:
                    rel_q = (cur_val - base_val) / float(base_val)
                    if rel_q < quality_drop_threshold:
                        alerts.append(
                            RegressionAlert(
                                metric_name=q_name,
                                current_value=float(cur_val),
                                baseline_value=float(base_val),
                                relative_change=round(rel_q, 3),
                                severity="WARNING",
                                description=f"Quality metric '{q_name}' dropped by {rel_q:.1%} ({cur_val} vs {base_val})",
                            )
                        )
                        recommendations.append(f"Verify confidence filters or detector heuristics affecting '{q_name}'.")

        return alerts, recommendations


class AuditStore:
    """Manages persistent SQLite audit store for runs and baseline tracking."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self._init_db()

    def _init_db(self):
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS audit_runs (
                run_id TEXT PRIMARY KEY,
                git_commit TEXT NOT NULL,
                git_dirty INT NOT NULL,
                timestamp TEXT NOT NULL,
                stream_id TEXT NOT NULL,
                media_duration_sec REAL NOT NULL,
                total_pipeline_duration_sec REAL NOT NULL,
                overall_rtf REAL NOT NULL,
                stages_json TEXT NOT NULL,
                quality_json TEXT NOT NULL,
                regressions_json TEXT NOT NULL,
                status TEXT NOT NULL,
                is_baseline INT DEFAULT 0
            )
        """)
        self.conn.commit()

    def record_run(self, record: AuditRunRecord, set_as_baseline: bool = False) -> None:
        """Saves a run record to SQLite."""
        cur = self.conn.cursor()
        if set_as_baseline:
            cur.execute("UPDATE audit_runs SET is_baseline = 0 WHERE is_baseline = 1")

        cur.execute("""
            INSERT OR REPLACE INTO audit_runs (
                run_id, git_commit, git_dirty, timestamp, stream_id,
                media_duration_sec, total_pipeline_duration_sec, overall_rtf,
                stages_json, quality_json, regressions_json, status, is_baseline
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.run_id,
            record.git_commit,
            int(record.git_dirty),
            record.timestamp or datetime.now(timezone.utc).isoformat(),
            record.stream_id,
            record.media_duration_sec,
            record.total_pipeline_duration_sec,
            record.overall_real_time_factor,
            json.dumps({k: v.model_dump() for k, v in record.stages.items()}),
            json.dumps(record.quality_metrics),
            json.dumps([r.model_dump() for r in record.regressions_detected]),
            record.status,
            int(set_as_baseline),
        ))
        self.conn.commit()

    def get_run(self, run_id: str) -> Optional[AuditRunRecord]:
        """Loads a run record by ID."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM audit_runs WHERE run_id = ?", (run_id,))
        row = cur.fetchone()
        if not row:
            return None
        return self._row_to_record(row)

    def get_baseline(self) -> Optional[AuditRunRecord]:
        """Retrieves the active baseline run."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM audit_runs WHERE is_baseline = 1 ORDER BY timestamp DESC LIMIT 1")
        row = cur.fetchone()
        if not row:
            return None
        return self._row_to_record(row)

    def get_recent_runs(self, limit: int = 10) -> List[AuditRunRecord]:
        """Returns the most recent N runs."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM audit_runs ORDER BY timestamp DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
        return [self._row_to_record(r) for r in rows]

    def _row_to_record(self, row: Tuple) -> AuditRunRecord:
        stages_data = json.loads(row[8])
        stages = {k: StageTelemetry(**v) for k, v in stages_data.items()}
        quality = json.loads(row[9])
        regressions_data = json.loads(row[10])
        regressions = [RegressionAlert(**r) for r in regressions_data]

        return AuditRunRecord(
            run_id=row[0],
            git_commit=row[1],
            git_dirty=bool(row[2]),
            timestamp=row[3],
            stream_id=row[4],
            media_duration_sec=row[5],
            total_pipeline_duration_sec=row[6],
            overall_real_time_factor=row[7],
            stages=stages,
            quality_metrics=quality,
            regressions_detected=regressions,
            status=row[11],
        )
