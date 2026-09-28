"""Worker Isolation Manager for Subprocess GPU Task Execution (Spec 18).

Launches heavy GPU tasks in isolated child processes to guarantee zero VRAM leakage.
"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional
import uuid

from stream_fusion.models.schemas import AudioSegment, VisualKeyframe
from stream_fusion.schema.envelope import (
    StreamEventType,
    EnvelopeTelemetry,
    StreamFusionEnvelope,
)
from stream_fusion.workers.protocol import WorkerTaskInput, WorkerTaskType


class WorkerIsolationManager:
    """Manages subprocess lifecycles for zero-leakage GPU inference execution."""

    def __init__(self, temp_dir: Optional[Path] = None):
        self.temp_dir = temp_dir or Path(tempfile.gettempdir()) / "stream_fusion_workers"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def run_task(self, task: WorkerTaskInput) -> StreamFusionEnvelope:
        """Executes a worker task in an isolated subprocess with timeout supervision."""
        job_id = str(uuid.uuid4())[:8]
        task_in_file = self.temp_dir / f"task_{job_id}_in.json"
        task_out_file = self.temp_dir / f"task_{job_id}_out.json"

        # Ensure task points to this specific output file if not set
        if not task.output_path or task.output_path == "":
            task.output_path = str(task_out_file)
        else:
            task_out_file = Path(task.output_path)

        with open(task_in_file, "w", encoding="utf-8") as f:
            f.write(task.model_dump_json(indent=2))

        cmd = [
            sys.executable,
            "-m",
            "stream_fusion.workers.worker_entry",
            "--task",
            str(task_in_file),
        ]

        start_time = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=task.timeout_sec,
                check=False,
            )

            if task_out_file.exists():
                with open(task_out_file, "r", encoding="utf-8") as f:
                    content = f.read()
                envelope = StreamFusionEnvelope.from_json(content)
            else:
                duration_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
                envelope = StreamFusionEnvelope.create(
                    stream_id=Path(task.media_path).stem,
                    event_type=StreamEventType.ERROR,
                    payload={
                        "error": f"Subprocess exited with code {proc.returncode} but produced no output file.",
                        "stderr": proc.stderr,
                        "stdout": proc.stdout,
                    },
                    telemetry=EnvelopeTelemetry(
                        duration_ms=duration_ms,
                        stage=f"worker_{task.task_type.value.lower()}",
                    ),
                )

        except subprocess.TimeoutExpired:
            duration_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            envelope = StreamFusionEnvelope.create(
                stream_id=Path(task.media_path).stem,
                event_type=StreamEventType.ERROR,
                payload={
                    "error": f"Worker task timed out after {task.timeout_sec} seconds."
                },
                telemetry=EnvelopeTelemetry(
                    duration_ms=duration_ms,
                    stage=f"worker_{task.task_type.value.lower()}",
                ),
            )
        except Exception as e:
            duration_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            envelope = StreamFusionEnvelope.create(
                stream_id=Path(task.media_path).stem,
                event_type=StreamEventType.ERROR,
                payload={"error": str(e)},
                telemetry=EnvelopeTelemetry(
                    duration_ms=duration_ms,
                    stage=f"worker_{task.task_type.value.lower()}",
                ),
            )
        finally:
            # Clean up temporary task files
            if task_in_file.exists():
                try:
                    task_in_file.unlink()
                except Exception:
                    pass
            if task_out_file.exists():
                try:
                    task_out_file.unlink()
                except Exception:
                    pass

        return envelope

    def transcribe_isolated(
        self,
        wav_path: Path,
        config: Optional[Dict[str, Any]] = None,
        timeout_sec: float = 600.0,
    ) -> List[AudioSegment]:
        """Runs Whisper transcription in an isolated child process."""
        task = WorkerTaskInput(
            task_type=WorkerTaskType.AUDIO_TRANSCRIPTION,
            media_path=str(wav_path),
            output_path="",
            config=config or {},
            timeout_sec=timeout_sec,
        )
        envelope = self.run_task(task)
        if envelope.event_type == StreamEventType.ERROR:
            raise RuntimeError(f"Isolated transcription failed: {envelope.payload}")

        raw_segments = (envelope.payload or {}).get("segments", [])
        return [AudioSegment.model_validate(s) for s in raw_segments]

    def diarize_isolated(
        self,
        wav_path: Path,
        segments: List[AudioSegment],
        config: Optional[Dict[str, Any]] = None,
        timeout_sec: float = 600.0,
    ) -> List[AudioSegment]:
        """Runs voice diarization in an isolated child process."""
        task = WorkerTaskInput(
            task_type=WorkerTaskType.VOICE_DIARIZATION,
            media_path=str(wav_path),
            output_path="",
            config=config or {},
            timeout_sec=timeout_sec,
            extra_payload={"segments": [s.model_dump() for s in segments]},
        )
        envelope = self.run_task(task)
        if envelope.event_type == StreamEventType.ERROR:
            raise RuntimeError(f"Isolated diarization failed: {envelope.payload}")

        raw_segments = (envelope.payload or {}).get("segments", [])
        return [AudioSegment.model_validate(s) for s in raw_segments]

    def process_keyframes_isolated(
        self,
        frame_items: List[Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
        timeout_sec: float = 600.0,
    ) -> List[VisualKeyframe]:
        """Runs vision OCR and scene captioning in an isolated child process."""
        task = WorkerTaskInput(
            task_type=WorkerTaskType.VISION_OCR,
            media_path=str(frame_items[0]["path"]) if frame_items else "",
            output_path="",
            config=config or {},
            timeout_sec=timeout_sec,
            extra_payload={"frame_items": frame_items},
        )
        envelope = self.run_task(task)
        if envelope.event_type == StreamEventType.ERROR:
            raise RuntimeError(f"Isolated vision OCR failed: {envelope.payload}")

        raw_keyframes = (envelope.payload or {}).get("keyframes", [])
        return [VisualKeyframe.model_validate(k) for k in raw_keyframes]
