"""Standalone child worker process entrypoint (Spec 18).

Executed as a distinct OS subprocess via `python -m stream_fusion.workers.worker_entry --task <task_input.json>`.
Guarantees 100% OS reclamation of CUDA memory upon completion and exit.
"""

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

from stream_fusion.schema.envelope import (
    StreamEventType,
    EnvelopeTelemetry,
    StreamFusionEnvelope,
)
from stream_fusion.monitoring.telemetry import SystemResourceProbe
from stream_fusion.workers.protocol import WorkerTaskInput, WorkerTaskType
from stream_fusion.models.schemas import AudioSegment


def run_worker_task(task_input_path: Path) -> int:
    """Executes the requested GPU/CPU workload in the child process."""
    if not task_input_path.exists():
        sys.stderr.write(f"Task input file not found: {task_input_path}\n")
        return 1

    try:
        with open(task_input_path, "r", encoding="utf-8") as f:
            task_dict = json.load(f)
        task = WorkerTaskInput.model_validate(task_dict)
    except Exception as e:
        sys.stderr.write(f"Failed to parse task input: {e}\n")
        return 1

    start_wall = time.perf_counter()
    peak_ram = 0.0
    peak_vram = 0.0

    out_path = Path(task.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        payload = {}
        if task.task_type == WorkerTaskType.AUDIO_TRANSCRIPTION:
            from stream_fusion.audio.transcriber import AudioTranscriber

            cfg = task.config
            transcriber = AudioTranscriber(
                model_size=cfg.get("whisper_model", "base"),
                device=cfg.get("device"),
                compute_type=cfg.get("compute_type"),
            )
            segments = transcriber.transcribe(
                Path(task.media_path),
                language=cfg.get("language", "en"),
                beam_size=int(cfg.get("beam_size", 5)),
                word_timestamps=bool(cfg.get("word_timestamps", True)),
            )
            payload = {"segments": [s.model_dump() for s in segments]}

        elif task.task_type == WorkerTaskType.VOICE_DIARIZATION:
            from stream_fusion.audio.diarizer import ReactionDiarizer

            cfg = task.config
            raw_segments_data = (task.extra_payload or {}).get("segments", [])
            segments = [AudioSegment.model_validate(s) for s in raw_segments_data]
            diarizer = ReactionDiarizer(
                hf_token=cfg.get("hf_token"),
                device=cfg.get("device", "cuda"),
            )
            diarized_segments = diarizer.diarize_and_tag(segments, Path(task.media_path))
            payload = {"segments": [s.model_dump() for s in diarized_segments]}

        elif task.task_type in (WorkerTaskType.VISION_OCR, WorkerTaskType.DENSE_VISION):
            from stream_fusion.vision.processor import VisionProcessor

            cfg = task.config
            processor = VisionProcessor(
                backend=cfg.get("backend", "fallback"),
                model_id=cfg.get("model_id", "microsoft/Florence-2-base"),
                device=cfg.get("device", "cpu"),
                detect_objects=bool(cfg.get("detect_objects", True)),
            )

            frame_items = (task.extra_payload or {}).get("frame_items", [])
            keyframes = []
            for item in frame_items:
                f_path = Path(item["path"])
                t_sec = float(item["timestamp_sec"])
                f_idx = int(item["frame_index"])
                kf = processor.process_frame(f_path, timestamp_sec=t_sec, frame_index=f_idx)
                keyframes.append(kf)

            payload = {"keyframes": [kf.model_dump() for kf in keyframes]}

        else:
            raise ValueError(f"Unknown task type: {task.task_type}")

        ram, vram = SystemResourceProbe.get_memory_usage_mb()
        peak_ram = max(peak_ram, ram)
        peak_vram = max(peak_vram, vram)

        duration_ms = round((time.perf_counter() - start_wall) * 1000.0, 2)
        telemetry = EnvelopeTelemetry(
            duration_ms=duration_ms,
            stage=f"worker_{task.task_type.value.lower()}",
            ram_mb=peak_ram,
            vram_mb=peak_vram,
        )

        envelope = StreamFusionEnvelope.create(
            stream_id=Path(task.media_path).stem,
            event_type=StreamEventType.STAGE_COMPLETE,
            payload=payload,
            producer=f"stream_fusion.workers.{task.task_type.value.lower()}",
            trace_id=task.trace_id,
            telemetry=telemetry,
        )

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(envelope.to_json(indent=2))

        return 0

    except Exception as e:
        duration_ms = round((time.perf_counter() - start_wall) * 1000.0, 2)
        err_telemetry = EnvelopeTelemetry(
            duration_ms=duration_ms,
            stage=f"worker_{task.task_type.value.lower()}",
        )
        err_envelope = StreamFusionEnvelope.create(
            stream_id=Path(task.media_path).stem,
            event_type=StreamEventType.ERROR,
            payload={"error": str(e), "traceback": traceback.format_exc()},
            producer=f"stream_fusion.workers.{task.task_type.value.lower()}",
            trace_id=task.trace_id,
            telemetry=err_telemetry,
        )
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(err_envelope.to_json(indent=2))

        sys.stderr.write(f"Worker task error: {e}\n{traceback.format_exc()}\n")
        return 1


def main():
    parser = argparse.ArgumentParser(description="StreamFusion Subprocess Worker Entrypoint")
    parser.add_argument("--task", required=True, help="Path to JSON task input file")
    args = parser.parse_args()
    code = run_worker_task(Path(args.task))
    sys.exit(code)


if __name__ == "__main__":
    main()
