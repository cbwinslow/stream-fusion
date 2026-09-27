"""Stateful Checkpoint Resumption Manager (Spec 10).

Manages atomic disk checkpoints for intermediate audio, vision, and chat slices across
long-duration broadcasts to support zero-redundant-compute resumption.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from stream_fusion.models.schemas import (
    AudioSegment,
    ChunkManifest,
    StreamAnalysisResult,
    VisualKeyframe,
)


class CheckpointManager:
    """Coordinates atomic intermediate stage caching and resumption for StreamFusion."""

    def __init__(
        self,
        cache_dir: Path,
        stream_id: str,
        chunk_duration_sec: float = 300.0,
        total_chunks_expected: int = 0,
    ):
        self.stream_id = stream_id
        self.chunk_duration_sec = chunk_duration_sec
        self.total_chunks_expected = total_chunks_expected

        self.root_dir = Path(cache_dir) / stream_id
        self.chunks_dir = self.root_dir / "chunks"
        self.final_dir = self.root_dir / "final"
        self.manifest_path = self.root_dir / "checkpoint_manifest.json"

        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.final_dir.mkdir(parents=True, exist_ok=True)

        self.manifest = self._load_or_create_manifest()

    def _load_or_create_manifest(self) -> ChunkManifest:
        """Loads existing manifest or initializes a new one."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return ChunkManifest(**data)
            except Exception:
                pass

        manifest = ChunkManifest(
            stream_id=self.stream_id,
            chunk_duration_sec=self.chunk_duration_sec,
            total_chunks_expected=self.total_chunks_expected,
            completed_chunks=[],
            status="IN_PROGRESS",
            last_updated=datetime.now(timezone.utc).isoformat(),
            chunk_files={},
        )
        self._write_manifest(manifest)
        return manifest

    def _write_manifest(self, manifest: ChunkManifest) -> None:
        """Writes manifest atomically."""
        manifest.last_updated = datetime.now(timezone.utc).isoformat()
        temp_path = self.manifest_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(manifest.model_dump_json(indent=2))
        temp_path.replace(self.manifest_path)
        self.manifest = manifest

    def is_chunk_completed(self, chunk_idx: int) -> bool:
        """Checks if a given chunk is fully completed."""
        return chunk_idx in self.manifest.completed_chunks

    def get_completed_chunks(self) -> List[int]:
        """Returns sorted list of completed chunk indices."""
        return sorted(self.manifest.completed_chunks)

    def mark_chunk_completed(self, chunk_idx: int) -> None:
        """Records a chunk as finished and persists to manifest."""
        if chunk_idx not in self.manifest.completed_chunks:
            self.manifest.completed_chunks.append(chunk_idx)
            self.manifest.completed_chunks.sort()
            if (
                self.manifest.total_chunks_expected > 0
                and len(self.manifest.completed_chunks) >= self.manifest.total_chunks_expected
            ):
                self.manifest.status = "COMPLETED"
            self._write_manifest(self.manifest)

    # -------------------------------------------------------------
    # Audio Persistence
    # -------------------------------------------------------------
    def save_chunk_audio(self, chunk_idx: int, audio_segments: List[AudioSegment]) -> Path:
        """Persists intermediate transcribed/diarized audio segments."""
        out_file = self.chunks_dir / f"chunk_{chunk_idx:04d}_audio.json"
        data = [seg.model_dump() for seg in audio_segments]
        temp_file = out_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        temp_file.replace(out_file)
        return out_file

    def load_chunk_audio(self, chunk_idx: int) -> Optional[List[AudioSegment]]:
        """Loads cached audio segments if present."""
        path = self.chunks_dir / f"chunk_{chunk_idx:04d}_audio.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [AudioSegment(**item) for item in data]

    # -------------------------------------------------------------
    # Vision Persistence
    # -------------------------------------------------------------
    def save_chunk_vision(self, chunk_idx: int, keyframes: List[VisualKeyframe]) -> Path:
        """Persists intermediate visual keyframes and OCR."""
        out_file = self.chunks_dir / f"chunk_{chunk_idx:04d}_vision.json"
        data = [kf.model_dump() for kf in keyframes]
        temp_file = out_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        temp_file.replace(out_file)
        return out_file

    def load_chunk_vision(self, chunk_idx: int) -> Optional[List[VisualKeyframe]]:
        """Loads cached visual keyframes if present."""
        path = self.chunks_dir / f"chunk_{chunk_idx:04d}_vision.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [VisualKeyframe(**item) for item in data]

    # -------------------------------------------------------------
    # Chat Persistence
    # -------------------------------------------------------------
    def save_chunk_chat(self, chunk_idx: int, chat_buckets: List[Dict[str, Any]]) -> Path:
        """Persists intermediate chat analysis buckets."""
        out_file = self.chunks_dir / f"chunk_{chunk_idx:04d}_chat.json"
        temp_file = out_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(chat_buckets, f, indent=2)
        temp_file.replace(out_file)
        return out_file

    def load_chunk_chat(self, chunk_idx: int) -> Optional[List[Dict[str, Any]]]:
        """Loads cached chat buckets if present."""
        path = self.chunks_dir / f"chunk_{chunk_idx:04d}_chat.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # -------------------------------------------------------------
    # Final Result Persistence
    # -------------------------------------------------------------
    def save_final_result(self, result: StreamAnalysisResult) -> Path:
        """Saves final consolidated StreamAnalysisResult to final dir."""
        out_file = self.final_dir / "stream_fusion_result.json"
        temp_file = out_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))
        temp_file.replace(out_file)
        return out_file

    def load_final_result(self) -> Optional[StreamAnalysisResult]:
        """Loads consolidated result if present."""
        path = self.final_dir / "stream_fusion_result.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return StreamAnalysisResult.model_validate_json(f.read())
