"""Bounded Rolling Frame Buffering & Temporary Image Purging (Spec 18).

Prevents disk bloat by processing video frames in sliding time windows and immediately
unlinking raw image files once visual features and OCR are extracted.
"""

from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Dict, List, Optional

from stream_fusion.models.schemas import VisualKeyframe
from stream_fusion.ingest.demuxer import MediaDemuxer


class BoundedFrameBuffer:
    """Manages windowed keyframe extraction and immediate disk cleanup."""

    def __init__(
        self,
        demuxer: Optional[MediaDemuxer] = None,
        base_temp_dir: Optional[Path] = None,
    ):
        self.demuxer = demuxer or MediaDemuxer()
        self.base_temp_dir = (
            base_temp_dir or Path(tempfile.gettempdir()) / "stream_fusion_buffer"
        )
        self.base_temp_dir.mkdir(parents=True, exist_ok=True)

    def process_stream_windowed(
        self,
        video_path: Path,
        total_duration_sec: float,
        sample_interval_sec: float = 2.0,
        window_size_sec: float = 30.0,
        frame_processor: Optional[
            Callable[[List[Dict[str, Any]], float, float], List[VisualKeyframe]]
        ] = None,
        purge_on_complete: bool = True,
    ) -> List[VisualKeyframe]:
        """Extracts and processes frames in sliding windows, unlinking images per window."""
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        all_keyframes: List[VisualKeyframe] = []
        current_time = 0.0
        global_frame_idx = 1

        while current_time < total_duration_sec:
            duration = min(window_size_sec, total_duration_sec - current_time)
            window_end = current_time + duration

            # Unique directory for this window
            w_idx = int(current_time // window_size_sec)
            window_dir = self.base_temp_dir / f"win_{w_idx}_{int(current_time)}"
            window_dir.mkdir(parents=True, exist_ok=True)

            try:
                # 1. Demux only frames within current window
                extracted_frames = self.demuxer.extract_frames_at_interval(
                    video_path,
                    output_dir=window_dir,
                    interval_sec=sample_interval_sec,
                    start_time_sec=current_time,
                    duration_sec=duration,
                )

                # 2. Build metadata items
                frame_items = []
                for idx, f_path in enumerate(extracted_frames):
                    t_frame = round(current_time + (idx * sample_interval_sec), 3)
                    frame_items.append(
                        {
                            "path": str(f_path),
                            "timestamp_sec": t_frame,
                            "frame_index": global_frame_idx,
                        }
                    )
                    global_frame_idx += 1

                # 3. Process frames via callback (e.g. isolated worker or local processor)
                if frame_processor:
                    window_kfs = frame_processor(frame_items, current_time, window_end)
                    all_keyframes.extend(window_kfs)

            finally:
                # 4. Immediately purge temporary frame images to maintain bounded disk footprint
                if purge_on_complete and window_dir.exists():
                    try:
                        shutil.rmtree(window_dir, ignore_errors=True)
                    except Exception:
                        pass

            current_time += duration

        return all_keyframes
