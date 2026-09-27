"""Automated vertical video (9:16) highlight clipper for YouTube Shorts & TikTok."""

from pathlib import Path
import subprocess
from typing import Dict, List, Optional

from stream_fusion.ingest.demuxer import find_ffmpeg_binary


class VerticalHighlightClipper:
    """Crops 16:9 streams into 9:16 vertical shorts (Facecam top + Content bottom)."""

    def __init__(self, ffmpeg_bin: Optional[str] = None):
        self.ffmpeg_bin = ffmpeg_bin or find_ffmpeg_binary()

    def build_crop_filter(
        self,
        facecam_box: Optional[Dict[str, float]] = None,
        content_box: Optional[Dict[str, float]] = None,
    ) -> str:
        """Constructs an FFmpeg filter_complex graph to stack facecam over content into 1080x1920."""
        # Default layout for standard react streamers (Asmongold: facecam in bottom right or fullscreen)
        # Facecam crop (default: bottom-right corner: x=in_w*0.7, y=in_h*0.6, w=in_w*0.3, h=in_h*0.4)
        # Main content crop (default: center: x=0, y=0, w=in_w*0.75, h=in_h)
        filter_graph = (
            "[0:v]crop=in_w*0.35:in_h*0.45:in_w*0.65:in_h*0.55,scale=1080:768[cam];"
            "[0:v]crop=in_w*0.75:in_h:0:0,scale=1080:1152[content];"
            "[cam][content]vstack=inputs=2[v]"
        )
        return filter_graph

    def build_clip_command(
        self,
        video_path: Path,
        start_sec: float,
        end_sec: float,
        output_path: Path,
        filter_graph: Optional[str] = None,
    ) -> List[str]:
        """Generates ffmpeg CLI argument list to produce the vertical video."""
        graph = filter_graph or self.build_crop_filter()
        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-ss", str(max(0.0, start_sec)),
            "-to", str(end_sec),
            "-i", str(video_path),
            "-filter_complex", graph,
            "-map", "[v]",
            "-map", "0:a?",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "22",
            "-c:a", "aac",
            "-b:a", "128k",
            str(output_path),
        ]
        return cmd

    def export_highlight_short(
        self,
        video_path: Path,
        start_sec: float,
        end_sec: float,
        output_path: Path,
        dry_run: bool = False,
    ) -> Path:
        """Executes clipping of a highlighted segment into a ready-to-upload Short."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = self.build_clip_command(video_path, start_sec, end_sec, output_path)

        if dry_run:
            return output_path

        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg vertical clip export failed (code {res.returncode}): {res.stderr}")

        return output_path
