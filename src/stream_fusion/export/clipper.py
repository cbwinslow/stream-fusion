"""Automated vertical video (9:16) highlight clipper for YouTube Shorts & TikTok."""

from pathlib import Path
import subprocess
from typing import Dict, List, Optional

from stream_fusion.ingest.demuxer import find_ffmpeg_binary
from stream_fusion.models.schemas import WordTiming
from stream_fusion.export.subtitles import generate_karaoke_ass


class VerticalHighlightClipper:
    """Crops 16:9 streams into 9:16 vertical shorts (Facecam top + Content bottom)."""

    def __init__(self, ffmpeg_bin: Optional[str] = None):
        self.ffmpeg_bin = ffmpeg_bin or find_ffmpeg_binary()

    def build_crop_filter(
        self,
        facecam_box: Optional[Dict[str, float]] = None,
        content_box: Optional[Dict[str, float]] = None,
        ass_subtitle_path: Optional[Path] = None,
    ) -> str:
        """Constructs an FFmpeg filter_complex graph to stack facecam over content into 1080x1920."""
        # Facecam crop
        if facecam_box:
            w = facecam_box.get("w", 0.35)
            h = facecam_box.get("h", 0.45)
            x = facecam_box.get("x", 0.65)
            y = facecam_box.get("y", 0.55)
            cam_crop = f"[0:v]crop=in_w*{w:.3f}:in_h*{h:.3f}:in_w*{x:.3f}:in_h*{y:.3f},scale=1080:768[cam];"
        else:
            cam_crop = "[0:v]crop=in_w*0.35:in_h*0.45:in_w*0.65:in_h*0.55,scale=1080:768[cam];"

        # Content crop
        if content_box:
            w = content_box.get("w", 0.75)
            h = content_box.get("h", 1.0)
            x = content_box.get("x", 0.0)
            y = content_box.get("y", 0.0)
            content_crop = f"[0:v]crop=in_w*{w:.3f}:in_h*{h:.3f}:in_w*{x:.3f}:in_h*{y:.3f},scale=1080:1152[content];"
        else:
            content_crop = "[0:v]crop=in_w*0.75:in_h:0:0,scale=1080:1152[content];"

        if ass_subtitle_path:
            # Escape path for FFmpeg filter_complex on Windows / POSIX
            escaped_path = str(ass_subtitle_path.resolve()).replace("\\", "/").replace(":", "\\:")
            filter_graph = (
                f"{cam_crop}{content_crop}[cam][content]vstack=inputs=2[stacked];"
                f"[stacked]subtitles=filename='{escaped_path}'[v]"
            )
        else:
            filter_graph = f"{cam_crop}{content_crop}[cam][content]vstack=inputs=2[v]"

        return filter_graph

    def build_clip_command(
        self,
        video_path: Path,
        start_sec: float,
        end_sec: float,
        output_path: Path,
        filter_graph: Optional[str] = None,
        facecam_box: Optional[Dict[str, float]] = None,
        content_box: Optional[Dict[str, float]] = None,
        ass_subtitle_path: Optional[Path] = None,
    ) -> List[str]:
        """Generates ffmpeg CLI argument list to produce the vertical video."""
        graph = filter_graph or self.build_crop_filter(
            facecam_box=facecam_box,
            content_box=content_box,
            ass_subtitle_path=ass_subtitle_path,
        )
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
        words: Optional[List[WordTiming]] = None,
        facecam_box: Optional[Dict[str, float]] = None,
        content_box: Optional[Dict[str, float]] = None,
        burn_subtitles: bool = True,
        dry_run: bool = False,
    ) -> Path:
        """Executes clipping of a highlighted segment into a ready-to-upload Short."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ass_path = None
        if words and burn_subtitles:
            ass_path = output_path.with_suffix(".ass")
            generate_karaoke_ass(words, ass_path, base_offset_sec=start_sec)

        cmd = self.build_clip_command(
            video_path=video_path,
            start_sec=start_sec,
            end_sec=end_sec,
            output_path=output_path,
            facecam_box=facecam_box,
            content_box=content_box,
            ass_subtitle_path=ass_path,
        )

        if dry_run:
            return output_path

        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg vertical clip export failed (code {res.returncode}): {res.stderr}")

        return output_path
