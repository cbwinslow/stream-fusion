"""Editor Agent for Autonomous Short Production (Spec 21).

Calculates dynamic crop boxes, subtitle styling, and audio ducking.
"""

from typing import Dict, List, Optional

from stream_fusion.models.schemas import (
    CropLayout,
    EditorialCutPlan,
    ShortCandidate,
    SubtitleStylePreset,
    VisualKeyframe,
)


class EditorAgent:
    """Intelligent editor mapping spatial cropping, karaoke subtitle styling, and pacing."""

    def __init__(
        self,
        default_layout: CropLayout = CropLayout.STACKED_CAM_CONTENT,
        default_subtitle_preset: SubtitleStylePreset = SubtitleStylePreset.KARAOKE_POP,
    ):
        self.default_layout = default_layout
        self.default_subtitle_preset = default_subtitle_preset

    def create_cut_plan(
        self,
        candidate: ShortCandidate,
        keyframes: Optional[List[VisualKeyframe]] = None,
        layout_override: Optional[CropLayout] = None,
        subtitle_preset_override: Optional[SubtitleStylePreset] = None,
        burn_subtitles: bool = True,
    ) -> EditorialCutPlan:
        """Determines optimal crop boxes and audio/visual styling for a short candidate."""
        layout = layout_override or self.default_layout
        preset = subtitle_preset_override or self.default_subtitle_preset

        # Find keyframes inside this candidate's window
        window_frames: List[VisualKeyframe] = []
        if keyframes:
            window_frames = [
                kf for kf in keyframes
                if candidate.start_sec <= kf.timestamp_sec <= candidate.end_sec
            ]

        facecam_box, content_box = self._detect_layout_boxes(window_frames, layout)
        color = self._select_highlight_color(candidate.primary_emotion)
        ducking_db = -14.0 if candidate.chat_burst_zscore >= 3.0 else -10.0

        return EditorialCutPlan(
            crop_layout=layout,
            facecam_box=facecam_box,
            content_box=content_box,
            subtitle_preset=preset,
            highlight_word_color=color,
            burn_subtitles=burn_subtitles,
            audio_duck_music_db=ducking_db,
            hook_duration_sec=3.0,
        )

    def _detect_layout_boxes(
        self,
        keyframes: List[VisualKeyframe],
        layout: CropLayout,
    ) -> tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
        """Infers facecam and content bounding boxes from visual keyframe metadata."""
        if layout == CropLayout.FULL_CONTENT_PAN_SCAN:
            return None, {"x": 0.15, "y": 0.0, "w": 0.70, "h": 1.0}

        # Inspect keyframes to detect if facecam exists in typical corners
        # Default layout: facecam top 40% (1080x768), gameplay/content bottom 60% (1080x1152)
        default_cam = {"x": 0.65, "y": 0.55, "w": 0.35, "h": 0.45}
        default_content = {"x": 0.0, "y": 0.0, "w": 0.75, "h": 1.0}

        if not keyframes:
            return default_cam, default_content

        # Scan for detected streamer face / bounding boxes if present
        for kf in keyframes:
            if kf.scene_type == "FULLSCREEN_CAM":
                # Fullscreen streamer - center pan & scan
                return {"x": 0.20, "y": 0.10, "w": 0.60, "h": 0.80}, default_content

            for obj in kf.detected_objects:
                if "face" in obj.label.lower() or "streamer" in obj.label.lower() or "person" in obj.label.lower():
                    # box: [ymin, xmin, ymax, xmax]
                    ymin, xmin, ymax, xmax = obj.box
                    w = max(0.2, xmax - xmin)
                    h = max(0.2, ymax - ymin)
                    return {"x": xmin, "y": ymin, "w": w, "h": h}, default_content

        return default_cam, default_content

    def _select_highlight_color(self, emotion: str) -> str:
        """Chooses ASS subtitle karaoke highlight color based on emotional tone."""
        # ASS color code format: &HAABBGGRR (or &H00BBGGRR)
        emotion_upper = emotion.upper()
        if "LAUGHTER" in emotion_upper or "HYSTERICAL" in emotion_upper:
            return "&H0000FFFF"  # Electric Yellow
        elif "HYPE" in emotion_upper or "EXCITEMENT" in emotion_upper:
            return "&H0000FF00"  # Vibrant Neon Green
        elif "CONTROVERSY" in emotion_upper or "HOT_TAKE" in emotion_upper:
            return "&H000080FF"  # High-Alert Orange
        elif "SHOCK" in emotion_upper or "SCREAM" in emotion_upper:
            return "&H000000FF"  # Bold Red
        else:
            return "&H00FFFF00"  # Cyan / Bright Aqua
