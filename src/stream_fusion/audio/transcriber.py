"""Speech-to-text transcription engine wrapping faster-whisper."""

import gc
import os
from pathlib import Path
import sys
from typing import List, Optional

from stream_fusion.models.schemas import AudioSegment, WordTiming


def setup_cuda_dll_paths():
    """Ensures CUDA DLLs (cublas, cudnn) shipped with PyTorch are found by CTranslate2 on Windows."""
    if sys.platform == "win32":
        try:
            import torch
            torch_lib = Path(torch.__file__).parent / "lib"
            if torch_lib.exists():
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(str(torch_lib))
                os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")
        except Exception:
            pass


class AudioTranscriber:
    """Performs fast speech-to-text using CTranslate2 faster-whisper."""

    def __init__(
        self,
        model_size: str = "base",
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
    ):
        setup_cuda_dll_paths()
        self.model_size = model_size

        # Auto-configure device & compute type
        if device is None:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        if compute_type is None:
            self.compute_type = "float16" if self.device == "cuda" else "int8"
        else:
            self.compute_type = compute_type

        self._model = None

    def _load_model(self):
        """Lazy load model to conserve VRAM until needed."""
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )

    def transcribe(
        self,
        audio_path: Path,
        language: str = "en",
        beam_size: int = 5,
        word_timestamps: bool = True,
    ) -> List[AudioSegment]:
        """Transcribes audio file to typed AudioSegment models with word timings."""
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self._load_model()
        try:
            segments_raw, info = self._model.transcribe(
                str(audio_path),
                language=language,
                beam_size=beam_size,
                word_timestamps=word_timestamps,
            )
            raw_list = list(segments_raw)
        except (IndexError, RuntimeError):
            # Fallback without word-level timestamp alignment on silent/music segments
            segments_raw, info = self._model.transcribe(
                str(audio_path),
                language=language,
                beam_size=beam_size,
                word_timestamps=False,
            )
            raw_list = list(segments_raw)

        segments: List[AudioSegment] = []
        for idx, seg in enumerate(raw_list):
            words = []
            if seg.words:
                for w in seg.words:
                    words.append(
                        WordTiming(
                            word=w.word.strip(),
                            start=round(w.start, 3),
                            end=round(w.end, 3),
                            probability=round(w.probability, 3),
                        )
                    )

            segments.append(
                AudioSegment(
                    segment_id=idx + 1,
                    start_sec=round(seg.start, 3),
                    end_sec=round(seg.end, 3),
                    speaker_label="UNKNOWN",
                    transcript=seg.text.strip(),
                    confidence=round(info.language_probability, 3) if info else 1.0,
                    words=words if words else None,
                )
            )

        return segments

    def unload(self):
        """Explicitly evict model and free GPU VRAM."""
        if self._model is not None:
            del self._model
            self._model = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
