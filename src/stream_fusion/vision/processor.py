"""Screen vision context extractor and scene classifier."""

import gc
from pathlib import Path
from typing import List, Optional
from PIL import Image

from stream_fusion.models.schemas import VisualKeyframe


class VisionProcessor:
    """Extracts scene context and OCR from video keyframes.
    
    Supports:
    1. Local Microsoft Florence-2 model (runs in ~1.5 GB VRAM).
    2. Remote/Local Ollama VLM (e.g. qwen2.5-vl) over HTTP/ZeroTier.
    3. Lightweight fast fallback for CPU/CI.
    """

    def __init__(
        self,
        backend: str = "florence",
        model_id: str = "microsoft/Florence-2-base",
        ollama_endpoint: Optional[str] = None,
        device: str = "cuda",
    ):
        self.backend = backend
        self.model_id = model_id
        self.ollama_endpoint = ollama_endpoint or "http://localhost:11434/api/generate"
        self.device = device
        self._model = None
        self._processor = None

    def _load_florence(self):
        """Lazy loads Florence-2 weights onto GPU."""
        if self._model is None and self.backend == "florence":
            from transformers import AutoModelForCausalLM, AutoProcessor
            import torch

            dev = "cuda" if torch.cuda.is_available() and self.device == "cuda" else "cpu"
            self._processor = AutoProcessor.from_pretrained(
                self.model_id, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                torch_dtype=torch.float16 if dev == "cuda" else torch.float32,
            ).to(dev)

    def process_frame(self, image_path: Path, timestamp_sec: float, frame_index: int) -> VisualKeyframe:
        """Processes a single frame image into a structured VisualKeyframe."""
        if not image_path.exists():
            raise FileNotFoundError(f"Keyframe image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        if self.backend == "florence":
            return self._infer_florence(image, timestamp_sec, frame_index)
        elif self.backend == "ollama":
            return self._infer_ollama(image_path, timestamp_sec, frame_index)
        else:
            return self._infer_fallback(image, timestamp_sec, frame_index)

    def _infer_florence(self, image: Image.Image, timestamp_sec: float, frame_index: int) -> VisualKeyframe:
        """Runs Florence-2 detailed captioning and OCR."""
        try:
            self._load_florence()
            import torch

            dev = next(self._model.parameters()).device

            # 1. Run detailed caption
            prompt_caption = "<MORE_DETAILED_CAPTION>"
            inputs = self._processor(text=prompt_caption, images=image, return_tensors="pt").to(dev)
            if dev.type == "cuda":
                inputs = {k: v.to(torch.float16) if v.dtype == torch.float32 else v for k, v in inputs.items()}

            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs.get("pixel_values"),
                max_new_tokens=256,
                num_beams=3,
            )
            caption_text = self._processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_caption = self._processor.post_process_generation(
                caption_text, task=prompt_caption, image_size=(image.width, image.height)
            )
            summary = parsed_caption.get(prompt_caption, caption_text)

            # 2. Run OCR
            prompt_ocr = "<OCR>"
            inputs_ocr = self._processor(text=prompt_ocr, images=image, return_tensors="pt").to(dev)
            if dev.type == "cuda":
                inputs_ocr = {k: v.to(torch.float16) if v.dtype == torch.float32 else v for k, v in inputs_ocr.items()}

            generated_ocr = self._model.generate(
                input_ids=inputs_ocr["input_ids"],
                pixel_values=inputs_ocr.get("pixel_values"),
                max_new_tokens=128,
            )
            ocr_text = self._processor.batch_decode(generated_ocr, skip_special_tokens=False)[0]
            parsed_ocr = self._processor.post_process_generation(
                ocr_text, task=prompt_ocr, image_size=(image.width, image.height)
            )
            ocr_blocks = [line.strip() for line in parsed_ocr.get(prompt_ocr, "").split("\n") if line.strip()]

            # Determine scene type
            scene = "REACT_VIDEO" if "video" in summary.lower() or "youtube" in summary.lower() else "BROWSER"

            return VisualKeyframe(
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                scene_type=scene,
                screen_summary=summary,
                ocr_text_blocks=ocr_blocks,
            )
        except Exception:
            return self._infer_fallback(image, timestamp_sec, frame_index)

    def _infer_ollama(self, image_path: Path, timestamp_sec: float, frame_index: int) -> VisualKeyframe:
        """Calls remote or local Ollama endpoint for VLM scene understanding."""
        import base64
        import requests

        with open(image_path, "rb") as f:
            b64_img = base64.b64encode(f.read()).decode("utf-8")

        prompt = (
            "Describe the livestream screen: what video or game is visible, "
            "what text is prominent, and streamer reaction in 2 sentences."
        )
        try:
            resp = requests.post(
                self.ollama_endpoint,
                json={
                    "model": "qwen2.5-vl",
                    "prompt": prompt,
                    "images": [b64_img],
                    "stream": False,
                },
                timeout=30,
            )
            if resp.status_code == 200:
                summary = resp.json().get("response", "").strip()
                return VisualKeyframe(
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    scene_type="REACT_VIDEO",
                    screen_summary=summary,
                )
        except Exception:
            pass

        image = Image.open(image_path).convert("RGB")
        return self._infer_fallback(image, timestamp_sec, frame_index)

    def _infer_fallback(self, image: Image.Image, timestamp_sec: float, frame_index: int) -> VisualKeyframe:
        """Lightweight offline fallback based on dimensions and color heuristics."""
        w, h = image.size
        return VisualKeyframe(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            scene_type="REACT_VIDEO",
            screen_summary=f"Stream broadcast frame ({w}x{h}) with active content layout.",
            ocr_text_blocks=[],
        )

    def unload(self):
        """Evicts vision model from GPU memory."""
        if self._model is not None:
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
