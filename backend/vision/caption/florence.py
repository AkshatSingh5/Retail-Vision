"""Florence-2 image-to-text: upload an image, return a detailed prompt.

Same generation path as the Hugging Face Space:
task=<MORE_DETAILED_CAPTION>, num_beams=3, max_new_tokens=1024.
"""

from __future__ import annotations

import io
import logging
import threading

import numpy as np
import torch
from PIL import Image

from backend.app.config import FLORENCE_MODEL
from vision.device import gpu_name, resolve_device

logger = logging.getLogger(__name__)

TASK = "<MORE_DETAILED_CAPTION>"

_lock = threading.Lock()
_captioner: "FlorenceCaptioner | None" = None


def _to_pil(image: Image.Image | np.ndarray | bytes) -> Image.Image:
    if isinstance(image, Image.Image):
        pil = image
    elif isinstance(image, (bytes, bytearray, memoryview)):
        pil = Image.open(io.BytesIO(bytes(image)))
    else:
        pil = Image.fromarray(image)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    return pil


def _move_to_device(inputs: dict, device: str) -> dict:
    moved = {}
    for key, value in inputs.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


class FlorenceCaptioner:
    """Lazy Florence-2 captioner. Loaded on first use."""

    def __init__(self, model_id: str = FLORENCE_MODEL) -> None:
        self.device = resolve_device("cuda")
        self.model_id = model_id
        gpu = gpu_name() or "n/a"
        print(f"[Florence-2] Loading {model_id} on {self.device}...")
        print(f"Florence device: {self.device}")
        if str(self.device).startswith("cuda"):
            print(f"  GPU: {gpu}")
        self.model, self.processor, self.model_id = self._load(model_id)
        self.model.to(self.device)
        self.model.eval()
        print(f"[Florence-2] Ready ({self.model_id})")

    def _load(self, model_id: str):
        last_error: Exception | None = None
        candidates = [model_id]
        if model_id != "florence-community/Florence-2-base":
            candidates.append("florence-community/Florence-2-base")

        for candidate in candidates:
            try:
                return self._load_remote(candidate)
            except Exception as extra:
                last_error = extra
                print(f"[Florence-2] Remote-code load failed for {candidate}: {extra}")
            try:
                return self._load_native(candidate)
            except Exception as extra:
                last_error = extra
                print(f"[Florence-2] Native load failed for {candidate}: {extra}")

        raise RuntimeError(
            f"Could not load Florence-2 ({model_id}). Last error: {last_error}"
        ) from last_error

    def _from_pretrained(self, cls, model_id: str):
        kwargs = {"trust_remote_code": True}
        dtype = torch.float32 if self.device == "cpu" else torch.float16
        try:
            return cls.from_pretrained(model_id, dtype=dtype, **kwargs)
        except TypeError:
            return cls.from_pretrained(model_id, torch_dtype=dtype, **kwargs)

    def _load_native(self, model_id: str):
        from transformers import AutoProcessor, Florence2ForConditionalGeneration

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = self._from_pretrained(Florence2ForConditionalGeneration, model_id)
        return model, processor, model_id

    def _load_remote(self, model_id: str):
        from transformers import AutoModelForCausalLM, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = self._from_pretrained(AutoModelForCausalLM, model_id)
        return model, processor, model_id

    @torch.inference_mode()
    def generate_caption(self, image: Image.Image | np.ndarray | bytes) -> str:
        pil = _to_pil(image)
        inputs = self.processor(text=TASK, images=pil, return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)
        else:
            inputs = _move_to_device(dict(inputs), self.device)

        payload = dict(inputs)
        gen_inputs = {
            key: payload[key]
            for key in ("input_ids", "pixel_values", "attention_mask")
            if payload.get(key) is not None
        }
        generated_ids = self.model.generate(
            **gen_inputs,
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=TASK,
            image_size=(pil.width, pil.height),
        )
        if isinstance(parsed_answer, dict):
            prompt = parsed_answer.get(TASK) or next(iter(parsed_answer.values()), "")
        else:
            prompt = str(parsed_answer)
        prompt = str(prompt).strip()
        print(f"\n\nGeneration completed!:{prompt}")
        return prompt


def get_florence() -> FlorenceCaptioner:
    global _captioner
    with _lock:
        if _captioner is None:
            _captioner = FlorenceCaptioner()
        return _captioner


def generate_caption(image: Image.Image | np.ndarray | bytes) -> str:
    return get_florence().generate_caption(image)


def reset_florence() -> None:
    global _captioner
    with _lock:
        _captioner = None
