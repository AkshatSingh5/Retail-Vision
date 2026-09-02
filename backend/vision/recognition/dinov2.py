from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from backend.app.config import DINO_DEVICE, DINOV2_MODEL, EMBEDDING_DIM
from vision.device import gpu_name, resolve_device

if TYPE_CHECKING:
    from PIL import Image as PILImage

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_embedder: "DinoV2Embedder | None" = None


class DinoV2Embedder:
    """Lazy DINOv2 visual embedder. Same model + preprocess for register and scan."""

    def __init__(self, model_id: str = DINOV2_MODEL) -> None:
        from transformers import AutoModel, AutoProcessor

        self.model_id = model_id
        self.device = resolve_device(DINO_DEVICE)
        gpu = gpu_name()
        print(f"Using device: {self.device}")
        if gpu:
            print(f"GPU: {gpu}")
        print(f"[DINOv2] Loading {model_id} on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.eval()
        self.model.to(self.device)
        self.dim = int(getattr(self.model.config, "hidden_size", EMBEDDING_DIM))
        print("DINOv2 loaded successfully")
        print(f"[DINOv2] Ready (dim={self.dim} device={self.device})")

    @torch.inference_mode()
    def embed_bgr(self, image: np.ndarray) -> list[float]:
        if image is None or getattr(image, "size", 0) == 0:
            raise ValueError("Empty product image.")
        frame = image
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        from PIL import Image

        pil: PILImage.Image = Image.fromarray(rgb)
        inputs = self.processor(images=pil, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = self.model(**inputs)
        # CLS token — standard DINOv2 image representation.
        vector = outputs.last_hidden_state[:, 0]
        before = float(torch.linalg.vector_norm(vector).item())
        vector = F.normalize(vector, p=2, dim=-1)
        after = float(torch.linalg.vector_norm(vector).item())
        values = vector.squeeze(0).detach().float().cpu().tolist()
        print(
            f"[DINOv2] model={self.model_id} input={frame.shape} "
            f"out_dim={len(values)} norm_before={before:.4f} norm_after={after:.4f}"
        )
        return [float(item) for item in values]


def get_dinov2() -> DinoV2Embedder:
    global _embedder
    with _lock:
        if _embedder is None:
            _embedder = DinoV2Embedder()
        return _embedder


def dinov2_embed(image: np.ndarray) -> list[float]:
    return get_dinov2().embed_bgr(image)


def reset_dinov2() -> None:
    global _embedder
    with _lock:
        _embedder = None
