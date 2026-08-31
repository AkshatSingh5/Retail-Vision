"""Shared image preprocessing for product registration and scanning.

Both pipelines must use the same BGR → prepare path so embeddings stay comparable.
"""

from __future__ import annotations

import cv2
import numpy as np

EMBED_SIZE = 96


def prepare_product_image(image: np.ndarray, size: int = EMBED_SIZE) -> np.ndarray:
    """Normalize a product crop/frame for embedding (OpenCV BGR in/out)."""
    if image is None or getattr(image, "size", 0) == 0:
        raise ValueError("Empty product image.")
    frame = image
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif frame.shape[2] != 3:
        raise ValueError("Unsupported product image channels.")

    height, width = frame.shape[:2]
    if height < 8 or width < 8:
        raise ValueError("Product crop is too small.")

    # Letterbox to square so aspect ratio does not distort pack layout.
    scale = size / max(height, width)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y0 = (size - new_h) // 2
    x0 = (size - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def pad_bbox(
    bbox: list[float],
    frame_shape: tuple[int, ...],
    pad_ratio: float = 0.08,
) -> list[float]:
    """Expand a detection box slightly so pack edges are not clipped."""
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    px = bw * pad_ratio
    py = bh * pad_ratio
    return [
        max(0.0, x1 - px),
        max(0.0, y1 - py),
        min(float(width), x2 + px),
        min(float(height), y2 + py),
    ]
