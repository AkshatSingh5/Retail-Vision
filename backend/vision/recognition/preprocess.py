"""Shared image preprocessing for product registration and scanning.

Both pipelines must use the same BGR → prepare path so embeddings stay comparable.
"""

from __future__ import annotations

import numpy as np

EMBED_SIZE = 96


def prepare_product_image(image: np.ndarray, size: int = EMBED_SIZE) -> np.ndarray:
    """Normalize a product crop/frame for embedding (OpenCV BGR in/out)."""
    if image is None or getattr(image, "size", 0) == 0:
        raise ValueError("Empty product image.")
    from vision.image_io import letterbox_bgr

    return letterbox_bgr(image, size)


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
