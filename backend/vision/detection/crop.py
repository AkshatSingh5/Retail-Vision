from __future__ import annotations

import numpy as np


def crop_box(frame: np.ndarray, bbox: list[float]) -> np.ndarray | None:
    """Return a copy of `frame` cropped to `bbox`, or None if the box is too small."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(value)) for value in bbox)
    x1, x2 = sorted((max(0, x1), min(width, x2)))
    y1, y2 = sorted((max(0, y1), min(height, y2)))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return frame[y1:y2, x1:x2].copy()
