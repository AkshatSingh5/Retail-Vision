"""Phase 8 robustness conditions.

These transforms probe held-out photos under named retail conditions.
They are not a substitute for 1000 unique real captures per SKU.
"""

from __future__ import annotations

import cv2
import numpy as np


def _rotate(image: np.ndarray, angle: float) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    return cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT_101)


def _scale(image: np.ndarray, scale: float) -> np.ndarray:
    height, width = image.shape[:2]
    new_w, new_h = max(1, int(width * scale)), max(1, int(height * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros_like(image)
    if scale <= 1.0:
        x0 = (width - new_w) // 2
        y0 = (height - new_h) // 2
        canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
        return canvas
    x0 = max(0, (new_w - width) // 2)
    y0 = max(0, (new_h - height) // 2)
    return resized[y0 : y0 + height, x0 : x0 + width]


def _brightness(image: np.ndarray, delta: int) -> np.ndarray:
    return cv2.convertScaleAbs(image, alpha=1.0, beta=delta)


def _shadow(image: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    height = overlay.shape[0]
    overlay[height // 2 :, :] = (overlay[height // 2 :, :].astype(np.int16) * 0.35).clip(0, 255).astype(np.uint8)
    return overlay


def _reflection(image: np.ndarray) -> np.ndarray:
    glare = np.zeros_like(image)
    height, width = image.shape[:2]
    cv2.ellipse(glare, (width // 2, height // 3), (width // 5, height // 8), 20, 0, 360, (220, 220, 220), -1)
    return cv2.addWeighted(image, 0.78, glare, 0.22, 0)


def _occlude(image: np.ndarray) -> np.ndarray:
    out = image.copy()
    height, width = out.shape[:2]
    x1, y1 = int(width * 0.35), int(height * 0.15)
    x2, y2 = int(width * 0.85), int(height * 0.55)
    cv2.rectangle(out, (x1, y1), (x2, y2), (20, 20, 20), -1)
    return out


def _clutter(image: np.ndarray) -> np.ndarray:
    noise = np.random.default_rng(7).integers(0, 80, size=image.shape, dtype=np.uint8)
    return cv2.addWeighted(image, 0.72, noise, 0.28, 0)


def _tilt(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    src = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    dst = np.float32([[width * 0.08, height * 0.12], [width * 0.92, 0], [0, height * 0.95], [width, height * 0.88]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT_101)


CONDITIONS: list[tuple[str, str, callable]] = [
    ("front_view", "Angle", lambda image: image.copy()),
    ("back_view_proxy_flip", "Angle", lambda image: cv2.flip(image, 1)),
    ("left_view_proxy", "Angle", lambda image: _rotate(image, 18)),
    ("right_view_proxy", "Angle", lambda image: _rotate(image, -18)),
    ("rotation_15", "Rotation", lambda image: _rotate(image, 15)),
    ("rotation_30", "Rotation", lambda image: _rotate(image, 30)),
    ("rotation_45", "Rotation", lambda image: _rotate(image, 45)),
    ("rotation_90", "Rotation", lambda image: _rotate(image, 90)),
    ("upside_down", "Orientation", lambda image: _rotate(image, 180)),
    ("tilted", "Orientation", lambda image: _tilt(image)),
    ("close_distance", "Distance", lambda image: _scale(image, 1.35)),
    ("far_distance", "Distance", lambda image: _scale(image, 0.55)),
    ("scale_small", "Scale", lambda image: _scale(image, 0.7)),
    ("lighting", "Lighting", lambda image: _brightness(image, 45)),
    ("low_light", "Lighting", lambda image: _brightness(image, -75)),
    ("shadows", "Lighting", _shadow),
    ("reflections", "Lighting", _reflection),
    ("partial_occlusion", "Occlusion", _occlude),
    ("cluttered_background", "Background", _clutter),
]
