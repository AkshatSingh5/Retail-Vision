"""OpenCV image-quality gates before visual recognition."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from backend.app.config import (
    BLUR_VARIANCE_MIN,
    MAX_BRIGHTNESS,
    MIN_BRIGHTNESS,
    MIN_CONTRAST_STD,
    MIN_IMAGE_SIDE,
)


@dataclass(frozen=True)
class QualityResult:
    ok: bool
    reason: str = ""
    message: str = ""
    blur_variance: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    width: int = 0
    height: int = 0


def assess_image_quality(image: np.ndarray) -> QualityResult:
    """Reject frames that are too small, too blurry, or poorly lit."""
    if image is None or getattr(image, "size", 0) == 0:
        return QualityResult(
            ok=False,
            reason="invalid_image",
            message="Invalid image.",
        )
    height, width = int(image.shape[0]), int(image.shape[1])
    if min(height, width) < MIN_IMAGE_SIDE:
        return QualityResult(
            ok=False,
            reason="too_small",
            message="IMAGE QUALITY TOO LOW\nPlease hold the product closer and scan again.",
            width=width,
            height=height,
        )

    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    if brightness < MIN_BRIGHTNESS:
        return QualityResult(
            ok=False,
            reason="too_dark",
            message="IMAGE QUALITY TOO LOW\nPlease improve lighting and scan again.",
            blur_variance=blur,
            brightness=brightness,
            contrast=contrast,
            width=width,
            height=height,
        )
    if brightness > MAX_BRIGHTNESS and contrast < MIN_CONTRAST_STD:
        return QualityResult(
            ok=False,
            reason="too_bright",
            message="IMAGE QUALITY TOO LOW\nPlease reduce glare and scan again.",
            blur_variance=blur,
            brightness=brightness,
            contrast=contrast,
            width=width,
            height=height,
        )
    # Flat + blurry → reject. Structured packs with edges can pass a modest blur floor.
    if blur < BLUR_VARIANCE_MIN and contrast < max(MIN_CONTRAST_STD, 12.0):
        return QualityResult(
            ok=False,
            reason="too_blurry",
            message="IMAGE QUALITY TOO LOW\nPlease hold the product steady and scan again.",
            blur_variance=blur,
            brightness=brightness,
            contrast=contrast,
            width=width,
            height=height,
        )
    if blur < (BLUR_VARIANCE_MIN * 0.45):
        return QualityResult(
            ok=False,
            reason="too_blurry",
            message="IMAGE QUALITY TOO LOW\nPlease hold the product steady and scan again.",
            blur_variance=blur,
            brightness=brightness,
            contrast=contrast,
            width=width,
            height=height,
        )

    return QualityResult(
        ok=True,
        blur_variance=blur,
        brightness=brightness,
        contrast=contrast,
        width=width,
        height=height,
    )
