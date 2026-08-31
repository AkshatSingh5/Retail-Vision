"""Shared helpers for product-scan tests (mock YOLO as a single product crop)."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

import numpy as np


def _single_detection(frame: np.ndarray):
    height, width = frame.shape[:2]
    meta = {
        "class_id": 0,
        "confidence": 0.95,
        "bbox": [0.0, 0.0, float(width), float(height)],
        "source": "yolo",
        "class_name": "product",
        "retail_trained": True,
    }
    info = {
        "model_path": "test",
        "model_loaded": True,
        "classes": {0: "product"},
        "detection_count": 1,
        "retail_trained": True,
        "error": None,
        "raw_detections": [
            {
                "class_id": 0,
                "class_name": "product",
                "confidence": 0.95,
                "bbox": [0.0, 0.0, float(width), float(height)],
            }
        ],
    }
    return [(frame.copy(), meta)], info


def _multi_detection(frame: np.ndarray):
    height, width = frame.shape[:2]
    left = frame[:, : width // 2].copy()
    right = frame[:, width // 2 :].copy()
    meta_a = {
        "class_id": 0,
        "confidence": 0.9,
        "bbox": [0.0, 0.0, float(width // 2), float(height)],
        "source": "yolo",
        "class_name": "product",
    }
    meta_b = {
        "class_id": 1,
        "confidence": 0.88,
        "bbox": [float(width // 2), 0.0, float(width), float(height)],
        "source": "yolo",
        "class_name": "product",
    }
    info = {
        "model_path": "test",
        "model_loaded": True,
        "classes": {0: "product", 1: "product"},
        "detection_count": 2,
        "retail_trained": True,
        "error": None,
        "raw_detections": [],
    }
    return [(left, meta_a), (right, meta_b)], info


def _zero_detection(frame: np.ndarray):
    return [], {
        "model_path": "test",
        "model_loaded": True,
        "classes": {0: "product"},
        "detection_count": 0,
        "retail_trained": True,
        "error": None,
        "raw_detections": [],
    }


@contextmanager
def mock_yolo_single_product():
    with patch("backend.app.services.scan_service._detect_crops", side_effect=_single_detection):
        yield


@contextmanager
def mock_yolo_multiple_products():
    with patch("backend.app.services.scan_service._detect_crops", side_effect=_multi_detection):
        yield


@contextmanager
def mock_yolo_no_product():
    with patch("backend.app.services.scan_service._detect_crops", side_effect=_zero_detection):
        yield
