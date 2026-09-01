"""Scan / visual recognition service (YOLO → DINOv2 → pgvector)."""

from backend.app.services.scan_service import (
    ScanError,
    recognize_frame,
    validate_image_bytes,
)

__all__ = ["ScanError", "recognize_frame", "validate_image_bytes"]
