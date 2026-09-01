from __future__ import annotations

import time
from typing import TypedDict

import cv2
import numpy as np

from vision.detection.crop import crop_box
from vision.detection.product_filter import filter_retail_detections
from vision.detection.yolo_detector import YOLODetector, _color_for_class, get_yolo_detector
from vision.recognition.embedding import EmbeddingRefiner
from vision.recognition.identity import IdentifiedDetection, ProductIdentifier
from vision.tracking.manager import TrackManager
from vision.tracking.tracks import TrackedProduct


class PipelineFrame(TypedDict):
    tracks: list[TrackedProduct]
    public: list[dict]
    prices: list[dict]
    cart: dict[str, int]
    latency_ms: float
    detections: int


class TrackingPipeline:
    """Camera frame → YOLO26m → identity → MOT → stabilized cart output."""

    def __init__(
        self,
        detector: YOLODetector | None = None,
        identifier: ProductIdentifier | None = None,
        manager: TrackManager | None = None,
        use_ultralytics_track: bool = True,
        stable_frames: int = 5,
        max_missing: int = 20,
        track_iou_threshold: float = 0.30,
        enable_embedding: bool = False,
    ) -> None:
        self.detector = detector or get_yolo_detector()
        refiner = EmbeddingRefiner(enabled=enable_embedding)
        self.identifier = identifier or ProductIdentifier(refiner=refiner)
        self.manager = manager or TrackManager(
            stable_frames=stable_frames,
            max_missing=max_missing,
            iou_threshold=track_iou_threshold,
        )
        self.use_ultralytics_track = use_ultralytics_track

    def process(self, frame: np.ndarray) -> PipelineFrame:
        start = time.perf_counter()
        assigned_ids: list[tuple[int, IdentifiedDetection]] | None = None
        if self.use_ultralytics_track:
            raw, _latency = self.detector.track(frame)
            raw = filter_retail_detections(raw, self.detector.names)
            identified = self.identifier.identify_all(raw, crops=[_crop_box(frame, item["bbox"]) for item in raw])
            assigned_ids = []
            missing = False
            for detection, identified_det in zip(raw, identified):
                track_id = detection.get("track_id")
                if track_id is None:
                    missing = True
                    break
                assigned_ids.append((int(track_id), identified_det))
            if missing:
                assigned_ids = None
        else:
            raw, _latency = self.detector.detect(frame)
            raw = filter_retail_detections(raw, self.detector.names)
            crops = [_crop_box(frame, item["bbox"]) for item in raw]
            identified = self.identifier.identify_all(raw, crops=crops)

        tracks = self.manager.update(identified, assigned=assigned_ids)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return {
            "tracks": tracks,
            "public": self.manager.visible_outputs(),
            "prices": self.manager.price_outputs(),
            "cart": self.manager.cart_by_name(),
            "latency_ms": latency_ms,
            "detections": len(identified),
        }

    def process_detections(self, detections: list[IdentifiedDetection]) -> PipelineFrame:
        start = time.perf_counter()
        tracks = self.manager.update(detections)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return {
            "tracks": tracks,
            "public": self.manager.visible_outputs(),
            "prices": self.manager.price_outputs(),
            "cart": self.manager.cart_by_name(),
            "latency_ms": latency_ms,
            "detections": len(detections),
        }

    def annotate(self, frame: np.ndarray, result: PipelineFrame, fps: float | None = None) -> np.ndarray:
        annotated = frame.copy()
        for track in result["tracks"]:
            _draw_track(annotated, track)
        _draw_hud(annotated, result, fps=fps, device=self.detector.device)
        return annotated


def _crop_box(frame: np.ndarray, bbox: list[float]) -> np.ndarray | None:
    return crop_box(frame, bbox)


def _draw_track(frame: np.ndarray, track: TrackedProduct) -> None:
    x1, y1, x2, y2 = (int(v) for v in track["bbox"])
    color = _color_for_class(track["class_id"])
    thickness = 3 if track["confirmed"] else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    confidence_pct = int(round(float(track["confidence"]) * 100))
    if track.get("is_unknown"):
        lines = ["New Product Detected", f"Confidence: {confidence_pct}%"]
    else:
        lines = [str(track["product_name"]), f"Confidence: {confidence_pct}%"]
    sku = str(track.get("sku") or "")
    if sku and not sku.lower().startswith("unknown"):
        lines.append(f"SKU: {sku}")
    if track.get("price") is not None:
        lines.append(f"Price: Rs {track['price']}")
    font = cv2.FONT_HERSHEY_SIMPLEX
    sizes = [cv2.getTextSize(line, font, 0.5, 1)[0] for line in lines]
    width = max(size[0] for size in sizes) + 16
    line_h = 20
    box_h = 10 + line_h * len(lines)
    top = y1 - box_h - 6 if y1 - box_h - 6 > 0 else y2 + 6
    cv2.rectangle(frame, (x1, top), (x1 + width, top + box_h), color, cv2.FILLED)
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x1 + 8, top + 18 + index * line_h),
            font,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def _draw_hud(frame: np.ndarray, result: PipelineFrame, fps: float | None, device: str) -> None:
    lines = []
    if fps is not None:
        lines.append(f"FPS: {fps:.0f}")
    lines.append(f"Latency: {result['latency_ms']:.0f} ms")
    lines.append("YOLO26m ACTIVE")
    lines.append(f"Tracks: {len(result['tracks'])}")
    lines.append(f"Device: {device}")
    if not result["tracks"]:
        lines.append("No products detected")
    lines.append("CART")
    if result["cart"]:
        prices_by_name = {item["name"]: item["price"] for item in result["prices"]}
        for name, qty in result["cart"].items():
            price = prices_by_name.get(name)
            if price is None:
                lines.append(f"  {name} x {qty}")
            else:
                lines.append(f"  {name} x {qty}  Rs {price}")
    else:
        lines.append("  (empty)")

    y = 24
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        y += 22
