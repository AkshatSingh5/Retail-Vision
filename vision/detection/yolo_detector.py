from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import NotRequired, TypedDict

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.downloads import attempt_download_asset

from backend.app.config import CONFIDENCE_THRESHOLD, INFER_IMGSZ, IOU_THRESHOLD, MODEL_PATH, ROOT_DIR, YOLO_DEVICE
from vision.device import gpu_name, resolve_device

OFFICIAL_MODEL_NAME = "yolo26m.pt"

_lock = threading.Lock()
_detector: "YOLODetector | None" = None


class Detection(TypedDict):
    class_id: int
    confidence: float
    bbox: list[float]
    track_id: NotRequired[int | None]


class YOLODetector:
    """YOLO26m detector for a single camera frame. Loaded once and reused."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence_threshold: float | None = None,
        iou_threshold: float | None = None,
    ) -> None:
        self.model_path = Path(model_path or MODEL_PATH)
        self.confidence_threshold = (
            CONFIDENCE_THRESHOLD if confidence_threshold is None else confidence_threshold
        )
        self.iou_threshold = IOU_THRESHOLD if iou_threshold is None else iou_threshold
        self.device = resolve_device(YOLO_DEVICE)

        self._ensure_weights()
        self.model = YOLO(str(self.model_path))
        self._move_to_device()
        self.names: dict[int, str] = dict(self.model.names)
        self._log_model_info()

    def _move_to_device(self) -> None:
        try:
            self.model.to(self.device)
        except Exception as extra:
            print(f"[YOLO26m] model.to({self.device}) failed: {extra}")
        inner = getattr(self.model, "model", None)
        if inner is not None and hasattr(inner, "eval"):
            inner.eval()

    def _log_model_info(self) -> None:
        from vision.detection.product_filter import is_retail_trained_model

        retail = is_retail_trained_model(self.names)
        gpu = gpu_name() or "n/a"
        print("YOLO26m model loaded")
        print(f"  Model path: {self.model_path.resolve()}")
        print(f"  Confidence threshold: {self.confidence_threshold:.2f}")
        print(f"  IoU threshold: {self.iou_threshold:.2f}")
        print(f"  Using device: {self.device}")
        if self.device.startswith("cuda"):
            print(f"  GPU: {gpu}")
        print(f"  Class count: {len(self.names)}")
        print(f"  Retail-trained: {retail}")
        for class_id in sorted(self.names):
            print(f"  class_id={class_id} class_name={self.names[class_id]!r}")

    def _ensure_weights(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        if self.model_path.exists():
            return

        print(f"Downloading pretrained {OFFICIAL_MODEL_NAME} to {self.model_path}...")
        downloaded = Path(
            attempt_download_asset(str(self.model_path), repo="ultralytics/assets", release="v8.4.0")
        )
        if downloaded.exists() and downloaded.resolve() != self.model_path.resolve():
            downloaded.replace(self.model_path)

        if not self.model_path.exists():
            cwd_copy = ROOT_DIR / OFFICIAL_MODEL_NAME
            if cwd_copy.exists():
                cwd_copy.replace(self.model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Could not load YOLO26m weights at {self.model_path}. "
                f"Place {OFFICIAL_MODEL_NAME} in vision/models/ and retry."
            )

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: float | None = None,
    ) -> tuple[list[Detection], float]:
        """Run inference and return detections plus latency in milliseconds."""
        conf = self.confidence_threshold if confidence_threshold is None else float(confidence_threshold)
        start = time.perf_counter()
        with np.errstate(all="ignore"):
            results = self.model.predict(
                source=frame,
                conf=conf,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
                **({"imgsz": INFER_IMGSZ} if INFER_IMGSZ else {}),
            )
        latency_ms = (time.perf_counter() - start) * 1000.0
        return self._parse_boxes(results[0]), latency_ms

    def track(self, frame: np.ndarray) -> tuple[list[Detection], float]:
        """Run ByteTrack via Ultralytics and return detections with track_id."""
        start = time.perf_counter()
        results = self.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
            **({"imgsz": INFER_IMGSZ} if INFER_IMGSZ else {}),
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        detections: list[Detection] = []
        result = results[0]
        if result.boxes is None:
            return detections, latency_ms

        ids = result.boxes.id
        for index, box in enumerate(result.boxes):
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].cpu().tolist())
            track_id = None
            if ids is not None:
                track_id = int(ids[index].item())
            detections.append(
                {
                    "class_id": int(box.cls[0].item()),
                    "confidence": round(float(box.conf[0].item()), 4),
                    "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                    "track_id": track_id,
                }
            )
        return detections, latency_ms

    def _parse_boxes(self, result) -> list[Detection]:
        detections: list[Detection] = []
        if result.boxes is None:
            return detections
        for box in result.boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].cpu().tolist())
            detections.append(
                {
                    "class_id": int(box.cls[0].item()),
                    "confidence": round(float(box.conf[0].item()), 4),
                    "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                }
            )
        return detections

    def annotate(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        fps: float | None = None,
        latency_ms: float | None = None,
    ) -> np.ndarray:
        annotated = frame.copy()
        for detection in detections:
            self._draw_detection(annotated, detection)
        self._draw_metrics(annotated, fps=fps, latency_ms=latency_ms, count=len(detections))
        return annotated

    def _draw_detection(self, frame: np.ndarray, detection: Detection) -> None:
        x1, y1, x2, y2 = (int(v) for v in detection["bbox"])
        class_id = detection["class_id"]
        confidence = detection["confidence"]
        class_name = self.names.get(class_id, "object")
        color = _color_for_class(class_id)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        lines = [
            class_name,
            f"Confidence: {confidence:.2f}",
            f"Class ID: {class_id}",
        ]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        padding = 4
        line_height = 18
        box_width = 0
        for line in lines:
            (tw, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
            box_width = max(box_width, tw)
        box_height = line_height * len(lines) + padding

        text_y = y1 - box_height - 2
        if text_y < 0:
            text_y = y1 + 2

        cv2.rectangle(
            frame,
            (x1, text_y),
            (x1 + box_width + padding * 2, text_y + box_height),
            color,
            thickness=cv2.FILLED,
        )
        for i, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (x1 + padding, text_y + (i + 1) * line_height - 4),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

    def _draw_metrics(
        self,
        frame: np.ndarray,
        fps: float | None,
        latency_ms: float | None,
        count: int,
    ) -> None:
        lines = []
        if fps is not None:
            lines.append(f"FPS: {fps:.0f}")
        if latency_ms is not None:
            lines.append(f"Latency: {latency_ms:.0f} ms")
        lines.append(f"Detections: {count}")
        lines.append(f"Conf: {self.confidence_threshold:.2f}")
        lines.append(f"Device: {self.device}")

        y = 24
        for line in lines:
            cv2.putText(
                frame,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            y += 26


def _color_for_class(class_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(class_id + 17)
    b, g, r = (int(v) for v in rng.integers(40, 255, size=3))
    return b, g, r


def get_yolo_detector() -> YOLODetector:
    """Load YOLO26m once and reuse it for scan + camera."""
    global _detector
    with _lock:
        if _detector is None:
            _detector = YOLODetector()
        return _detector


def reset_yolo_detector() -> None:
    global _detector
    with _lock:
        _detector = None
