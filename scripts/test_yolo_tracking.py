"""YOLO + identity + tracking on a 5-product composite scene.

Run from the project root:

    python scripts/test_yolo_tracking.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.dataset_common import DATASET_DIR, IMAGE_EXTENSIONS
from vision.detection.yolo_detector import YOLODetector
from vision.recognition.identity import ProductIdentifier
from vision.tracking.manager import TrackManager


def _label_path(image_path: Path, split: str) -> Path:
    return DATASET_DIR / "labels" / split / f"{image_path.stem}.txt"


def _read_boxes(path: Path) -> list[tuple[int, list[float]]]:
    boxes: list[tuple[int, list[float]]] = []
    if not path.exists():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        cx, cy, w, h = (float(v) for v in parts[1:5])
        boxes.append((class_id, [cx, cy, w, h]))
    return boxes


def _crop(image: np.ndarray, xywhn: list[float], pad: float = 0.08) -> np.ndarray:
    height, width = image.shape[:2]
    cx, cy, w, h = xywhn
    x1 = int(max(0, (cx - w / 2 - pad) * width))
    y1 = int(max(0, (cy - h / 2 - pad) * height))
    x2 = int(min(width, (cx + w / 2 + pad) * width))
    y2 = int(min(height, (cy + h / 2 + pad) * height))
    if x2 <= x1 or y2 <= y1:
        return image
    return image[y1:y2, x1:x2]


def _first_crop_for_class(class_id: int) -> np.ndarray | None:
    for split in ("test", "val", "train"):
        folder = DATASET_DIR / "images" / split
        if not folder.exists():
            continue
        for image_path in sorted(folder.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            for label_id, xywhn in _read_boxes(_label_path(image_path, split)):
                if label_id != class_id:
                    continue
                frame = cv2.imread(str(image_path))
                if frame is None:
                    continue
                crop = _crop(frame, xywhn)
                if crop.size:
                    return crop
    return None


def compose_scene(class_ids: list[int], size: int = 960) -> np.ndarray:
    canvas = np.full((size, size, 3), 30, dtype=np.uint8)
    slot_w = size // len(class_ids)
    for index, class_id in enumerate(class_ids):
        crop = _first_crop_for_class(class_id)
        if crop is None:
            raise FileNotFoundError(f"No labeled crop for class_id={class_id}")
        target_w = max(80, slot_w - 24)
        scale = target_w / crop.shape[1]
        target_h = max(80, int(crop.shape[0] * scale))
        resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)
        x = index * slot_w + 12
        y = (size - target_h) // 2
        canvas[y : y + target_h, x : x + target_w] = resized
    return canvas


def main() -> int:
    class_ids = [0, 1, 2, 3, 4]
    print("Composing 5-product scene: Coke, Lays, Maggi, Dairy Milk, Pepsi")
    scene = compose_scene(class_ids)
    out_dir = ROOT_DIR / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "five_product_scene.jpg"), scene)

    detector = YOLODetector(confidence_threshold=0.15)
    identifier = ProductIdentifier()
    manager = TrackManager(stable_frames=3, max_missing=8, iou_threshold=0.25)
    last_public: list[dict] = []
    for frame_index in range(12):
        dx = (frame_index % 4) * 2
        shifted = np.roll(scene, dx, axis=1)
        detections, latency_ms = detector.detect(shifted)
        identified = identifier.identify_all(detections)
        tracks = manager.update(identified)
        last_public = manager.visible_outputs()
        print(
            f"frame {frame_index:02d}  detections={len(identified)}  "
            f"tracks={len(tracks)}  confirmed={len(last_public)}  "
            f"{latency_ms:.0f} ms"
        )

    cart = manager.cart_by_name()
    payload = {
        "frames": 12,
        "products_in_scene": 5,
        "unique_track_ids": sorted({item["track_id"] for item in manager.tracks.values()}),
        "cart": cart,
        "public": last_public,
    }
    (out_dir / "five_product_tracking.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Cart:", json.dumps(cart, indent=2))
    print("Sample tracks:", json.dumps(last_public, indent=2))

    unique_ids = len(payload["unique_track_ids"])
    if unique_ids > 12:
        print("FAIL: track IDs exploded (duplicate prevention failed)")
        return 1
    print("PASS: track count stayed bounded across 12 frames of 5 simultaneous products")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
