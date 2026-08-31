"""Propose YOLO bounding boxes for single-product photos in dataset/raw."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.dataset_common import (
    RAW_DIR,
    load_image,
    load_manifest,
    save_manifest,
    write_yolo_labels,
    xyxy_to_yolo,
)
from vision.detection.yolo_detector import YOLODetector

SKIP_COCO = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "chair",
    "couch",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "keyboard",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
}


def _best_box(detections: list[dict], names: dict[int, str], width: int, height: int) -> list[float] | None:
    image_area = float(width * height)
    ranked: list[tuple[float, list[float]]] = []
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        area = max(0.0, (x2 - x1) * (y2 - y1)) / image_area
        if area < 0.02 or area > 0.98:
            continue
        coco_name = names.get(detection["class_id"], "")
        if coco_name in SKIP_COCO:
            continue
        ranked.append((area, detection["bbox"]))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1]


def annotate_raw(confidence: float) -> None:
    manifest = load_manifest()
    if not manifest:
        raise SystemExit("No raw images. Run scripts/collect_seed_images.py first.")

    detector = YOLODetector(confidence_threshold=confidence)
    labeled = 0
    skipped = 0
    kept: list[dict] = []

    for entry in manifest:
        image_path = RAW_DIR / "images" / entry["file"]
        if not image_path.exists():
            print(f"skip (missing file): {entry['file']}")
            skipped += 1
            continue
        image = load_image(image_path)
        if image is None:
            skipped += 1
            continue
        height, width = image.shape[:2]
        detections, _latency = detector.detect(image)
        box = _best_box(detections, detector.names, width, height)
        if box is None:
            # Catalog-style photos are usually a single centered pack.
            margin_x, margin_y = width * 0.12, height * 0.10
            box = [margin_x, margin_y, width - margin_x, height - margin_y]
            print(f"fallback box: {entry['file']}")
        yolo = xyxy_to_yolo(*box, width, height)
        label_file = RAW_DIR / "labels" / f"{image_path.stem}.txt"
        write_yolo_labels(label_file, [(int(entry["class_id"]), *yolo)])
        entry["labeled"] = True
        kept.append(entry)
        labeled += 1
        print(f"labeled {entry['file']} class={entry['class_id']}")

    save_manifest(kept)
    print(f"Labeled {labeled}, skipped {skipped}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-label raw product photos.")
    parser.add_argument("--confidence", type=float, default=0.15)
    args = parser.parse_args()
    annotate_raw(args.confidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
