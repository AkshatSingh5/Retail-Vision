"""Measure recognition across 1000+ realistic visual variations of the test set.

This does not treat augmentation as extra unique products. It asks whether the
trained model still finds the correct class when held-out test photos are
shown at different angles, distances, and lighting.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.config import ROOT_DIR as PROJECT_ROOT
from scripts.dataset_common import (
    IMAGES_DIR,
    LABELS_DIR,
    class_names,
    detect_device,
    list_images,
    load_image,
    read_yolo_labels,
)

ROTATIONS = (-60, -30, -15, 0, 15, 30, 45, 60)
BRIGHTNESS = (-60, -30, 0, 30, 60)
SCALES = (0.7, 1.0, 1.3)


def _letterbox(image: np.ndarray, scale: float) -> np.ndarray:
    height, width = image.shape[:2]
    new_w, new_h = max(1, int(width * scale)), max(1, int(height * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros_like(image)
    if scale <= 1.0:
        x0 = (width - new_w) // 2
        y0 = (height - new_h) // 2
        canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
        return canvas
    x0 = (new_w - width) // 2
    y0 = (new_h - height) // 2
    return resized[y0 : y0 + height, x0 : x0 + width]


def _rotate(image: np.ndarray, angle: float) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    return cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT_101)


def _brightness(image: np.ndarray, delta: int) -> np.ndarray:
    return cv2.convertScaleAbs(image, alpha=1.0, beta=delta)


def _variant(image: np.ndarray, angle: float, brightness: int, scale: float) -> np.ndarray:
    return _brightness(_letterbox(_rotate(image, angle), scale), brightness)


def _hit(result, class_id: int, confidence: float) -> bool:
    if result.boxes is None or len(result.boxes) == 0:
        return False
    for box in result.boxes:
        if int(box.cls[0].item()) == class_id and float(box.conf[0].item()) >= confidence:
            return True
    return False


def evaluate_robustness(
    weights: Path,
    data_yaml: Path,
    device: str,
    imgsz: int,
    confidence: float = 0.35,
) -> dict:
    names = class_names()
    images = list_images(IMAGES_DIR / "test")
    if not images:
        raise SystemExit("No test images found.")
    model = YOLO(str(weights))
    variants = list(itertools.product(ROTATIONS, BRIGHTNESS, SCALES))
    total = len(images) * len(variants)
    hits = 0
    per_class = {class_id: {"total": 0, "hits": 0} for class_id in names}
    failures: dict[str, int] = defaultdict(int)

    print(f"Robustness protocol: {len(images)} test images x {len(variants)} variations = {total}")
    index = 0
    for image_path in images:
        image = load_image(image_path)
        if image is None:
            continue
        labels = read_yolo_labels(LABELS_DIR / "test" / f"{image_path.stem}.txt")
        if not labels:
            continue
        class_id = int(labels[0][0])
        for angle, brightness, scale in variants:
            variant = _variant(image, angle, brightness, scale)
            result = model.predict(
                source=variant,
                imgsz=imgsz,
                device=device,
                conf=confidence,
                verbose=False,
            )[0]
            ok = _hit(result, class_id, confidence)
            per_class[class_id]["total"] += 1
            if ok:
                hits += 1
                per_class[class_id]["hits"] += 1
            else:
                failures[names[class_id]] += 1
            index += 1
            if index % 100 == 0:
                print(f"  {index}/{total}  acc={hits / index:.3f}")

    per_class_report = []
    for class_id, name in names.items():
        stats = per_class[class_id]
        rate = stats["hits"] / stats["total"] if stats["total"] else 0.0
        per_class_report.append(
            {
                "class_id": class_id,
                "name": name,
                "hits": stats["hits"],
                "total": stats["total"],
                "recognition_rate": round(rate, 4),
            }
        )
    per_class_report.sort(key=lambda item: item["recognition_rate"])
    report = {
        "weights": str(weights),
        "device": device,
        "test_images": len(images),
        "variations_per_image": len(variants),
        "total_evaluations": total,
        "hits": hits,
        "recognition_rate": round(hits / total if total else 0.0, 4),
        "protocol": {
            "rotations_deg": list(ROTATIONS),
            "brightness_delta": list(BRIGHTNESS),
            "scales": list(SCALES),
            "goal": "Recognize each held-out product under 1000+ realistic visual variations.",
        },
        "per_class": per_class_report,
        "most_failures": dict(sorted(failures.items(), key=lambda item: item[1], reverse=True)),
    }
    out_dir = weights.parent.parent
    (out_dir / "robustness.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("total_evaluations", "hits", "recognition_rate")}, indent=2))
    print(f"Wrote {out_dir / 'robustness.json'}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="1000+ variation test-set protocol.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.35)
    args = parser.parse_args()
    evaluate_robustness(args.weights, args.data, detect_device(args.device), args.imgsz, args.confidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
