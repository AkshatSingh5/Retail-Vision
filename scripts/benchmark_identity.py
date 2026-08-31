"""Benchmark YOLO26m class_id → SKU identity without a second model.

Run from the project root:

    python scripts/benchmark_identity.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.config import CONFIDENCE_THRESHOLD, MODEL_PATH, ROOT_DIR as PROJECT_ROOT
from scripts.dataset_common import DATASET_DIR, IMAGE_EXTENSIONS, class_names
from vision.detection.yolo_detector import YOLODetector
from vision.recognition.identity import ProductIdentifier
from vision.tracking.iou_tracker import box_iou

SIMILAR_PAIRS = {
    (0, 4),
    (4, 0),
    (0, 5),
    (5, 0),
    (1, 7),
    (7, 1),
    (3, 6),
    (6, 3),
}
IDENTITY_OK = 0.70
MATCH_IOU = 0.30


def _load_yolo_label(path: Path) -> list[tuple[int, list[float]]]:
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


def _xywhn_to_xyxy(box: list[float], width: int, height: int) -> list[float]:
    cx, cy, w, h = box
    x1 = (cx - w / 2) * width
    y1 = (cy - h / 2) * height
    x2 = (cx + w / 2) * width
    y2 = (cy + h / 2) * height
    return [x1, y1, x2, y2]


def _list_split_images(split: str) -> list[Path]:
    folder = DATASET_DIR / "images" / split
    if not folder.exists():
        return []
    return sorted(
        path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _score_pass(
    images: list[Path],
    identifier: ProductIdentifier,
    names: dict[int, str],
    raw_by_image: dict[str, list],
    min_conf: float,
) -> dict:
    matched = 0
    identity_correct = 0
    misses = 0
    similar_confusions = 0
    per_class = defaultdict(lambda: {"gt": 0, "correct": 0, "wrong": 0, "miss": 0})
    confusion = Counter()

    for image_path in images:
        frame_dets = [
            item for item in raw_by_image.get(image_path.name, []) if item["confidence"] >= min_conf
        ]
        identified = identifier.identify_all(frame_dets)
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        height, width = frame.shape[:2]
        split = image_path.parent.name
        ground_truth = _load_yolo_label(DATASET_DIR / "labels" / split / f"{image_path.stem}.txt")
        used: set[int] = set()
        for class_id, xywhn in ground_truth:
            gt_box = _xywhn_to_xyxy(xywhn, width, height)
            per_class[class_id]["gt"] += 1
            best_iou = 0.0
            best_index = -1
            for index, detection in enumerate(identified):
                if index in used:
                    continue
                score = box_iou(gt_box, detection["bbox"])
                if score > best_iou:
                    best_iou = score
                    best_index = index
            if best_index < 0 or best_iou < MATCH_IOU:
                misses += 1
                per_class[class_id]["miss"] += 1
                continue
            used.add(best_index)
            matched += 1
            predicted = identified[best_index]["class_id"]
            if predicted == class_id:
                identity_correct += 1
                per_class[class_id]["correct"] += 1
            else:
                per_class[class_id]["wrong"] += 1
                confusion[(class_id, predicted)] += 1
                if (class_id, predicted) in SIMILAR_PAIRS:
                    similar_confusions += 1

    total_gt = sum(item["gt"] for item in per_class.values())
    accuracy = identity_correct / total_gt if total_gt else 0.0
    recall = matched / total_gt if total_gt else 0.0
    identity_among_matches = identity_correct / matched if matched else 0.0
    return {
        "min_confidence": min_conf,
        "ground_truth_objects": total_gt,
        "matched_detections": matched,
        "identity_correct": identity_correct,
        "misses": misses,
        "detection_recall": round(recall, 4),
        "identity_accuracy": round(accuracy, 4),
        "identity_accuracy_among_matches": round(identity_among_matches, 4),
        "similar_product_confusions": similar_confusions,
        "per_class": [
            {
                "class_id": class_id,
                "name": names.get(class_id, str(class_id)),
                "sku": identifier.identity_for_class(class_id).sku,
                "product_id": identifier.identity_for_class(class_id).product_id,
                **stats,
            }
            for class_id, stats in sorted(per_class.items())
        ],
        "confusion": [
            {
                "true_class": true_id,
                "pred_class": pred_id,
                "count": count,
                "true_sku": identifier.identity_for_class(true_id).sku,
                "pred_sku": identifier.identity_for_class(pred_id).sku,
            }
            for (true_id, pred_id), count in confusion.most_common()
        ],
    }


def benchmark(split: str = "test") -> dict:
    identifier = ProductIdentifier()
    detector = YOLODetector(confidence_threshold=0.10)
    names = class_names()
    images = _list_split_images(split)
    if not images:
        raise FileNotFoundError(f"No images found in dataset/images/{split}")

    raw_by_image: dict[str, list] = {}
    for image_path in images:
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        detections, _latency = detector.detect(frame)
        raw_by_image[image_path.name] = detections

    operating = _score_pass(images, identifier, names, raw_by_image, CONFIDENCE_THRESHOLD)
    diagnostic = _score_pass(images, identifier, names, raw_by_image, 0.10)
    classification_weak = diagnostic["matched_detections"] > 0 and (
        diagnostic["identity_accuracy_among_matches"] < IDENTITY_OK
        or diagnostic["similar_product_confusions"] > 0
    )
    if operating["detection_recall"] < 0.5:
        reason = (
            "YOLO26m misses most objects at the live confidence threshold. "
            "That is a detection/recall problem, not a same-SKU identity collision. "
            "No second embedding model was added; retrain with a larger dataset first."
        )
        recommend_embedding = False
    elif classification_weak:
        reason = (
            "When boxes are produced, YOLO class_id mapping is below the identity threshold "
            "or confuses similar SKUs. Embedding refinement remains available but unloaded."
        )
        recommend_embedding = True
    else:
        reason = "YOLO class_id mapping is sufficient for this split. No second model added."
        recommend_embedding = False

    return {
        "model": str(Path(MODEL_PATH)),
        "split": split,
        "images": len(images),
        "second_stage_loaded": False,
        "recommend_embedding_stage": recommend_embedding,
        "reason": reason,
        "operating_point": operating,
        "low_confidence_diagnostic": diagnostic,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark YOLO-only product identity.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()
    report = benchmark(args.split)
    out_dir = PROJECT_ROOT / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "identity_benchmark.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
