"""Evaluate a trained Retail Vision YOLO26m checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.config import ROOT_DIR as PROJECT_ROOT
from scripts.dataset_common import class_names, detect_device

SIMILAR_PAIRS = [
    (0, 4, "Coca-Cola 500ml", "Pepsi 500ml"),
    (4, 0, "Pepsi 500ml", "Coca-Cola 500ml"),
    (5, 0, "Sprite 500ml", "Coca-Cola 500ml"),
    (1, 7, "Lays Classic", "Kurkure Masala Munch"),
    (7, 1, "Kurkure Masala Munch", "Lays Classic"),
    (3, 6, "Dairy Milk", "KitKat"),
    (6, 3, "KitKat", "Dairy Milk"),
]


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _as_list(value) -> list[float]:
    array = np.array(value, dtype=float).reshape(-1)
    return [float(item) for item in array.tolist()]


def _metrics_from_ultralytics(metrics, names: dict[int, str]) -> dict:
    box = metrics.box
    precision = float(np.mean(box.p)) if len(box.p) else 0.0
    recall = float(np.mean(box.r)) if len(box.r) else 0.0
    per_class = []
    maps = _as_list(getattr(box, "maps", []))
    precisions = _as_list(box.p)
    recalls = _as_list(box.r)
    f1s = _as_list(box.f1) if len(getattr(box, "f1", [])) else []
    class_ids = [int(item) for item in getattr(box, "ap_class_index", range(len(names)))]
    if not class_ids:
        class_ids = list(names)
    for index, class_id in enumerate(class_ids):
        p = precisions[index] if index < len(precisions) else 0.0
        r = recalls[index] if index < len(recalls) else 0.0
        f1 = f1s[index] if index < len(f1s) else _f1(p, r)
        map50_95 = maps[class_id] if class_id < len(maps) else (maps[index] if index < len(maps) else 0.0)
        per_class.append(
            {
                "class_id": int(class_id),
                "name": names.get(int(class_id), str(class_id)),
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(float(f1), 4),
                "mAP50-95": round(float(map50_95), 4),
            }
        )
    per_class.sort(key=lambda item: item["f1"])
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(_f1(precision, recall), 4),
        "mAP50": round(float(box.map50), 4),
        "mAP50-95": round(float(box.map), 4),
        "per_class": per_class,
    }


def _confusion_report(metrics, names: dict[int, str]) -> dict:
    matrix = np.array(metrics.confusion_matrix.matrix, dtype=int)
    nc = len(names)
    if matrix.shape[0] < nc:
        return {"false_positives": [], "false_negatives": [], "confusions": [], "similar_product_pairs": []}

    false_positives = []
    false_negatives = []
    confusions = []
    background = nc if matrix.shape[0] > nc else None
    for class_id, name in names.items():
        fp = int(matrix[class_id, background]) if background is not None else 0
        fn = int(matrix[background, class_id]) if background is not None else 0
        tp = int(matrix[class_id, class_id])
        false_positives.append({"class_id": class_id, "name": name, "count": fp, "true_positives": tp})
        false_negatives.append({"class_id": class_id, "name": name, "count": fn, "true_positives": tp})
        for pred_id, pred_name in names.items():
            if pred_id == class_id:
                continue
            count = int(matrix[pred_id, class_id])
            if count:
                confusions.append(
                    {
                        "true": name,
                        "predicted": pred_name,
                        "count": count,
                    }
                )

    false_positives.sort(key=lambda item: item["count"], reverse=True)
    false_negatives.sort(key=lambda item: item["count"], reverse=True)
    confusions.sort(key=lambda item: item["count"], reverse=True)

    similar = []
    for true_id, pred_id, true_name, pred_name in SIMILAR_PAIRS:
        count = int(matrix[pred_id, true_id]) if true_id < matrix.shape[1] and pred_id < matrix.shape[0] else 0
        similar.append(
            {
                "true": true_name,
                "predicted": pred_name,
                "count": count,
                "note": f"{true_name} confused as {pred_name}",
            }
        )
    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "confusions": confusions,
        "similar_product_pairs": similar,
        "matrix_shape": list(matrix.shape),
    }


def _markdown(report: dict) -> str:
    overall = report["test"]
    lines = [
        "# Retail Vision — YOLO26m evaluation",
        "",
        f"Weights: `{report['weights']}`",
        f"Device: `{report['device']}`",
        "",
        "## Test-set metrics",
        "",
        f"- Precision: **{overall['precision']:.4f}**",
        f"- Recall: **{overall['recall']:.4f}**",
        f"- F1: **{overall['f1']:.4f}**",
        f"- mAP50: **{overall['mAP50']:.4f}**",
        f"- mAP50-95: **{overall['mAP50-95']:.4f}**",
        "",
        "## Validation-set metrics",
        "",
        f"- Precision: {report['val']['precision']:.4f}",
        f"- Recall: {report['val']['recall']:.4f}",
        f"- F1: {report['val']['f1']:.4f}",
        f"- mAP50: {report['val']['mAP50']:.4f}",
        f"- mAP50-95: {report['val']['mAP50-95']:.4f}",
        "",
        "## Per-class test performance (worst first)",
        "",
        "| Class | Precision | Recall | F1 | mAP50-95 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in overall["per_class"]:
        lines.append(
            f"| {row['name']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['mAP50-95']:.4f} |"
        )
    weak = [row["name"] for row in overall["per_class"] if row["f1"] < 0.6]
    lines += ["", "## Products that need more data", ""]
    if weak:
        for name in weak:
            lines.append(f"- {name}")
    else:
        lines.append("- No class fell below F1 0.60 on this seed test set. Grow real photos before production use.")

    confusion = report["confusion"]
    lines += ["", "## Confusion analysis", "", "### False positives (background predicted as product)", ""]
    for row in confusion["false_positives"]:
        lines.append(f"- {row['name']}: {row['count']}")
    lines += ["", "### False negatives (missed products)", ""]
    for row in confusion["false_negatives"]:
        lines.append(f"- {row['name']}: {row['count']}")
    lines += ["", "### Class confusions", ""]
    if confusion["confusions"]:
        for row in confusion["confusions"]:
            lines.append(f"- {row['true']} → {row['predicted']}: {row['count']}")
    else:
        lines.append("- No off-diagonal class confusions on the evaluated split.")
    lines += ["", "### Similar-product pairs", ""]
    for row in confusion["similar_product_pairs"]:
        lines.append(f"- {row['note']}: {row['count']}")
    lines.append("")
    return "\n".join(lines)


def evaluate_run(run_dir: Path, data: Path, device: str, imgsz: int) -> dict:
    weights = run_dir / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"Missing weights: {weights}")
    names = class_names()
    model = YOLO(str(weights))
    val_metrics = model.val(
        data=str(data),
        split="val",
        device=device,
        imgsz=imgsz,
        plots=True,
        verbose=False,
        project=str(run_dir.parent),
        name=f"{run_dir.name}_val",
        exist_ok=True,
    )
    test_metrics = model.val(
        data=str(data),
        split="test",
        device=device,
        imgsz=imgsz,
        plots=True,
        verbose=False,
        project=str(run_dir.parent),
        name=f"{run_dir.name}_test",
        exist_ok=True,
    )
    report = {
        "weights": str(weights),
        "device": device,
        "imgsz": imgsz,
        "val": _metrics_from_ultralytics(val_metrics, names),
        "test": _metrics_from_ultralytics(test_metrics, names),
        "confusion": _confusion_report(test_metrics, names),
    }
    (run_dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (run_dir / "evaluation.md").write_text(_markdown(report), encoding="utf-8")
    print(_markdown(report))
    print(f"Wrote {run_dir / 'evaluation.md'}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLO26m run.")
    parser.add_argument("--run", type=Path, required=True, help="Run directory, e.g. runs/retail_yolo26m_v1")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    evaluate_run(args.run, args.data, detect_device(args.device), args.imgsz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
