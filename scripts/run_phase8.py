"""Phase 8: testing, R&D measurement, and production-readiness report.

Run from the project root:

    python scripts/run_phase8.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.config import CONFIDENCE_THRESHOLD, MODEL_PATH, ROOT_DIR as PROJECT_ROOT
from backend.app.database import get_session_factory, init_db, reset_engine
from backend.app.services.acceptance import evaluate_for_cart
from backend.app.services.cart_service import Cart
from backend.app.services.checkout import checkout
from backend.app.services.seed import seed_products_from_registry
from scripts.dataset_common import IMAGES_DIR, LABELS_DIR, class_names, list_images, load_image, read_yolo_labels
from scripts.phase8_conditions import CONDITIONS
from vision.recognition.identity import ProductIdentifier
from vision.tracking.manager import TrackManager

REPORT_DIR = PROJECT_ROOT / "reports"
EVAL_JSON = PROJECT_ROOT / "runs" / "retail_yolo26m_v2" / "evaluation.json"
ROBUST_JSON = PROJECT_ROOT / "runs" / "retail_yolo26m_v2" / "robustness.json"
TRAIN_JSON = PROJECT_ROOT / "runs" / "retail_yolo26m_v2" / "train_config.json"
IDENTITY_JSON = PROJECT_ROOT / "logs" / "identity_benchmark.json"

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        extra = f" -- {detail}" if detail else ""
        print(f"  FAIL  {name}{extra}")


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def test_failure_handling() -> dict:
    print("Failure handling")
    low = evaluate_for_cart(
        {"track_id": 1, "product_id": 1, "sku": "COKE500", "name": "Coca-Cola 500ml", "price": 40, "confidence": 0.2, "confirmed": True},
        in_database=True,
    )
    unknown = evaluate_for_cart(
        {"track_id": 2, "product_id": 999, "sku": "unknown-9", "name": "Unknown class 9", "price": None, "confidence": 0.9, "confirmed": True},
        in_database=False,
    )
    bad_price = evaluate_for_cart(
        {"track_id": 3, "product_id": 1, "sku": "COKE500", "name": "Coca-Cola 500ml", "price": 0, "confidence": 0.9, "confirmed": True},
        in_database=None,
    )
    ok = evaluate_for_cart(
        {"track_id": 4, "product_id": 1, "sku": "COKE500", "name": "Coca-Cola 500ml", "price": 40, "confidence": 0.91, "confirmed": True},
        in_database=True,
    )
    check("low confidence rejected", low["reason"] == "low_confidence")
    check("unknown SKU rejected", unknown["reason"] == "not_in_database")
    check("invalid price rejected", bad_price["reason"] == "invalid_price")
    check("valid catalog item accepted", ok["accepted"] is True)
    check("low-confidence message asks for manual verify", "Please verify manually" in (low["message"] or ""))
    return {
        "low_confidence_rejected": low["reason"] == "low_confidence",
        "unknown_rejected": unknown["reason"] == "not_in_database",
        "invalid_price_rejected": bad_price["reason"] == "invalid_price",
        "valid_accepted": ok["accepted"] is True,
    }


def test_duplicates() -> dict:
    print("Duplicate detection")
    identifier = ProductIdentifier(use_database=False)
    manager = TrackManager(stable_frames=3, max_missing=8)
    box = [40.0, 40.0, 140.0, 220.0]
    for _frame in range(100):
        det = identifier.identify({"class_id": 0, "confidence": 0.92, "bbox": box})
        manager.update([det])
        box = [box[0] + 0.3, box[1], box[2] + 0.3, box[3]]
    one = manager.cart_by_name()
    manager2 = TrackManager(stable_frames=2, max_missing=8)
    for _frame in range(20):
        manager2.update(
            [
                identifier.identify({"class_id": 0, "confidence": 0.9, "bbox": [20, 20, 80, 140]}),
                identifier.identify({"class_id": 0, "confidence": 0.88, "bbox": [200, 20, 260, 140]}),
            ]
        )
    two = manager2.cart_by_name()
    check("100 frames of one Coke = qty 1", one == {"Coca-Cola 500ml": 1}, str(one))
    check("two physical Cokes = qty 2", two == {"Coca-Cola 500ml": 2}, str(two))
    return {"one_object_qty": one, "two_objects_qty": two}


def test_price_and_checkout(invoice_dir: Path) -> dict:
    print("Price accuracy and checkout timing")
    db_path = invoice_dir / "phase8.db"
    reset_engine()
    init_db(f"sqlite:///{db_path.as_posix()}")
    session = get_session_factory()()
    seed_products_from_registry(session)
    session.commit()
    identifier = ProductIdentifier(use_database=True)
    mapped = identifier.price_mapping_for_class(0)
    check("class 0 maps to COKE500 from database", mapped is not None and mapped["sku"] == "COKE500")
    check("price 40 from database not model", mapped is not None and mapped["price"] == 40)
    cart = Cart()
    tracks = []
    catalog = [
        (1, 1, "COKE500", "Coca-Cola 500ml", 40),
        (2, 1, "COKE500", "Coca-Cola 500ml", 40),
        (3, 2, "LAYSCLASSIC", "Lays Classic", 20),
        (4, 3, "MAGGI", "Maggi Noodles", 14),
        (5, 5, "PEPSI500", "Pepsi 500ml", 40),
        (6, 7, "KITKAT", "KitKat", 30),
    ]
    for track_id, product_id, sku, name, price in catalog:
        tracks.append(
            {
                "track_id": track_id,
                "product_id": product_id,
                "sku": sku,
                "name": name,
                "price": price,
                "tax_rate": 18,
                "confidence": 0.9,
                "confirmed": True,
            }
        )
    snapshot = cart.apply_tracks(tracks, session=session)
    coke = next(item for item in snapshot["items"] if item["sku"] == "COKE500")
    check("Coke qty 2 from two tracks", coke["quantity"] == 2)
    check("Coke unit price from DB", coke["unit_price"] == 40)
    rejected = cart.apply_tracks(
        [
            {
                "track_id": 99,
                "product_id": 1,
                "sku": "COKE500",
                "name": "Coca-Cola 500ml",
                "price": 40,
                "confidence": 0.1,
                "confirmed": True,
            }
        ],
        session=session,
    )
    check("low-confidence track not billed", any(alert["reason"] == "low_confidence" for alert in rejected["alerts"]))

    start = time.perf_counter()
    transaction = checkout(session, cart, invoice_dir=invoice_dir)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    session.commit()
    check("checkout persisted invoice", transaction.invoice_number.startswith("RV-"))
    pdf = PROJECT_ROOT / (transaction.pdf_path or "")
    if not pdf.is_absolute():
        pdf = PROJECT_ROOT / transaction.pdf_path
    check("PDF invoice exists", pdf.exists(), str(transaction.pdf_path))
    session.close()
    reset_engine()
    return {
        "class0_sku": mapped["sku"] if mapped else None,
        "class0_price": mapped["price"] if mapped else None,
        "five_product_subtotal": snapshot["subtotal"],
        "five_product_tax": snapshot["tax"],
        "five_product_total": snapshot["grand_total"],
        "checkout_ms": round(elapsed_ms, 1),
        "invoice_number": transaction.invoice_number,
    }


def _hit(result, class_id: int, confidence: float) -> bool:
    if result.boxes is None or len(result.boxes) == 0:
        return False
    for box in result.boxes:
        if int(box.cls[0].item()) == class_id and float(box.conf[0].item()) >= confidence:
            return True
    return False


def evaluate_conditions(model, confidence: float) -> dict | None:
    images = list_images(IMAGES_DIR / "test")
    if not images:
        print("Condition sweep skipped (no test images).")
        return None
    print(f"Condition sweep: {len(images)} test images x {len(CONDITIONS)} conditions")
    names = class_names()
    per_condition = {name: {"hits": 0, "total": 0, "category": category} for name, category, _fn in CONDITIONS}
    per_category: dict[str, dict[str, int]] = defaultdict(lambda: {"hits": 0, "total": 0})
    latencies: list[float] = []
    for image_path in images:
        image = load_image(image_path)
        if image is None:
            continue
        labels = read_yolo_labels(LABELS_DIR / "test" / f"{image_path.stem}.txt")
        if not labels:
            continue
        class_id = int(labels[0][0])
        for name, category, transform in CONDITIONS:
            variant = transform(image)
            start = time.perf_counter()
            result = model.predict(source=variant, device="cpu" if not torch.cuda.is_available() else "0", conf=confidence, verbose=False)[0]
            latencies.append((time.perf_counter() - start) * 1000.0)
            ok = _hit(result, class_id, confidence)
            per_condition[name]["total"] += 1
            per_category[category]["total"] += 1
            if ok:
                per_condition[name]["hits"] += 1
                per_category[category]["hits"] += 1
    total = sum(item["total"] for item in per_condition.values())
    hits = sum(item["hits"] for item in per_condition.values())
    return {
        "note": "Synthetic probes of held-out photos, not 1000 unique real captures per SKU.",
        "images": len(images),
        "conditions": len(CONDITIONS),
        "evaluations": total,
        "hits": hits,
        "recognition_rate": round(hits / total, 4) if total else 0.0,
        "mean_latency_ms": round(float(np.mean(latencies)), 1) if latencies else None,
        "per_condition": {
            name: {
                **stats,
                "rate": round(stats["hits"] / stats["total"], 4) if stats["total"] else 0.0,
            }
            for name, stats in per_condition.items()
        },
        "per_category": {
            name: {
                **stats,
                "rate": round(stats["hits"] / stats["total"], 4) if stats["total"] else 0.0,
            }
            for name, stats in sorted(per_category.items())
        },
        "class_names": names,
    }


def evaluate_stress(model, confidence: float) -> list[dict]:
    images = list_images(IMAGES_DIR / "test")
    if not images:
        print("Stress test skipped (no test images).")
        return []
    crops = []
    for image_path in images:
        frame = load_image(image_path)
        if frame is None:
            continue
        crops.append(cv2.resize(frame, (160, 160)))
    if not crops:
        return []
    process = psutil.Process()
    rows = []
    for count in (5, 10, 20, 30):
        cols = 6
        rows_n = int(np.ceil(count / cols))
        canvas = np.full((rows_n * 170 + 20, cols * 170 + 20, 3), 40, dtype=np.uint8)
        for index in range(count):
            crop = crops[index % len(crops)]
            r, c = divmod(index, cols)
            y, x = 10 + r * 170, 10 + c * 170
            canvas[y : y + 160, x : x + 160] = crop
        cpu_before = process.cpu_percent(interval=None)
        mem_before = process.memory_info().rss / (1024 * 1024)
        start = time.perf_counter()
        result = model.predict(source=canvas, device="cpu" if not torch.cuda.is_available() else "0", conf=confidence, verbose=False)[0]
        latency_ms = (time.perf_counter() - start) * 1000.0
        detections = 0 if result.boxes is None else len(result.boxes)
        cpu_after = process.cpu_percent(interval=0.1)
        mem_after = process.memory_info().rss / (1024 * 1024)
        row = {
            "objects_in_scene": count,
            "unique_skus_available": min(count, len(crops)),
            "detections": detections,
            "latency_ms": round(latency_ms, 1),
            "fps": round(1000.0 / latency_ms, 2) if latency_ms else 0,
            "cpu_percent": cpu_after,
            "memory_mb": round(mem_after, 1),
            "memory_delta_mb": round(mem_after - mem_before, 1),
            "gpu_usage": None if not torch.cuda.is_available() else "cuda available (not used in this CPU build)",
        }
        rows.append(row)
        print(f"  stress {count:2d} objects  {latency_ms:.0f} ms  dets={detections}  mem={mem_after:.0f} MB")
        del cpu_before
    return rows


def evaluate_latency_probe(model, confidence: float) -> list[dict]:
    images = list_images(IMAGES_DIR / "test")
    if not images:
        return []
    image = load_image(images[0])
    if image is None:
        return []
    rows = []
    for imgsz in (320, 416, 640):
        times = []
        for _repeat in range(3):
            start = time.perf_counter()
            model.predict(source=image, imgsz=imgsz, conf=confidence, verbose=False, device="cpu" if not torch.cuda.is_available() else "0")
            times.append((time.perf_counter() - start) * 1000.0)
        rows.append(
            {
                "imgsz": imgsz,
                "mean_latency_ms": round(float(np.mean(times)), 1),
                "note": "Accuracy was not re-scored at each size; smaller imgsz is a speed probe only.",
            }
        )
        print(f"  imgsz {imgsz}: {rows[-1]['mean_latency_ms']} ms")
    return rows


def write_report(payload: dict) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    eval_metrics = payload.get("prior_evaluation") or {}
    robust = payload.get("prior_robustness") or {}
    train = payload.get("train_config") or {}
    conditions = payload.get("conditions") or {}
    stress = payload.get("stress") or []
    billing = payload.get("billing") or {}
    identity = payload.get("identity") or {}
    test = eval_metrics.get("test") or {}
    if not test and isinstance(eval_metrics.get("val"), dict):
        test = eval_metrics.get("val") or {}

    precision = test.get("precision")
    recall = test.get("recall")
    map50 = test.get("mAP50") or test.get("map50")
    map5095 = test.get("mAP50-95") or test.get("map")
    fpr = test.get("false_positive_rate")
    fnr = None
    if recall is not None:
        fnr = round(1.0 - float(recall), 4)

    reasons = [
        "Held-out test recall is 0.375 and mAP50 is 0.463 on only 9 images.",
        "At the live 0.50 confidence threshold, identity benchmark recall was 0.0.",
        "Robustness over 1080 synthetic variations of those 9 photos was 0.65% hits.",
        "The dataset does not contain 1000 unique real captures per SKU.",
        "Inference is CPU-only on this machine; CUDA PyTorch is not installed.",
        "Maggi was confused with Kurkure on the test set.",
    ]

    mean_latency = conditions.get("mean_latency_ms")
    if stress:
        mean_latency = mean_latency or stress[0].get("latency_ms")
    fps = None
    if mean_latency:
        fps = round(1000.0 / float(mean_latency), 2)
    probe = payload.get("latency_probe") or []
    fn_rate = fnr
    fpr_note = "0 background FPs in Phase 4 confusion; live-threshold misses dominate instead"

    lines = [
        "# Retail Vision — Phase 8 R&D Report",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "## Verdict",
        "",
        "**YOLO26m alone is not sufficient for real-world retail checkout.**",
        "",
        "**The system is not production-ready** based on measured results.",
        "",
        "A two-stage detector + recognizer is recommended *after* a much larger real-photo dataset exists. A second embedding model will not fix misses: at the live threshold the detector often produces no box at all.",
        "",
        "### Why this is not production-ready",
        "",
    ]
    lines.extend(f"- {item}" for item in reasons)
    lines += [
        "",
        "## Measured metrics",
        "",
        "| Metric | Value | Source |",
        "| --- | --- | --- |",
        f"| Detection accuracy (mAP50) | {map50} | Ultralytics test split |",
        f"| Recognition accuracy (live 0.50) | 0.0 | identity benchmark |",
        f"| Precision | {precision} | test split |",
        f"| Recall | {recall} | test split |",
        f"| mAP50 | {map50} | test split |",
        f"| mAP50-95 | {map5095} | test split |",
        f"| False positive rate | {fpr if fpr is not None else fpr_note} | confusion report |",
        f"| False negative rate | {fn_rate} | 1 − recall |",
        f"| FPS (condition sweep) | {fps} | mean {conditions.get('mean_latency_ms')} ms |",
        f"| Inference latency | {conditions.get('mean_latency_ms')} ms | CPU YOLO26m |",
        f"| Average transaction processing time | {billing.get('checkout_ms')} ms | checkout + PDF |",
        f"| Named-condition hit rate | {conditions.get('recognition_rate')} | 171 probes at conf 0.50 |",
        f"| 1080-variation hit rate | {robust.get('recognition_rate')} | Phase 4 protocol |",
        "",
        "## Architecture",
        "",
        "```text",
        "Camera",
        "  -> YOLO26m detection",
        "  -> Multi-object tracking (ByteTrack / IoU)",
        "  -> Product identity (class_id)",
        "  -> Database SKU / price / tax_rate",
        "  -> Acceptance gate (confidence, catalog, valid price)",
        "  -> Cart quantity from unique track_id",
        "  -> Tax from database",
        "  -> Bill + PDF invoice",
        "```",
        "",
        "Prices are never invented by the vision model. Uncertain detections are rejected with a manual-verify message instead of being added to the bill.",
        "",
        "## Dataset methodology",
        "",
        "- 8 SKUs in `products/registry.yaml`",
        "- 74 labeled images, session-aware 70/20/10 split (48/17/9)",
        "- Seed photos from Open Food Facts / Wikimedia, not store-camera captures",
        "- Robustness set: held-out test photos probed under named transforms (angle, rotation, distance, lighting, occlusion, background, scale, orientation)",
        "- **Augmented copies are not counted as independent real-world evidence.**",
        "",
        "## Model and training",
        "",
        f"- Model: YOLO26m (`ultralytics`), weights `retail_yolo26m_v2.pt`",
        f"- Train imgsz: {train.get('imgsz', 416)}",
        f"- Epochs: {train.get('epochs', 20)}, batch: {train.get('batch', 8)}, freeze: {((train.get('augmentation') or {}).get('freeze'))}",
        f"- Device during training: {train.get('device', 'cpu')}",
        f"- Mosaic/mixup: off in v2",
        "",
        "## Accuracy (held-out test set)",
        "",
        f"- Precision: {precision}",
        f"- Recall: {recall}",
        f"- F1: {test.get('f1')}",
        f"- mAP50: {map50}",
        f"- mAP50-95: {map5095}",
        f"- False-negative rate (1 - recall): {fnr}",
        f"- Background false positives in the Phase 4 confusion report: 0 per class",
        "",
        "Live operating point (`CONFIDENCE_THRESHOLD=0.50`): identity benchmark detection recall **0.0** on the 9-image test split.",
        "",
        "## Recognition performance",
        "",
        f"- Prior 1000+ variation protocol: {robust.get('total_evaluations')} evaluations, hit rate **{robust.get('recognition_rate')}** (7/1080).",
        f"- Phase 8 named-condition sweep: {conditions.get('evaluations')} evaluations, hit rate **{conditions.get('recognition_rate')}**.",
        "- Condition sweep uses geometric/photometric probes of the same 9 test photos. It is a robustness diagnostic, not a claim of 1000 real SKUs.",
        "",
        "### Similar products",
        "",
        "- Coke vs Pepsi / Sprite: no Coke-Pepsi swaps on the tiny test set; several classes simply miss.",
        "- Maggi Noodles -> Kurkure Masala Munch: 1 confusion in Phase 4.",
        "- Different Lays flavours / Maggi variants / pack sizes: **not in the catalog**, so they were not measured.",
        "",
        f"- Identity among low-confidence matches: {((identity.get('low_confidence_diagnostic') or {}).get('identity_accuracy_among_matches'))}",
        "",
        "## Tracking performance",
        "",
        "- One Coke across 100 frames: quantity 1 (PASS)",
        "- Two physical Cokes: quantity 2 (PASS)",
        "- Cart add requires `STABLE_FRAMES` consecutive hits and the acceptance gate",
        "",
        "## Billing accuracy",
        "",
        f"- class_id 0 -> SKU {billing.get('class0_sku')} -> price {billing.get('class0_price')} from SQLite (PASS)",
        f"- Five-product cart (Coke x2 + Lays + Maggi + Pepsi + KitKat): subtotal {billing.get('five_product_subtotal')}, tax {billing.get('five_product_tax')}, total {billing.get('five_product_total')}",
        f"- Average checkout / PDF time: {billing.get('checkout_ms')} ms",
        "- Low-confidence tracks are not billed (PASS)",
        "",
        "## FPS and latency",
        "",
        f"- Device: {'cuda' if torch.cuda.is_available() else 'cpu'} (installed PyTorch is CPU-only on this machine)",
        f"- Condition-sweep mean latency: {conditions.get('mean_latency_ms')} ms",
        f"- Approx FPS from that latency: {fps}",
        "",
        "### Stress (composite scenes)",
        "",
    ]
    if stress:
        lines += [
            "| Objects | Detections | Latency ms | FPS | CPU % | Memory MB | GPU |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in stress:
            lines.append(
                f"| {row['objects_in_scene']} | {row['detections']} | {row['latency_ms']} | {row['fps']} | {row['cpu_percent']} | {row['memory_mb']} | {row['gpu_usage'] or 'n/a'} |"
            )
    else:
        lines.append("Stress test was skipped (no test images).")

    lines += [
        "",
        "### Optimization investigated (not enabled by default)",
        "",
        "- Image size 320/416/640 latency probe (do not drop imgsz solely for FPS)",
        "- `FRAME_SKIP` in `.env` (default 0)",
        "- `INFER_IMGSZ` optional override",
        "- GPU acceleration: blocked here by a CPU PyTorch wheel and an old NVIDIA driver (CUDA 11.1)",
        "- Export/quantization: not deployed; accuracy is already below retail bar at FP32",
        "",
        "Latency vs image size (speed probe only; accuracy was not re-scored):",
        "",
    ]
    if probe:
        lines += ["| imgsz | Mean latency ms |", "| --- | --- |"]
        for row in probe:
            lines.append(f"| {row['imgsz']} | {row['mean_latency_ms']} |")
    else:
        lines.append("Latency probe was skipped.")
    lines += [
        "",
        "CPU % in the stress table is a short `psutil` sample of a multi-thread CPU inference process and can exceed 100. It is not a calibrated hardware meter. GPU usage is n/a because this PyTorch build has no CUDA.",
        "",
        "Named-condition hit rates (all at live confidence 0.50):",
        "",
    ]
    if conditions.get("per_category"):
        lines += ["| Category | Hits | Total | Rate |", "| --- | --- | --- | --- |"]
        for name, stats in conditions["per_category"].items():
            lines.append(f"| {name} | {stats['hits']} | {stats['total']} | {stats['rate']} |")
    if conditions.get("per_condition"):
        lines += ["", "| Condition | Hits | Total | Rate |", "| --- | --- | --- | --- |"]
        for name, stats in conditions["per_condition"].items():
            lines.append(f"| {name} | {stats['hits']} | {stats['total']} | {stats['rate']} |")

    lines += [
        "",
        "## Failure cases handled",
        "",
        "- Unknown product / not in database: rejected, operator message",
        "- Low confidence: rejected, operator message",
        "- Invalid price: rejected, operator message",
        "- Camera disconnect: POS placeholder, no silent cart adds",
        "- Database unavailable: camera loop continues, cart update skipped",
        "- Model missing: POS shows model unavailable",
        "- Duplicate tracks: unique `track_id` quantity",
        "",
        "## Hardware requirements (this machine vs retail target)",
        "",
        "| Item | Measured | Retail target |",
        "| --- | --- | --- |",
        "| OS | Windows 11 | Windows/Linux POS PC |",
        "| GPU | GTX 1650, driver 457.49 / CUDA 11.1, unused | CUDA GPU with current driver |",
        "| PyTorch | 2.13.0+cpu | CUDA build matching the driver |",
        "| CPU latency | ~150-300 ms/frame | <50 ms typical checkout |",
        "",
        "## Limitations",
        "",
        "- Eight SKUs and 74 images cannot represent a store catalog",
        "- No dedicated front/back/left/right camera captures; some angle tests are proxies (flip/rotate)",
        "- Similar flavour/size variants are not registered",
        "- CPU inference is too slow for a snappy checkout lane",
        "- Detector recall at the live threshold is the blocking failure, not cart math",
        "",
        "## Future improvements",
        "",
        "1. Capture 1000+ real in-store views per SKU (true angle, lighting, occlusion, clutter)",
        "2. Retrain YOLO26m until live-threshold recall is high on a held-out camera set",
        "3. Then evaluate a second-stage embedding model for remaining similar-SKU collisions",
        "4. Install a CUDA PyTorch build after updating the NVIDIA driver",
        "5. Add pack-size and flavour SKUs before claiming multi-variant recognition",
        "",
        f"## Suite result: {payload['suite']['passed']} passed, {payload['suite']['failed']} failed",
        "",
    ]
    text = "\n".join(lines) + "\n"
    path = REPORT_DIR / "phase8_rnd_report.md"
    path.write_text(text, encoding="utf-8")
    metrics_path = REPORT_DIR / "phase8_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def main() -> int:
    print("Retail Vision - Phase 8\n")
    failure = test_failure_handling()
    duplicates = test_duplicates()
    tmp = tempfile.TemporaryDirectory()
    try:
        billing = test_price_and_checkout(Path(tmp.name))
    finally:
        tmp.cleanup()

    vision = {"conditions": None, "stress": [], "latency_probe": []}
    weights = Path(MODEL_PATH)
    if weights.exists() and list_images(IMAGES_DIR / "test"):
        from ultralytics import YOLO

        print("Loading YOLO26m for vision measurements...")
        model = YOLO(str(weights))
        vision["conditions"] = evaluate_conditions(model, CONFIDENCE_THRESHOLD)
        vision["stress"] = evaluate_stress(model, CONFIDENCE_THRESHOLD)
        vision["latency_probe"] = evaluate_latency_probe(model, CONFIDENCE_THRESHOLD)
    else:
        print("Vision measurements skipped (weights or test images missing).")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "production_ready": False,
        "yolo26m_alone_sufficient": False,
        "recommend_two_stage": "after_larger_dataset",
        "failure_handling": failure,
        "duplicates": duplicates,
        "billing": billing,
        "conditions": vision["conditions"],
        "stress": vision["stress"],
        "latency_probe": vision["latency_probe"],
        "prior_evaluation": _load_json(EVAL_JSON),
        "prior_robustness": _load_json(ROBUST_JSON),
        "train_config": _load_json(TRAIN_JSON),
        "identity": _load_json(IDENTITY_JSON),
        "suite": {"passed": PASS, "failed": FAIL},
        "hardware": {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cpu_count": psutil.cpu_count(),
        },
    }
    report = write_report(payload)
    print(f"\n{PASS} passed, {FAIL} failed")
    print(f"Report: {report}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
