"""One-shot camera / upload product recognition for POS scanning.

Architecture:
  Camera → YOLO (detect/crop only) → quality check → DINOv2 → pgvector
  → product-level score → threshold + margin → found / not_found / ambiguous

Identity NEVER comes from YOLO class labels, OCR, or barcodes.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
from sqlalchemy.orm import Session

from backend.app.config import (
    DETECT_CONFIDENCE,
    MAX_UPLOAD_BYTES,
    MODEL_PATH,
    PRODUCT_MATCH_MARGIN,
    PRODUCT_MATCH_THRESHOLD,
    RECOGNITION_DEBUG,
    SCAN_DEBUG_CROPS,
    STORAGE_DIR,
)
from backend.app.models.product import Product
from backend.app.models.scan_log import ScanLog
from backend.app.schemas.product import public_image_url, serialize_money
from backend.app.services.product_service import ProductNotFoundError, get_product
from vision.detection.product_filter import filter_scan_detections, is_retail_trained_model, scan_crop_priority
from vision.recognition.gallery import EMBEDDING_VERSION, crop_embedding, get_gallery
from vision.recognition.preprocess import pad_bbox
from vision.recognition.quality import assess_image_quality
from vision.recognition.vector_bridge import current_model_name
from vision.tracking.pipeline import _crop_box

logger = logging.getLogger(__name__)

PENDING_TTL_SECONDS = 30 * 60

_pending_lock = threading.Lock()
_pending_scans: dict[str, dict[str, Any]] = {}
_detector_lock = threading.Lock()
_detector = None


class ScanError(ValueError):
    """User-facing scan / image validation error."""


@dataclass
class ScanProductHit:
    product_id: int
    name: str
    sku: str
    price: int | float
    tax_rate: int | float
    image_url: str | None
    confidence: float
    barcode: str | None = None
    category: str | None = None
    brand: str | None = None
    variant: str | None = None
    weight: str | None = None


def _purge_expired_pending(now: float | None = None) -> None:
    moment = time.monotonic() if now is None else now
    expired = [key for key, value in _pending_scans.items() if moment - float(value["created"]) > PENDING_TTL_SECONDS]
    for key in expired:
        _pending_scans.pop(key, None)


def store_pending_scan(image_bytes: bytes, crop_bytes: bytes | None = None) -> str:
    scan_id = uuid4().hex
    with _pending_lock:
        _purge_expired_pending()
        _pending_scans[scan_id] = {
            "image": image_bytes,
            "crop": crop_bytes or image_bytes,
            "created": time.monotonic(),
        }
    return scan_id


def pop_pending_scan(scan_id: str | None) -> bytes | None:
    bundle = pop_pending_scan_bundle(scan_id)
    if bundle is None:
        return None
    return bundle.get("crop") or bundle.get("image")


def pop_pending_scan_bundle(scan_id: str | None) -> dict[str, bytes] | None:
    if not scan_id:
        return None
    with _pending_lock:
        _purge_expired_pending()
        payload = _pending_scans.pop(str(scan_id), None)
    if payload is None:
        return None
    return {
        "image": payload.get("image") or b"",
        "crop": payload.get("crop") or payload.get("image") or b"",
    }


def peek_pending_scan(scan_id: str | None) -> bytes | None:
    if not scan_id:
        return None
    with _pending_lock:
        _purge_expired_pending()
        payload = _pending_scans.get(str(scan_id))
    if payload is None:
        return None
    return payload.get("crop") or payload.get("image")


def validate_image_bytes(image_bytes: bytes, content_type: str | None = None) -> bytes:
    if not image_bytes:
        raise ScanError("Invalid image.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise ScanError("Image too large.")
    if content_type:
        mime = content_type.split(";")[0].strip().lower()
        if mime and mime not in {"application/octet-stream", "binary/octet-stream"}:
            from backend.app.config import ALLOWED_IMAGE_MIME

            if mime not in ALLOWED_IMAGE_MIME:
                raise ScanError("Invalid image.")
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    decoded = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ScanError("Invalid image.")
    return image_bytes


def _decode(image_bytes: bytes) -> np.ndarray:
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ScanError("Invalid image.")
    return frame


def _encode_jpeg(frame: np.ndarray, quality: int = 90) -> bytes:
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ScanError("Image processing failed.")
    return encoded.tobytes()


def _get_detector():
    global _detector
    try:
        from backend.app.services.camera_hub import get_camera_hub

        hub = get_camera_hub()
        if hub.pipeline is not None and getattr(hub.pipeline, "detector", None) is not None:
            return hub.pipeline.detector
    except Exception:
        pass

    with _detector_lock:
        if _detector is not None:
            return _detector
        from vision.detection.yolo_detector import get_yolo_detector

        print(f"[YOLO] Loading model from {MODEL_PATH}")
        _detector = get_yolo_detector()
        print(f"[YOLO] Model loaded successfully ({len(_detector.names)} classes)")
        return _detector


def _product_image_url(product: Product) -> str | None:
    if product.image_url:
        return str(product.image_url)
    return public_image_url(int(product.id), product.image_path)


def _hit_from_product(session: Session, product_id: int, confidence: float) -> ScanProductHit | None:
    try:
        product = get_product(session, int(product_id))
    except ProductNotFoundError:
        return None
    if not product.is_active:
        return None
    price = serialize_money(product.price)
    tax = serialize_money(product.tax_rate)
    if price is None or tax is None:
        return None
    return ScanProductHit(
        product_id=int(product.id),
        name=str(product.name),
        sku=str(product.sku),
        price=price,
        tax_rate=tax,
        image_url=_product_image_url(product),
        confidence=float(confidence),
        barcode=product.barcode,
        category=product.category,
        brand=product.brand,
        variant=product.variant,
        weight=product.weight,
    )


def _bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def _merge_overlapping(
    candidates: list[tuple[np.ndarray, dict]],
    *,
    iou_threshold: float = 0.55,
) -> list[tuple[np.ndarray, dict]]:
    """Collapse highly-overlapping boxes so one physical product is not counted twice."""
    if len(candidates) <= 1:
        return candidates
    kept: list[tuple[np.ndarray, dict]] = []
    for crop, meta in candidates:
        bbox = meta.get("bbox") or []
        merged = False
        for index, (_crop, existing) in enumerate(kept):
            if len(bbox) == 4 and len(existing.get("bbox") or []) == 4:
                if _bbox_iou(bbox, existing["bbox"]) >= iou_threshold:
                    if scan_crop_priority(meta, None) > scan_crop_priority(existing, None):
                        kept[index] = (crop, meta)
                    merged = True
                    break
        if not merged:
            kept.append((crop, meta))
    return kept


def _save_debug_crops(frame: np.ndarray, crop: np.ndarray | None, meta: dict, scan_token: str):
    if not (SCAN_DEBUG_CROPS or RECOGNITION_DEBUG):
        return None
    try:
        folder = STORAGE_DIR / "debug" / "scans" / scan_token
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / "original_camera_image.jpg"), frame)
        if crop is not None:
            cv2.imwrite(str(folder / "product_crop.jpg"), crop)
        bbox = meta.get("bbox") or []
        annotated = frame.copy()
        if len(bbox) == 4:
            x1, y1, x2, y2 = (int(round(v)) for v in bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(str(folder / "yolo_detection.jpg"), annotated)
        return folder
    except Exception as extra:
        logger.debug("Scan debug crop save failed: %s", extra)
        return None


def _detect_crops(frame: np.ndarray) -> tuple[list[tuple[np.ndarray, dict]], dict]:
    yolo_info: dict[str, Any] = {
        "model_path": MODEL_PATH,
        "model_loaded": False,
        "classes": {},
        "detection_count": 0,
        "retail_trained": False,
        "error": None,
    }
    try:
        detector = _get_detector()
        yolo_info["model_path"] = str(getattr(detector, "model_path", MODEL_PATH))
        yolo_info["model_loaded"] = True
        yolo_info["classes"] = {int(k): str(v) for k, v in dict(detector.names).items()}
        detections, latency_ms = detector.detect(frame, confidence_threshold=DETECT_CONFIDENCE)
        yolo_info["detection_count"] = len(detections)
        yolo_info["latency_ms"] = round(latency_ms, 1)
        yolo_info["raw_detections"] = [
            {
                "class_id": int(item["class_id"]),
                "class_name": detector.names.get(int(item["class_id"]), "object"),
                "confidence": float(item["confidence"]),
                "bbox": list(item["bbox"]),
            }
            for item in detections[:10]
        ]
        detections = filter_scan_detections(
            detections,
            detector.names,
            log=True,
            frame_shape=frame.shape,
        )
        retail = is_retail_trained_model(detector.names)
        yolo_info["retail_trained"] = retail
    except Exception as extra:
        logger.warning("Scan detector unavailable: %s", extra)
        yolo_info["error"] = str(extra)
        return [], yolo_info

    scored: list[tuple[float, np.ndarray, dict]] = []
    for item in detections:
        padded = pad_bbox(item["bbox"], frame.shape, pad_ratio=0.08)
        crop = _crop_box(frame, padded)
        if crop is None:
            continue
        meta = dict(item)
        meta["bbox"] = padded
        meta["retail_trained"] = retail
        meta["source"] = "yolo"
        meta["class_name"] = detector.names.get(int(item["class_id"]), "object")
        scored.append((scan_crop_priority(item, frame.shape), crop, meta))
    scored.sort(key=lambda row: row[0], reverse=True)
    candidates = [(crop, meta) for _score, crop, meta in scored]
    return _merge_overlapping(candidates), yolo_info


def _write_scan_log(
    session: Session,
    *,
    scan_token: str,
    detected_count: int,
    ranked: list,
    decision: Any,
    selected_product_id: int | None,
    confidence: float | None,
    threshold: float,
    margin: float | None,
    result_state: str,
) -> None:
    try:
        candidates = [
            {
                "product_id": int(row.product_id),
                "name": str(row.name),
                "score": round(float(row.score), 4),
            }
            for row in ranked[:10]
        ]
        session.add(
            ScanLog(
                scan_token=scan_token,
                detected_count=int(detected_count),
                embedding_model=current_model_name(),
                candidate_json=json.dumps(candidates),
                selected_product_id=selected_product_id,
                confidence=confidence,
                threshold=threshold,
                margin=margin,
                result_state=result_state,
            )
        )
        session.flush()
    except Exception as extra:
        logger.debug("Scan log write failed: %s", extra)


def _matched_response(hit: ScanProductHit, *, similarity: float, margin: float, detection_confidence: float | None, match_threshold: float, detections_count: int, bbox: list[float] | None = None) -> dict[str, Any]:
    return {
        "found": True,
        "success": True,
        "status": "found",
        "message": "Product found!",
        "confidence": round(similarity, 4),
        "threshold": match_threshold,
        "detections": detections_count,
        "items": [
            {
                "product_id": hit.product_id,
                "product_name": hit.name,
                "sku": hit.sku,
                "confidence": round(similarity, 4),
                "bbox": bbox,
                "status": "identified",
            }
        ],
        "scan_id": None,
        "preview_url": None,
        "reason": None,
        "match_type": "visual",
        "product": {
            "product_id": hit.product_id,
            "id": hit.product_id,
            "name": hit.name,
            "sku": hit.sku,
            "price": hit.price,
            "tax_rate": hit.tax_rate,
            "image_url": hit.image_url,
            "confidence": round(similarity, 4),
            "barcode": hit.barcode,
            "category": hit.category,
            "brand": hit.brand,
            "variant": hit.variant,
            "weight": hit.weight,
        },
        "recognition": {
            "detection_confidence": round(detection_confidence, 4) if detection_confidence is not None else None,
            "similarity": round(similarity, 4),
            "margin": round(margin, 4),
            "threshold": match_threshold,
            "required_margin": PRODUCT_MATCH_MARGIN,
            "embedding_model": current_model_name(),
        },
    }


def _status_response(
    *,
    status: str,
    message: str,
    reason: str,
    confidence: float,
    match_threshold: float,
    detections_count: int,
    scan_id: str | None,
    recognition: dict | None = None,
    candidates: list[dict] | None = None,
    bbox: list[float] | None = None,
) -> dict[str, Any]:
    item_status = "unknown" if status in {"not_found", "unknown"} else status
    items = None
    if detections_count:
        items = [
            {
                "product_id": None,
                "product_name": "Unknown Product" if item_status == "unknown" else None,
                "sku": None,
                "confidence": round(confidence, 4) if confidence else 0.0,
                "bbox": bbox,
                "status": item_status,
            }
        ]
    return {
        "found": False,
        "success": False,
        "status": status,
        "message": message,
        "reason": reason,
        "confidence": round(confidence, 4) if confidence else 0.0,
        "threshold": match_threshold,
        "detections": detections_count,
        "items": items,
        "scan_id": scan_id,
        "preview_url": f"/products/scan/{scan_id}/preview" if scan_id else None,
        "match_type": "visual",
        "product": None,
        "recognition": recognition,
        "candidates": candidates,
    }


def recognize_frame(
    session: Session,
    image_bytes: bytes,
    *,
    threshold: float | None = None,
    margin: float | None = None,
) -> dict[str, Any]:
    """Run one-shot visual recognition. Prices always come from the database."""
    match_threshold = PRODUCT_MATCH_THRESHOLD if threshold is None else float(threshold)
    match_margin = PRODUCT_MATCH_MARGIN if margin is None else float(margin)
    frame = _decode(image_bytes)
    scan_token = uuid4().hex[:12]

    frame_quality = assess_image_quality(frame)
    if not frame_quality.ok and frame_quality.reason in {"invalid_image", "too_small"}:
        return _status_response(
            status="low_image_quality",
            message=frame_quality.message,
            reason=frame_quality.reason,
            confidence=0.0,
            match_threshold=match_threshold,
            detections_count=0,
            scan_id=store_pending_scan(image_bytes),
        )

    try:
        yolo_candidates, yolo_info = _detect_crops(frame)
    except Exception as extra:
        logger.exception("YOLO failed")
        return _status_response(
            status="processing_error",
            message="Recognition service unavailable.",
            reason="model_unavailable",
            confidence=0.0,
            match_threshold=match_threshold,
            detections_count=0,
            scan_id=store_pending_scan(image_bytes),
            recognition={"error": str(extra)} if RECOGNITION_DEBUG else None,
        )

    detections_count = len(yolo_candidates)

    if detections_count == 0:
        # COCO often misses branded packs. When frame quality is OK, use the
        # centered scanning-area crop so DINOv2/pgvector can still run.
        if frame_quality.ok:
            h, w = frame.shape[:2]
            y0, y1 = int(h * 0.12), int(h * 0.88)
            x0, x1 = int(w * 0.18), int(w * 0.82)
            center_crop = frame[y0:y1, x0:x1].copy()
            crop_q = assess_image_quality(center_crop) if center_crop.size else None
            # Reject near-empty / low-contrast center crops (avoid false matches).
            usable = (
                center_crop.size > 0
                and min(center_crop.shape[:2]) >= 64
                and crop_q is not None
                and crop_q.ok
            )
            if usable:
                print(
                    "[YOLO] 0 detections → using center scan-area crop for recognition "
                    f"({center_crop.shape[1]}x{center_crop.shape[0]})"
                )
                yolo_candidates = [
                    (
                        center_crop,
                        {
                            "bbox": [float(x0), float(y0), float(x1), float(y1)],
                            "confidence": 0.0,
                            "class_id": -1,
                            "class_name": "scan_area",
                            "source": "scan_frame_fallback",
                            "retail_trained": False,
                        },
                    )
                ]
                detections_count = 1
            else:
                print(
                    "[YOLO] 0 detections and center crop unusable "
                    f"(quality={getattr(crop_q, 'reason', None)}) → no_product"
                )

        if detections_count == 0:
            scan_id = store_pending_scan(image_bytes)
            _write_scan_log(
                session,
                scan_token=scan_token,
                detected_count=0,
                ranked=[],
                decision=None,
                selected_product_id=None,
                confidence=None,
                threshold=match_threshold,
                margin=None,
                result_state="no_product",
            )
            return _status_response(
                status="no_product",
                message="No product detected.\nPlease place only one product inside the scanning area.",
                reason="no_product_detected",
                confidence=0.0,
                match_threshold=match_threshold,
                detections_count=0,
                scan_id=scan_id,
            )

    if detections_count > 1:
        scan_id = store_pending_scan(image_bytes)
        _write_scan_log(
            session,
            scan_token=scan_token,
            detected_count=detections_count,
            ranked=[],
            decision=None,
            selected_product_id=None,
            confidence=None,
            threshold=match_threshold,
            margin=None,
            result_state="multiple_products",
        )
        return _status_response(
            status="multiple_products",
            message="MULTIPLE PRODUCTS DETECTED\nPlease place only one product inside the scanning area.",
            reason="multiple_products",
            confidence=0.0,
            match_threshold=match_threshold,
            detections_count=detections_count,
            scan_id=scan_id,
        )

    crop, crop_meta = yolo_candidates[0]
    crop_quality = assess_image_quality(crop)
    if not crop_quality.ok:
        scan_id = store_pending_scan(image_bytes, _encode_jpeg(crop))
        _write_scan_log(
            session,
            scan_token=scan_token,
            detected_count=1,
            ranked=[],
            decision=None,
            selected_product_id=None,
            confidence=None,
            threshold=match_threshold,
            margin=None,
            result_state="low_image_quality",
        )
        return _status_response(
            status="low_image_quality",
            message=crop_quality.message or "IMAGE QUALITY TOO LOW\nPlease hold the product steady and scan again.",
            reason=crop_quality.reason or "low_image_quality",
            confidence=0.0,
            match_threshold=match_threshold,
            detections_count=1,
            scan_id=scan_id,
        )

    _save_debug_crops(frame, crop, crop_meta, scan_token)
    detection_confidence = float(crop_meta.get("confidence") or 0.0) or None

    try:
        query_vector = crop_embedding(crop)
    except Exception as extra:
        logger.exception("Embedding failed")
        return _status_response(
            status="processing_error",
            message="Recognition service unavailable.",
            reason="embedding_failure",
            confidence=0.0,
            match_threshold=match_threshold,
            detections_count=1,
            scan_id=store_pending_scan(image_bytes, _encode_jpeg(crop)),
            recognition={"error": str(extra)} if RECOGNITION_DEBUG else None,
        )

    gallery = get_gallery()
    try:
        ranked = gallery.rank_products(crop, top_k=10, session=session, query=query_vector)
        decision = gallery.match_decision(
            crop,
            threshold=match_threshold,
            margin=match_margin,
            session=session,
            query=query_vector,
        )
    except Exception as extra:
        logger.exception("Vector search failed")
        return _status_response(
            status="processing_error",
            message="Recognition service unavailable.",
            reason="vector_search_failure",
            confidence=0.0,
            match_threshold=match_threshold,
            detections_count=1,
            scan_id=store_pending_scan(image_bytes, _encode_jpeg(crop)),
            recognition={"error": str(extra)} if RECOGNITION_DEBUG else None,
        )

    best_score = float(decision.score) if decision else (float(ranked[0].score) if ranked else 0.0)
    second_score = float(decision.second_score) if decision else (float(ranked[1].score) if len(ranked) > 1 else 0.0)
    gap = float(decision.margin) if decision else (best_score - second_score)

    print(
        f"[SCAN] model={current_model_name()} best={best_score:.4f} "
        f"second={second_score:.4f} margin={gap:.4f} "
        f"threshold={match_threshold:.4f} required_margin={match_margin:.4f} "
        f"decision={getattr(decision, 'reason', None)} "
        f"query_dim={len(query_vector)} crop_source={crop_meta.get('source')}"
    )
    print("[SCAN] TOP 5 MATCHES:")
    if not ranked:
        print("  (none)")
    for index, row in enumerate(ranked[:5], start=1):
        print(
            f"  {index}. Product ID: {row.product_id}\n"
            f"     Product: {row.name}\n"
            f"     Similarity: {float(row.score):.4f}"
        )

    recognition_meta = {
        "detection_confidence": round(detection_confidence, 4) if detection_confidence is not None else None,
        "best_similarity": round(best_score, 4),
        "second_similarity": round(second_score, 4),
        "margin": round(gap, 4),
        "actual_margin": round(gap, 4),
        "threshold": match_threshold,
        "required_margin": match_margin,
        "embedding_model": current_model_name() or EMBEDDING_VERSION,
        "embedding_dimensions": len(query_vector),
        "normalized": True,
        "crop_source": crop_meta.get("source"),
        "top_matches": [
            {
                "product_id": int(row.product_id),
                "name": str(row.name),
                "sku": str(row.sku),
                "similarity": round(float(row.score), 4),
            }
            for row in ranked[:10]
        ],
    }
    if ranked:
        recognition_meta["best_match"] = {
            "product_id": int(ranked[0].product_id),
            "name": str(ranked[0].name),
            "similarity": round(float(ranked[0].score), 4),
        }
    if len(ranked) > 1:
        recognition_meta["second_match"] = {
            "product_id": int(ranked[1].product_id),
            "name": str(ranked[1].name),
            "similarity": round(float(ranked[1].score), 4),
        }

    if decision is not None and decision.accepted:
        db_hit = _hit_from_product(session, int(decision.product_id), float(decision.score))
        if db_hit is not None:
            _write_scan_log(
                session,
                scan_token=scan_token,
                detected_count=1,
                ranked=ranked,
                decision=decision,
                selected_product_id=db_hit.product_id,
                confidence=float(decision.score),
                threshold=match_threshold,
                margin=float(decision.margin),
                result_state="found",
            )
            response = _matched_response(
                db_hit,
                similarity=float(decision.score),
                margin=float(decision.margin),
                detection_confidence=detection_confidence,
                match_threshold=match_threshold,
                detections_count=1,
                bbox=list(crop_meta.get("bbox") or []),
            )
            if RECOGNITION_DEBUG:
                response["recognition"] = {**(response.get("recognition") or {}), **recognition_meta}
            return response

    pending = _encode_jpeg(crop)
    scan_id = store_pending_scan(image_bytes, pending)

    if decision is not None and decision.reason == "ambiguous_margin":
        _write_scan_log(
            session,
            scan_token=scan_token,
            detected_count=1,
            ranked=ranked,
            decision=decision,
            selected_product_id=None,
            confidence=best_score,
            threshold=match_threshold,
            margin=gap,
            result_state="ambiguous",
        )
        response = _status_response(
            status="ambiguous",
            message="Multiple visually similar products found.\nPlease scan again.",
            reason="ambiguous_match",
            confidence=best_score,
            match_threshold=match_threshold,
            detections_count=1,
            scan_id=scan_id,
            bbox=list(crop_meta.get("bbox") or []),
            recognition=recognition_meta if RECOGNITION_DEBUG else {
                "best_similarity": round(best_score, 4),
                "second_similarity": round(second_score, 4),
                "margin": round(gap, 4),
                "threshold": match_threshold,
                "required_margin": match_margin,
            },
            candidates=recognition_meta.get("top_matches"),
        )
        if RECOGNITION_DEBUG:
            response["best_match"] = recognition_meta.get("best_match")
            response["second_match"] = recognition_meta.get("second_match")
        return response

    _write_scan_log(
        session,
        scan_token=scan_token,
        detected_count=1,
        ranked=ranked,
        decision=decision,
        selected_product_id=None,
        confidence=best_score,
        threshold=match_threshold,
        margin=gap,
        result_state="not_found",
    )
    response = _status_response(
        status="unknown",
        message="Unknown Product\nWe couldn't confidently identify this product.",
        reason="low_confidence" if ranked else "not_found",
        confidence=best_score,
        match_threshold=match_threshold,
        detections_count=1,
        scan_id=scan_id,
        bbox=list(crop_meta.get("bbox") or []),
        recognition=recognition_meta if RECOGNITION_DEBUG else {
            "best_similarity": round(best_score, 4),
            "second_similarity": round(second_score, 4),
            "margin": round(gap, 4),
            "threshold": match_threshold,
            "required_margin": match_margin,
        },
    )
    if RECOGNITION_DEBUG:
        response["best_match"] = recognition_meta.get("best_match")
        response["second_match"] = recognition_meta.get("second_match")
    return response
