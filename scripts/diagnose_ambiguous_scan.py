"""Diagnose ambiguous recognition on live registered products."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.config import ROOT_DIR, SCAN_MATCH_MARGIN, SCAN_MATCH_THRESHOLD
from backend.app.database import get_session_factory, init_db
from backend.app.services.scan_service import recognize_frame
from sqlalchemy import select
from backend.app.models.product import Product
from vision.recognition.gallery import (
    EMBEDDING_VERSION,
    cosine_similarity,
    crop_embedding,
    embedding_from_json,
    get_gallery,
)


def main() -> int:
    init_db()
    gallery = get_gallery()
    gallery.invalidate()
    gallery.reload()
    session = get_session_factory()()
    products = [
        p
        for p in session.scalars(select(Product).where(Product.is_active.is_(True))).all()
        if p.image_path
    ]
    print("CONFIG", SCAN_MATCH_THRESHOLD, SCAN_MATCH_MARGIN, EMBEDDING_VERSION)
    print("products_with_images", [(p.id, p.name) for p in products])

    for product in products:
        path = ROOT_DIR / product.image_path
        img = cv2.imread(str(path))
        print(f"\n======== {product.id} {product.name} ========")
        print("path", path, "exists", path.exists(), "shape", None if img is None else img.shape)
        if img is None:
            continue
        a = crop_embedding(img)
        b = crop_embedding(img)
        print("self_similarity", round(cosine_similarity(a, b), 6), "dim", len(a))
        stored = embedding_from_json(product.recognition_embedding)
        if stored:
            print("fresh_vs_stored", round(cosine_similarity(a, stored), 6))
        else:
            print("stored_embedding", "missing_or_stale_version")

        ok, buf = cv2.imencode(".jpg", img)
        result = recognize_frame(session, buf.tobytes())
        print(
            "scan_status",
            result.get("status"),
            "found",
            result.get("found"),
            "product",
            (result.get("product") or {}).get("name"),
        )
        recog = result.get("recognition") or {}
        print("failing_stage", recog.get("failing_stage"))
        print("best_match", recog.get("best_match") or result.get("best_match"))
        print("second_match", recog.get("second_match") or result.get("second_match"))
        print("top_matches", recog.get("top_matches"))
        print(
            "threshold/margin/actual",
            recog.get("threshold"),
            recog.get("required_margin"),
            recog.get("actual_margin"),
        )

    session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
