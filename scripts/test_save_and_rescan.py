"""End-to-end: save a product, then immediately recognize it (DINOv2 + vector search).

Requires the project's DATABASE_URL / EMBEDDING_BACKEND from .env.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _make_product_jpeg() -> bytes:
    rng = np.random.default_rng()
    base = (int(rng.integers(20, 80)), int(rng.integers(40, 120)), int(rng.integers(160, 255)))
    accent = (int(rng.integers(0, 80)), int(rng.integers(0, 80)), int(rng.integers(180, 255)))
    img = np.full((480, 360, 3), base, dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (310, 430), accent, -1)
    cv2.rectangle(img, (70, 90), (290, 160), (255, 255, 255), -1)
    label = f"G{int(rng.integers(100, 999))}"
    cv2.putText(img, label, (95, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (20, 20, 20), 3)
    cv2.circle(img, (180, 280), 55, (0, int(rng.integers(150, 255)), 255), -1)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def main() -> int:
    from fastapi.testclient import TestClient
    from sqlalchemy import select, func, update

    from backend.app.main import app
    from backend.app.database import get_session_factory, init_db
    from backend.app.models.product import Product
    from backend.app.models.product_embedding import ProductEmbedding
    from backend.app.models.product_image import ProductImage

    init_db()
    jpeg = _make_product_jpeg()
    print(f"[TEST] jpeg bytes={len(jpeg)}")
    unique_name = f"E2E Glow Lotion {np.random.randint(10000, 99999)}"

    # Prior identical E2E registrations make margin checks AMBIGUOUS.
    session = get_session_factory()()
    try:
        session.execute(
            update(Product)
            .where(Product.name.ilike("E2E Glow Lotion%"))
            .values(is_active=False)
        )
        session.commit()
        print("[TEST] Deactivated prior E2E Glow Lotion products")
    finally:
        session.close()

    with TestClient(app) as client:
        # --- TEST 1: Save ---
        r = client.post(
            "/products/register",
            data={
                "name": unique_name,
                "price": "250",
                "brand": "Nivea",
                "category": "Personal Care",
                "variant": "Natural Glow",
                "weight": "200ml",
                "tax_rate": "18",
                "add_to_cart": "false",
                "force_create": "true",
            },
            files={"image": ("e2e.jpg", jpeg, "image/jpeg")},
        )
        print(f"[TEST] REGISTER status={r.status_code}")
        print(f"[TEST] REGISTER body={r.text[:1500]}")
        if r.status_code != 201:
            print("[FAIL] Product save failed")
            return 1

        payload = r.json()
        product_id = int(payload.get("id") or payload.get("product_id") or 0)
        if not product_id and isinstance(payload.get("product"), dict):
            product_id = int(payload["product"].get("id") or 0)
        print(f"[TEST] product_id={product_id}")

        session = get_session_factory()()
        try:
            images = session.scalar(
                select(func.count()).select_from(ProductImage).where(ProductImage.product_id == product_id)
            )
            embeddings = session.scalar(
                select(func.count())
                .select_from(ProductEmbedding)
                .where(ProductEmbedding.product_id == product_id)
            )
            print(f"[TEST] DB images={images} embeddings={embeddings}")
            if int(images or 0) < 1 or int(embeddings or 0) < 1:
                print("[FAIL] Missing image or embedding rows after save")
                return 1
        finally:
            session.close()

        # --- TEST 2: Immediate rescan of same image ---
        s = client.post(
            "/products/scan",
            data={"use_camera": "false"},
            files={"image": ("scan.jpg", jpeg, "image/jpeg")},
        )
        print(f"[TEST] SCAN status={s.status_code}")
        print(f"[TEST] SCAN body={s.text[:2000]}")
        if s.status_code != 200:
            print("[FAIL] Scan HTTP error")
            return 1
        scan = s.json()
        status = str(scan.get("status") or "")
        found = bool(scan.get("found"))
        matched_id = None
        product = scan.get("product") or {}
        if isinstance(product, dict):
            matched_id = product.get("product_id") or product.get("id")
        print(f"[TEST] scan status={status} found={found} matched_id={matched_id}")
        if not found or status != "found" or int(matched_id or 0) != product_id:
            print("[FAIL] Saved product was not recognized on immediate rescan")
            return 1

        print("[PASS] Save + immediate rescan both succeeded")
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
