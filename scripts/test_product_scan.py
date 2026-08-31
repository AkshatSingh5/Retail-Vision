"""End-to-end camera product scan → register → rescan → bill tests."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Fast deterministic embeddings for CI / local tests (production uses DINOv2).
os.environ["EMBEDDING_BACKEND"] = "color"
os.environ["PRODUCT_MATCH_THRESHOLD"] = "0.78"
os.environ["PRODUCT_MATCH_MARGIN"] = "0.05"
os.environ["DUPLICATE_MATCH_THRESHOLD"] = "0.97"
os.environ["RECOGNITION_DEBUG"] = "false"
os.environ["MIN_IMAGE_SIDE"] = "32"
os.environ["BLUR_VARIANCE_MIN"] = "5"

import cv2
import numpy as np
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.database import get_session_factory, init_db, reset_engine
from backend.app.main import app
from backend.app.services.cart_service import reset_cart
from backend.app.services.seed import seed_products_from_registry
from scripts.scan_test_helpers import mock_yolo_multiple_products, mock_yolo_no_product, mock_yolo_single_product
from vision.recognition.gallery import get_gallery, refresh_embedding_version

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


def _jpeg(color=(18, 42, 200), size=(120, 90), pattern="cola") -> bytes:
    frame = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    if pattern == "cola":
        cv2.rectangle(frame, (10, 10), (size[0] - 10, size[1] - 10), (40, 90, 240), -1)
        cv2.putText(frame, "COLA", (18, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.circle(frame, (size[0] // 2, 25), 12, (0, 0, 255), -1)
    elif pattern == "snack":
        for y in range(0, size[1], 8):
            cv2.line(frame, (0, y), (size[0], y), (0, 220, 80), 2)
        cv2.rectangle(frame, (20, 20), (size[0] - 20, size[1] - 20), (255, 255, 0), 3)
        cv2.putText(frame, "CHIP", (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    else:
        noise = np.random.default_rng(42).integers(0, 255, frame.shape, dtype=np.uint8)
        frame = noise
        cv2.putText(frame, "MISC", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    return encoded.tobytes()


def _other_jpeg() -> bytes:
    return _jpeg(color=(20, 180, 40), size=(160, 120), pattern="misc")


def main() -> int:
    print("Retail Vision - product scan workflow\n")
    refresh_embedding_version()
    tmp = tempfile.TemporaryDirectory()
    try:
        db_path = Path(tmp.name) / "scan.db"
        reset_engine()
        reset_cart()
        init_db(f"sqlite:///{db_path.as_posix()}")
        session = get_session_factory()()
        seed_products_from_registry(session)
        session.commit()
        session.close()
        get_gallery().invalidate()

        known = _jpeg()
        unknown = _other_jpeg()

        with TestClient(app) as client:
            with mock_yolo_single_product():
                created = client.post(
                    "/products/register",
                    data={
                        "name": "Scan Cola 500ml",
                        "sku": "SCANCOLA500",
                        "price": "40",
                        "tax_rate": "18",
                        "category": "Beverage",
                        "add_to_cart": "false",
                        "force_create": "true",
                    },
                    files={"image": ("cola.jpg", known, "image/jpeg")},
                )
                check("register known product", created.status_code == 201, created.text)
                product = created.json() if created.status_code == 201 else {}
                product_id = product.get("id")
                check("image_path stored (not blob)", bool(product.get("image_path")))
                check("primary image endpoint", client.get(f"/products/{product_id}/image").status_code == 200)

                scan = client.post(
                    "/products/scan",
                    data={"use_camera": "false"},
                    files={"image": ("frame.jpg", known, "image/jpeg")},
                )
                check("scan known returns 200", scan.status_code == 200, scan.text)
                body = scan.json() if scan.status_code == 200 else {}
                check("scan found true", body.get("found") is True, str(body))
                check("scan status found", body.get("status") == "found", str(body))
                matched = body.get("product") or {}
                check("correct sku", matched.get("sku") == "SCANCOLA500", str(matched))
                check("price from database", matched.get("price") == 40, str(matched))
                check("confidence present", float(matched.get("confidence") or 0) > 0)
                check("public image_url", str(matched.get("image_url") or "").startswith("/products/"))
                check("match_type visual", body.get("match_type") == "visual", str(body))

                add1 = client.post("/cart/items", json={"product_id": product_id, "quantity": 1})
                add2 = client.post("/cart/items", json={"product_id": product_id, "quantity": 1})
                check("add to cart twice", add1.status_code == 200 and add2.status_code == 200)
                cart = client.get("/cart").json()
                line = next((item for item in cart["items"] if item["product_id"] == product_id), None)
                check("duplicate scan increases qty", line is not None and line["quantity"] == 2, str(line))

                missing = client.post(
                    "/products/scan",
                    data={"use_camera": "false"},
                    files={"image": ("unknown.jpg", unknown, "image/jpeg")},
                )
                check("unknown scan 200", missing.status_code == 200, missing.text)
                miss = missing.json() if missing.status_code == 200 else {}
                check("unknown not found", miss.get("found") is False, str(miss))
                check("scan_id issued", bool(miss.get("scan_id")), str(miss))
                preview = miss.get("preview_url")
                check("preview available", bool(preview) and client.get(preview).status_code == 200)

                registered = client.post(
                    "/products/register",
                    data={
                        "name": "Green Snack 50g",
                        "sku": "GREENSNACK50",
                        "price": "20",
                        "tax_rate": "18",
                        "category": "Snacks",
                        "scan_id": miss.get("scan_id"),
                        "add_to_cart": "false",
                        "force_create": "true",
                    },
                )
                check("register from scan_id", registered.status_code == 201, registered.text)
                new_product = registered.json() if registered.status_code == 201 else {}
                new_id = new_product.get("id")
                check("success flag", new_product.get("success") is True)
                check("new product has image path", bool(new_product.get("image_path")))

                rescan = client.post(
                    "/products/scan",
                    data={"use_camera": "false"},
                    files={"image": ("again.jpg", unknown, "image/jpeg")},
                )
                check("rescan 200", rescan.status_code == 200, rescan.text)
                again = rescan.json() if rescan.status_code == 200 else {}
                check("newly added product recognized", again.get("found") is True, str(again))
                check(
                    "rescan sku matches",
                    (again.get("product") or {}).get("sku") == "GREENSNACK50",
                    str(again),
                )

                client.post("/cart/items", json={"product_id": new_id, "quantity": 1})
                cart = client.get("/cart").json()
                check("multiple products in cart", cart.get("item_count", 0) >= 3, str(cart))
                checkout = client.post("/checkout")
                check("generate bill", checkout.status_code == 200, checkout.text)
                bill = checkout.json() if checkout.status_code == 200 else {}
                check("invoice number present", bool(bill.get("invoice_number")))
                check("pdf url present", bool(bill.get("pdf_url")))

                bill2 = client.post("/bills/generate")
                check("bills/generate empty cart rejected", bill2.status_code == 400)

            with mock_yolo_multiple_products():
                multi = client.post(
                    "/products/scan",
                    data={"use_camera": "false"},
                    files={"image": ("multi.jpg", known, "image/jpeg")},
                )
                multi_body = multi.json() if multi.status_code == 200 else {}
                check("multiple products status", multi_body.get("status") == "multiple_products", str(multi_body))

            with mock_yolo_no_product():
                none = client.post(
                    "/products/scan",
                    data={"use_camera": "false"},
                    files={"image": ("none.jpg", known, "image/jpeg")},
                )
                none_body = none.json() if none.status_code == 200 else {}
                # YOLO may return 0 on COCO for some packs; center scan-area fallback
                # must still allow DINOv2 matching of a registered product.
                check(
                    "zero-yolo fallback recognizes known product",
                    none_body.get("status") == "found" and none_body.get("found") is True,
                    str(none_body),
                )

            huge = b"x" * (9 * 1024 * 1024)
            too_big = client.post(
                "/products/scan",
                data={"use_camera": "false"},
                files={"image": ("big.bin", huge, "image/jpeg")},
            )
            check("image too large rejected", too_big.status_code == 400)

            bad = client.post(
                "/products/scan",
                data={"use_camera": "false"},
                files={"image": ("bad.jpg", b"not-an-image", "image/jpeg")},
            )
            check("invalid image rejected", bad.status_code == 400)

            with mock_yolo_single_product():
                dup = client.post(
                    "/products/register",
                    data={
                        "name": "Dup",
                        "sku": "SCANCOLA500",
                        "price": "10",
                        "add_to_cart": "false",
                        "force_create": "true",
                    },
                    files={"image": ("dup.jpg", known, "image/jpeg")},
                )
                check("duplicate SKU rejected", dup.status_code == 409)

            no_cam = client.post("/products/scan", data={"use_camera": "true"})
            check("camera required when offline", no_cam.status_code == 400, no_cam.text)

    finally:
        reset_cart()
        reset_engine()
        tmp.cleanup()

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
