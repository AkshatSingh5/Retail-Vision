"""Recognition reliability: similar products, unknown, margin, new SKU."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ["EMBEDDING_BACKEND"] = "color"
os.environ["PRODUCT_MATCH_THRESHOLD"] = "0.78"
os.environ["PRODUCT_MATCH_MARGIN"] = "0.05"
os.environ["DUPLICATE_MATCH_THRESHOLD"] = "0.99"
os.environ["MIN_IMAGE_SIDE"] = "32"
os.environ["BLUR_VARIANCE_MIN"] = "5"
os.environ["RECOGNITION_DEBUG"] = "false"

import cv2
import numpy as np
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.database import init_db, reset_engine
from backend.app.main import app
from backend.app.services.cart_service import reset_cart
from scripts.scan_test_helpers import mock_yolo_single_product
from vision.recognition.gallery import crop_embedding, cosine_similarity, get_gallery, refresh_embedding_version

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


def _pack(label: str, base_bgr: tuple[int, int, int], accent: tuple[int, int, int], size=(160, 220)) -> bytes:
    w, h = size
    frame = np.full((h, w, 3), base_bgr, dtype=np.uint8)
    cv2.rectangle(frame, (12, 12), (w - 12, h - 12), accent, 4)
    cv2.rectangle(frame, (24, 40), (w - 24, 100), accent, -1)
    cv2.putText(frame, label[:8], (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    seed = sum(ord(c) for c in label)
    rng = np.random.default_rng(seed)
    for _ in range(12):
        x, y = int(rng.integers(20, w - 20)), int(rng.integers(110, h - 20))
        color = tuple(int(c) for c in rng.integers(0, 255, size=3))
        if seed % 2 == 0:
            cv2.circle(frame, (x, y), int(rng.integers(4, 12)), color, -1)
        else:
            cv2.rectangle(frame, (x, y), (x + 14, y + 10), color, -1)
    stripe = tuple(int((seed * (i + 3)) % 255) for i in range(3))
    cv2.rectangle(frame, (0, h - 28), (w, h), stripe, -1)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    return encoded.tobytes()


def _noise_jpeg() -> bytes:
    frame = np.random.default_rng(99).integers(0, 255, (180, 140, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    return encoded.tobytes()


def main() -> int:
    print("Retail Vision - recognition reliability\n")
    refresh_embedding_version()
    tmp = tempfile.TemporaryDirectory()
    try:
        db_path = Path(tmp.name) / "recog.db"
        reset_engine()
        reset_cart()
        init_db(f"sqlite:///{db_path.as_posix()}")
        get_gallery().invalidate()

        cola = _pack("COLA", (20, 20, 180), (0, 0, 255))
        pepsi = _pack("PEPSI", (160, 40, 20), (255, 180, 0))
        lays = _pack("LAYS", (0, 200, 255), (0, 140, 255))
        kurkure = _pack("KURKURE", (0, 120, 255), (40, 40, 220))
        unknown = _noise_jpeg()

        with TestClient(app) as client, mock_yolo_single_product():
            regs = []
            for name, sku, price, blob in [
                ("Coca Cola 500ml", "TESTCOLA", "40", cola),
                ("Pepsi 500ml", "TESTPEPSI", "40", pepsi),
                ("Lays Classic", "TESTLAYS", "20", lays),
                ("Kurkure Masala", "TESTKURK", "20", kurkure),
            ]:
                res = client.post(
                    "/products/register",
                    data={
                        "name": name,
                        "sku": sku,
                        "price": price,
                        "tax_rate": "18",
                        "add_to_cart": "false",
                        "force_create": "true",
                    },
                    files={"image": (f"{sku}.jpg", blob, "image/jpeg")},
                )
                check(f"register {sku}", res.status_code == 201, res.text)
                regs.append((sku, blob, res.json() if res.status_code == 201 else {}))

            cola_vec = crop_embedding(cv2.imdecode(np.frombuffer(cola, np.uint8), cv2.IMREAD_COLOR))
            pepsi_vec = crop_embedding(cv2.imdecode(np.frombuffer(pepsi, np.uint8), cv2.IMREAD_COLOR))
            sim_cp = cosine_similarity(cola_vec, pepsi_vec)
            check("cola vs pepsi embedding not identical", sim_cp < 0.97, f"sim={sim_cp:.4f}")

            for sku, blob, meta in regs:
                scan = client.post(
                    "/products/scan",
                    data={"use_camera": "false"},
                    files={"image": ("frame.jpg", blob, "image/jpeg")},
                )
                body = scan.json() if scan.status_code == 200 else {}
                check(f"scan {sku} 200", scan.status_code == 200, scan.text)
                check(f"scan {sku} matched", body.get("found") is True and body.get("status") == "found", str(body))
                product = body.get("product") or {}
                check(f"scan {sku} correct id", product.get("product_id") == meta.get("id"), str(product))
                expected_price = int(float(str(meta.get("price", 0))))
                check(f"scan {sku} correct price", product.get("price") == expected_price, str(product))

            for _ in range(3):
                scan = client.post(
                    "/products/scan",
                    data={"use_camera": "false"},
                    files={"image": ("cola.jpg", cola, "image/jpeg")},
                )
                body = scan.json()
                check("cola consistent", (body.get("product") or {}).get("sku") == "TESTCOLA", str(body))

            miss = client.post(
                "/products/scan",
                data={"use_camera": "false"},
                files={"image": ("unk.jpg", unknown, "image/jpeg")},
            )
            miss_body = miss.json() if miss.status_code == 200 else {}
            check("unknown not found", miss_body.get("found") is False, str(miss_body))
            check(
                "unknown status not_found or ambiguous",
                miss_body.get("status") in {"not_found", "unknown", "ambiguous", "no_product"},
                str(miss_body),
            )

            new_blob = _pack("NEWITEM", (80, 200, 80), (0, 255, 100))
            created = client.post(
                "/products/register",
                data={
                    "name": "New Green Item",
                    "sku": "TESTNEW",
                    "price": "55",
                    "add_to_cart": "false",
                    "force_create": "true",
                },
                files={"image": ("new.jpg", new_blob, "image/jpeg")},
            )
            check("register new", created.status_code == 201, created.text)
            new_id = (created.json() or {}).get("id")
            rescan = client.post(
                "/products/scan",
                data={"use_camera": "false"},
                files={"image": ("new2.jpg", new_blob, "image/jpeg")},
            )
            again = rescan.json() if rescan.status_code == 200 else {}
            check("new product recognized", again.get("found") is True, str(again))
            check("new product correct id", (again.get("product") or {}).get("product_id") == new_id, str(again))

            add = client.post("/cart/items", json={"product_id": new_id, "quantity": 1})
            cart = client.get("/cart").json()
            line = next((item for item in cart["items"] if item["product_id"] == new_id), None)
            check(
                "bill line uses recognition product_id",
                line is not None and float(line.get("unit_price", 0)) == 55,
                str(line),
            )
            check("add to cart ok", add.status_code == 200)

            from vision.recognition.gallery import GalleryMatch, ProductGallery

            g = ProductGallery(threshold=0.50, margin=0.20)
            g._ready = True
            base = crop_embedding(cv2.imdecode(np.frombuffer(cola, np.uint8), cv2.IMREAD_COLOR))
            twin = list(base)
            if twin:
                twin[0] = twin[0] + 1e-4
            meta_a = GalleryMatch(1, "A", "Product A", 10, 18, 0.0)
            meta_b = GalleryMatch(2, "B", "Product B", 10, 18, 0.0)
            g._entries = [(base, meta_a), (twin, meta_b)]
            decision = g.match_decision(
                cv2.imdecode(np.frombuffer(cola, np.uint8), cv2.IMREAD_COLOR),
                threshold=0.50,
                margin=0.20,
            )
            check("margin rejects ambiguous pair", decision is not None and decision.accepted is False, str(decision))
            check("ambiguous reason", decision is not None and decision.reason == "ambiguous_margin", str(decision))

    finally:
        reset_cart()
        reset_engine()
        tmp.cleanup()

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
