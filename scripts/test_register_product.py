"""New product registration: DB, image, cart bind, no YOLO retrain."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ["EMBEDDING_BACKEND"] = "color"
os.environ["PRODUCT_MATCH_THRESHOLD"] = "0.78"
os.environ["DUPLICATE_MATCH_THRESHOLD"] = "0.97"
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
from vision.recognition.gallery import crop_embedding, get_gallery, refresh_embedding_version

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


def _jpeg_bytes() -> bytes:
    frame = np.full((96, 72, 3), (18, 42, 200), dtype=np.uint8)
    cv2.rectangle(frame, (8, 8), (64, 88), (40, 90, 240), -1)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    return encoded.tobytes()


def main() -> int:
    print("Retail Vision - product registration\n")
    refresh_embedding_version()
    tmp = tempfile.TemporaryDirectory()
    try:
        db_path = Path(tmp.name) / "register.db"
        reset_engine()
        reset_cart()
        init_db(f"sqlite:///{db_path.as_posix()}")
        session = get_session_factory()()
        seed_products_from_registry(session)
        session.commit()
        session.close()
        get_gallery().invalidate()
        image = _jpeg_bytes()
        with TestClient(app) as client:
            created = client.post(
                "/products/register",
                data={
                    "name": "Coca-Cola 750ml",
                    "sku": "COKE750",
                    "price": "55",
                    "tax_rate": "18",
                    "brand": "Coca-Cola",
                    "category": "Beverage",
                    "variant": "750ml",
                    "weight": "750g",
                    "description": "Chilled cola 750ml",
                    "add_to_cart": "true",
                    "force_create": "true",
                },
                files={"image": ("crop.jpg", image, "image/jpeg")},
            )
            check("register 201", created.status_code == 201, created.text)
            body = created.json() if created.status_code == 201 else {}
            check("sku COKE750", body.get("sku") == "COKE750")
            check("price 55 from database", body.get("price") == 55)
            check("cart_added true", body.get("cart_added") is True)
            check("success message", "added successfully" in str(body.get("message", "")).lower())
            product_id = body.get("id")
            check("description stored", body.get("description") == "Chilled cola 750ml")
            check("variant stored", body.get("variant") == "750ml")
            check("image_url is public route", body.get("image_url") == f"/products/{product_id}/image")
            nested = body.get("product") or {}
            check("nested product sku", nested.get("sku") == "COKE750")
            check("nested product image_url", nested.get("image_url") == f"/products/{product_id}/image")

            duplicate = client.post(
                "/products/register",
                data={"name": "Coke 750", "sku": "COKE750", "price": "55", "add_to_cart": "false", "force_create": "true"},
                files={"image": ("crop.jpg", image, "image/jpeg")},
            )
            check("duplicate SKU 409", duplicate.status_code == 409, str(duplicate.status_code))
            check(
                "duplicate message",
                "already exists" in str(duplicate.json().get("detail", "")).lower(),
            )

            missing_name = client.post(
                "/products/register",
                data={"name": " ", "sku": "EMPTYNAME", "price": "10", "add_to_cart": "false", "force_create": "true"},
                files={"image": ("crop.jpg", image, "image/jpeg")},
            )
            check("empty name 400", missing_name.status_code == 400)

            cart = client.get("/cart").json()
            names = [item["name"] for item in cart["items"]]
            check("Coke 750ml is in the current cart", "Coca-Cola 750ml" in names, str(names))
            line = next((item for item in cart["items"] if item["sku"] == "COKE750"), None)
            check("quantity is 1", line is not None and line["quantity"] == 1, str(line))

            listed = client.get("/products").json()
            check("catalog includes COKE750", any(item["sku"] == "COKE750" for item in listed))

            images = client.get(f"/products/{product_id}/images")
            check("product has at least one image", images.status_code == 200 and len(images.json()) >= 1)

            extra = client.post(
                f"/products/{product_id}/images",
                data={"image_type": "side"},
                files={"image": ("left.jpg", image, "image/jpeg")},
            )
            check("additional image 201", extra.status_code == 201, extra.text)

            deleted = client.delete(f"/products/{product_id}")
            check("soft delete 204", deleted.status_code == 204)
            gone = client.get(f"/products/{product_id}")
            check("inactive product hidden", gone.status_code == 404)
            still = client.get("/products").json()
            check("COKE750 not in active list", all(item["sku"] != "COKE750" for item in still))

        vector = crop_embedding(cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR))
        check("embedding generated", len(vector) > 10)
    finally:
        reset_cart()
        reset_engine()
        tmp.cleanup()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
