"""Phase 6 product database, API, and price-mapping tests.

Run from the project root:

    python scripts/test_products.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient

from backend.app.database import get_session_factory, init_db, reset_engine
from backend.app.main import app
from backend.app.services.seed import seed_products_from_registry
from vision.recognition.identity import ProductIdentifier
from vision.recognition.store import DatabaseProductStore

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        extra = f" — {detail}" if detail else ""
        print(f"  FAIL  {name}{extra}")


def _setup_db(path: Path) -> None:
    reset_engine()
    init_db(f"sqlite:///{path.as_posix()}")
    session = get_session_factory()()
    try:
        created = seed_products_from_registry(session)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
    if created != 8:
        raise RuntimeError(f"expected 8 seeded products, got {created}")


def test_api_and_mapping(client: TestClient) -> None:
    print("Product API")
    listing = client.get("/products")
    check("GET /products → 200", listing.status_code == 200, str(listing.status_code))
    products = listing.json()
    check("8 active products", len(products) == 8, str(len(products)))

    coke = client.get("/products/1")
    check("GET /products/1 → 200", coke.status_code == 200, str(coke.status_code))
    body = coke.json()
    check("id is 1", body["id"] == 1)
    check("sku COKE500", body["sku"] == "COKE500")
    check("price from database is 40", body["price"] == 40, str(body.get("price")))
    check("tax_rate 18", body["tax_rate"] == 18, str(body.get("tax_rate")))
    check("yolo_class_id 0", body["yolo_class_id"] == 0)

    by_sku = client.get("/products/sku/COKE500")
    check("GET /products/sku/COKE500", by_sku.status_code == 200 and by_sku.json()["id"] == 1)

    mapping = client.get("/products/class/0")
    check("GET /products/class/0 → 200", mapping.status_code == 200, str(mapping.status_code))
    payload = mapping.json()
    print("  mapping:", json.dumps(payload, separators=(",", ":")))
    check(
        "class 0 → product_id 1 / COKE500 / ₹40",
        payload
        == {
            "product_id": 1,
            "sku": "COKE500",
            "name": "Coca-Cola 500ml",
            "price": 40,
            "tax_rate": 18,
        },
        str(payload),
    )

    created = client.post(
        "/products",
        json={
            "sku": "TESTSKU",
            "name": "Test Item",
            "brand": "Test",
            "category": "Other",
            "price": 99,
            "tax_rate": 18,
        },
    )
    check("POST /products → 201", created.status_code == 201, str(created.status_code))
    new_id = created.json()["id"]
    updated = client.put(f"/products/{new_id}", json={"price": 120})
    check("PUT price 120 from API not detector", updated.status_code == 200 and updated.json()["price"] == 120)
    deleted = client.delete(f"/products/{new_id}")
    check("DELETE /products/{id} → 204", deleted.status_code == 204, str(deleted.status_code))
    missing = client.get(f"/products/{new_id}")
    check("deleted product 404", missing.status_code == 404)


def test_detection_price_lookup() -> None:
    print("Detection → database price")
    store = DatabaseProductStore(ttl_seconds=60)
    identifier = ProductIdentifier(store=store, use_database=True)
    detected = identifier.identify({"class_id": 0, "confidence": 0.94, "bbox": [100, 150, 300, 500]})
    check("class_id 0 stays 0", detected["class_id"] == 0)
    check("product_id 1 from database", detected["product_id"] == 1, str(detected["product_id"]))
    check("SKU COKE500 from database", detected["sku"] == "COKE500")
    check("name Coca-Cola 500ml", detected["product_name"] == "Coca-Cola 500ml")
    check("price 40 from database", detected["price"] == 40, str(detected["price"]))
    check("yaml fallback is not used for price", detected["price"] != 101)

    mapping = identifier.price_mapping_for_class(0)
    check("price mapping contract", mapping == {
        "product_id": 1,
        "sku": "COKE500",
        "name": "Coca-Cola 500ml",
        "price": 40,
        "tax_rate": 18,
    }, str(mapping))

    lays = identifier.identify({"class_id": 1, "confidence": 0.9, "bbox": [10, 10, 40, 40]})
    check("Lays price 20 from database", lays["price"] == 20 and lays["sku"] == "LAYSCLASSIC")


def test_no_hardcoded_prices() -> None:
    print("No hardcoded prices in detection code")
    path = ROOT_DIR / "vision" / "detection" / "yolo_detector.py"
    text = path.read_text(encoding="utf-8")
    check("detector has no product price branches", "price =" not in text and "COKE500" not in text)
    identity_src = (ROOT_DIR / "vision" / "recognition" / "identity.py").read_text(encoding="utf-8")
    check("identifier does not assign rupee literals", "price = 40" not in identity_src)


def main() -> int:
    print("Retail Vision — Phase 6 tests\n")
    tmp = tempfile.TemporaryDirectory()
    try:
        db_path = Path(tmp.name) / "phase6.db"
        _setup_db(db_path)
        with TestClient(app) as client:
            test_api_and_mapping(client)
        test_detection_price_lookup()
        test_no_hardcoded_prices()
    finally:
        reset_engine()
        tmp.cleanup()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
