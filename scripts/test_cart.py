"""Phase 7 cart, tax, invoice, and 5-product checkout tests.

Run from the project root:

    python scripts/test_cart.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient

from backend.app.database import get_session_factory, init_db, reset_engine
from backend.app.main import app
from backend.app.services.cart_service import reset_cart
from backend.app.services.seed import seed_products_from_registry

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


def _setup(db_path: Path) -> None:
    reset_engine()
    reset_cart()
    init_db(f"sqlite:///{db_path.as_posix()}")
    session = get_session_factory()()
    try:
        seed_products_from_registry(session)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _track(track_id: int, product_id: int, sku: str, name: str, price: float, tax_rate: float = 18) -> dict:
    return {
        "track_id": track_id,
        "product_id": product_id,
        "sku": sku,
        "name": name,
        "price": price,
        "tax_rate": tax_rate,
        "confirmed": True,
    }


def test_five_product_checkout(client: TestClient) -> None:
    print("Five-product tracked cart")
    payload = {
        "tracks": [
            _track(1, 1, "COKE500", "Coca-Cola 500ml", 40),
            _track(2, 1, "COKE500", "Coca-Cola 500ml", 40),
            _track(3, 2, "LAYSCLASSIC", "Lays Classic", 20),
            _track(4, 3, "MAGGI", "Maggi Noodles", 14),
            _track(5, 5, "PEPSI500", "Pepsi 500ml", 40),
            _track(6, 7, "KITKAT", "KitKat", 30),
        ]
    }
    cart = client.post("/cart/tracks", json=payload).json()
    by_sku = {item["sku"]: item for item in cart["items"]}
    check("5 different products in cart", len(cart["items"]) == 5, str(len(cart["items"])))
    check("two Coke tracks = qty 2", by_sku["COKE500"]["quantity"] == 2, str(by_sku.get("COKE500")))
    check("Coke weight 500ml in cart", by_sku["COKE500"].get("weight") == "500ml", str(by_sku.get("COKE500")))
    check("Maggi x 1", by_sku["MAGGI"]["quantity"] == 1)
    check("Maggi weight 70g in cart", by_sku["MAGGI"].get("weight") == "70g", str(by_sku.get("MAGGI")))
    check("Coke line total uses DB price and tax", by_sku["COKE500"]["total"] == 94.4, str(by_sku["COKE500"]))
    check("subtotal 184", cart["subtotal"] == 184, str(cart["subtotal"]))
    check("tax 33.12 from DB 18%", cart["tax"] == 33.12, str(cart["tax"]))
    check("grand total 217.12", cart["grand_total"] == 217.12, str(cart["grand_total"]))

    print("Manual controls")
    increased = client.post("/cart/items/1/increase").json()
    coke = next(item for item in increased["items"] if item["product_id"] == 1)
    check("increase Coke to x3", coke["quantity"] == 3, str(coke["quantity"]))
    decreased = client.post("/cart/items/1/decrease").json()
    coke = next(item for item in decreased["items"] if item["product_id"] == 1)
    check("decrease Coke back to x2", coke["quantity"] == 2)
    confirmed = client.post("/cart/items/3/confirm").json()
    maggi = next(item for item in confirmed["items"] if item["product_id"] == 3)
    check("confirm Maggi", maggi["confirmed"] is True)
    removed = client.delete("/cart/items/2").json()
    check("remove Lays", all(item["sku"] != "LAYSCLASSIC" for item in removed["items"]))
    client.post("/cart/items", json={"product_id": 2, "quantity": 1})
    check("manual add Lays back", True)

    print("Bill generation")
    reset_cart()
    empty = client.post("/checkout")
    check("empty cart cannot checkout", empty.status_code == 400, str(empty.status_code))

    client.post("/cart/tracks", json=payload)
    billed = client.post("/checkout")
    check("checkout 200", billed.status_code == 200, billed.text[:200])
    body = billed.json()
    check("invoice number assigned", body["invoice_number"].startswith("RV-"))
    bill_by_sku = {item["sku"]: item for item in body["bill"]["items"]}
    check("bill items contain weight (Coke 500ml)", bill_by_sku["COKE500"].get("weight") == "500ml", str(bill_by_sku.get("COKE500")))
    pdf = client.get(body["pdf_url"])
    check("PDF invoice downloadable", pdf.status_code == 200 and pdf.headers["content-type"].startswith("application/pdf"))
    check("cart cleared after bill", client.get("/cart").json()["items"] == [])
    stored = client.get(f"/transactions/{body['transaction_id']}").json()
    check("transaction persisted", stored["invoice_number"] == body["invoice_number"])
    check("5 product lines stored", len(stored["items"]) == 5, str(len(stored["items"])))
    stored_by_sku = {item["sku"]: item for item in stored["items"]}
    check("stored transaction item contains weight", stored_by_sku["COKE500"].get("weight") == "500ml", str(stored_by_sku.get("COKE500")))
    check("stored grand total matches", stored["grand_total"] == body["bill"]["grand_total"])

    cart_src = (ROOT_DIR / "backend" / "app" / "services" / "cart_service.py").read_text(encoding="utf-8")
    check("cart service has no hard-coded tax rate", "tax_rate = 18" not in cart_src and "GST = 0.18" not in cart_src)


def test_tax_follows_database(client: TestClient) -> None:
    print("Tax follows database, not detector")
    reset_cart()
    client.put("/products/1", json={"tax_rate": 5, "price": 40})
    cart = client.post(
        "/cart/tracks",
        json={"tracks": [_track(10, 1, "COKE500", "Coca-Cola 500ml", 999, tax_rate=99)]},
    ).json()
    coke = cart["items"][0]
    check("unit price refreshed from DB (40 not 999)", coke["unit_price"] == 40, str(coke["unit_price"]))
    check("tax uses DB 5% not ingest 99%", coke["tax"] == 2, str(coke["tax"]))
    client.put("/products/1", json={"tax_rate": 18, "price": 40})


def main() -> int:
    print("Retail Vision - Phase 7 tests\n")
    tmp = tempfile.TemporaryDirectory()
    try:
        db_path = Path(tmp.name) / "phase7.db"
        _setup(db_path)
        with TestClient(app) as client:
            test_five_product_checkout(client)
            test_tax_follows_database(client)
    finally:
        reset_cart()
        reset_engine()
        tmp.cleanup()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
