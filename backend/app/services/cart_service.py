from __future__ import annotations

from decimal import Decimal
from threading import Lock
from uuid import uuid4

from sqlalchemy.orm import Session

from backend.app.config import DISCOUNT_PERCENT
from backend.app.services.acceptance import evaluate_for_cart
from backend.app.services.product_service import ProductNotFoundError, get_product
from backend.app.utils.money import line_subtotal, line_tax, line_total, money, money_json


class CartError(ValueError):
    pass


class CartLine:
    def __init__(
        self,
        product_id: int,
        sku: str,
        name: str,
        unit_price: Decimal,
        tax_rate: Decimal,
        weight: str | None = None,
    ) -> None:
        self.product_id = int(product_id)
        self.sku = sku
        self.name = name
        self.weight = weight
        self.unit_price = money(unit_price)
        self.tax_rate = money(tax_rate)
        self.track_ids: set[int] = set()
        self.manual_adjust = 0
        self.confirmed = False

    @property
    def quantity(self) -> int:
        return max(0, len(self.track_ids) + self.manual_adjust)

    def amounts(self) -> tuple[Decimal, Decimal, Decimal]:
        qty = self.quantity
        return (
            line_subtotal(self.unit_price, qty),
            line_tax(self.unit_price, qty, self.tax_rate),
            line_total(self.unit_price, qty, self.tax_rate),
        )

    def as_item(self) -> dict:
        _subtotal, tax, total = self.amounts()
        return {
            "product_id": self.product_id,
            "sku": self.sku,
            "name": self.name,
            "weight": self.weight,
            "quantity": self.quantity,
            "unit_price": money_json(self.unit_price),
            "tax": money_json(tax),
            "total": money_json(total),
            "tax_rate": money_json(self.tax_rate),
            "confirmed": self.confirmed,
            "track_ids": sorted(self.track_ids),
        }


class Cart:
    """In-memory POS cart for the current transaction."""

    def __init__(self, discount_percent: Decimal | None = None) -> None:
        self._lock = Lock()
        self.lines: dict[int, CartLine] = {}
        self.ignored_track_ids: set[int] = set()
        self.discount_percent = money(discount_percent if discount_percent is not None else DISCOUNT_PERCENT)
        self.transaction_id = uuid4().hex[:12]
        self.last_invoice_number: str | None = None
        self.alerts: list[dict] = []

    def snapshot(self, session: Session | None = None) -> dict:
        with self._lock:
            if session is not None:
                self._refresh_from_db(session)
            items = [line.as_item() for line in self._ordered_lines() if line.quantity > 0]
            subtotal = money(0)
            tax = money(0)
            for line in self._ordered_lines():
                if line.quantity <= 0:
                    continue
                line_sub, line_gst, _total = line.amounts()
                subtotal += line_sub
                tax += line_gst
            discount = money(subtotal * self.discount_percent / Decimal("100"))
            grand_total = money(subtotal + tax - discount)
            if grand_total < 0:
                grand_total = money(0)
            return {
                "transaction_id": self.transaction_id,
                "items": items,
                "subtotal": money_json(subtotal),
                "tax": money_json(tax),
                "discount": money_json(discount),
                "discount_percent": money_json(self.discount_percent),
                "grand_total": money_json(grand_total),
                "item_count": sum(item["quantity"] for item in items),
                "alerts": list(self.alerts[-12:]),
            }

    def apply_tracks(self, tracks: list[dict], session: Session | None = None) -> dict:
        with self._lock:
            alerts: list[dict] = []
            for track in tracks:
                track_id = int(track["track_id"])
                if track_id in self.ignored_track_ids:
                    continue
                in_database = None
                if session is not None:
                    try:
                        get_product(session, int(track["product_id"]))
                        in_database = True
                    except (ProductNotFoundError, TypeError, ValueError):
                        in_database = False
                decision = evaluate_for_cart(track, in_database=in_database)
                if not decision["accepted"]:
                    if not decision.get("silent"):
                        alerts.append(
                            {
                                "track_id": track_id,
                                "reason": decision["reason"],
                                "message": decision["message"],
                                "sku": track.get("sku"),
                                "confidence": track.get("confidence"),
                            }
                        )
                    continue
                product_id = int(track["product_id"])
                line = self._line_for_product(product_id, track, session)
                if line is None:
                    alerts.append(
                        {
                            "track_id": track_id,
                            "reason": "not_in_database",
                            "message": "UNKNOWN PRODUCT\nProduct not found in database.\nPlease verify manually.",
                            "sku": track.get("sku"),
                            "confidence": track.get("confidence"),
                        }
                    )
                    continue
                if line.unit_price <= 0:
                    alerts.append(
                        {
                            "track_id": track_id,
                            "reason": "invalid_price",
                            "message": "UNKNOWN PRODUCT\nInvalid product price.\nPlease verify manually.",
                            "sku": track.get("sku"),
                            "confidence": track.get("confidence"),
                        }
                    )
                    continue
                line.track_ids.add(track_id)
            self.alerts = alerts
        return self.snapshot(session)

    def add_from_registration(self, session: Session, product_id: int, track_id: int | None = None) -> dict:
        with self._lock:
            product = get_product(session, product_id)
            if money(product.price) < 0:
                raise CartError("Please enter a valid price.")
            line = self._ensure_line_from_product(product)
            line.confirmed = True
            if track_id is not None:
                line.track_ids.add(int(track_id))
            else:
                line.manual_adjust += 1
        return self.snapshot(session)

    def add_product(self, session: Session, product_id: int, quantity: int = 1) -> dict:
        if quantity < 1:
            raise CartError("Quantity must be at least 1.")
        with self._lock:
            product = get_product(session, product_id)
            if money(product.price) <= 0:
                raise CartError("Invalid product price. Please verify manually.")
            line = self._ensure_line_from_product(product)
            line.manual_adjust += int(quantity)
            line.confirmed = True
        return self.snapshot(session)

    def increase(self, product_id: int) -> dict:
        with self._lock:
            line = self._require_line(product_id)
            line.manual_adjust += 1
            line.confirmed = True
        return self.snapshot()

    def decrease(self, product_id: int) -> dict:
        with self._lock:
            line = self._require_line(product_id)
            if line.quantity <= 0:
                raise CartError("Quantity is already zero.")
            if line.manual_adjust > 0:
                line.manual_adjust -= 1
            elif line.track_ids:
                removed = max(line.track_ids)
                line.track_ids.remove(removed)
                self.ignored_track_ids.add(removed)
            else:
                line.manual_adjust -= 1
            if line.quantity <= 0:
                self.lines.pop(product_id, None)
        return self.snapshot()

    def confirm(self, product_id: int) -> dict:
        with self._lock:
            line = self._require_line(product_id)
            line.confirmed = True
        return self.snapshot()

    def remove(self, product_id: int) -> dict:
        with self._lock:
            line = self.lines.pop(int(product_id), None)
            if line is None:
                raise CartError(f"Product {product_id} is not in the cart.")
            self.ignored_track_ids.update(line.track_ids)
        return self.snapshot()

    def set_discount_percent(self, percent: Decimal | float | int) -> dict:
        value = money(percent)
        if value < 0 or value > 100:
            raise CartError("Discount percent must be between 0 and 100.")
        with self._lock:
            self.discount_percent = value
        return self.snapshot()

    def clear(self) -> dict:
        with self._lock:
            self.lines.clear()
            self.ignored_track_ids.clear()
        return self.snapshot()

    def new_transaction(self) -> dict:
        with self._lock:
            self.lines.clear()
            self.ignored_track_ids.clear()
            self.discount_percent = money(DISCOUNT_PERCENT)
            self.transaction_id = uuid4().hex[:12]
            self.last_invoice_number = None
        return self.snapshot()

    def _ordered_lines(self) -> list[CartLine]:
        return sorted(self.lines.values(), key=lambda item: item.name.lower())

    def _require_line(self, product_id: int) -> CartLine:
        line = self.lines.get(int(product_id))
        if line is None:
            raise CartError(f"Product {product_id} is not in the cart.")
        return line

    def _ensure_line_from_product(self, product) -> CartLine:
        product_id = int(product.id)
        line = self.lines.get(product_id)
        if line is None:
            line = CartLine(
                product_id=product_id,
                sku=str(product.sku),
                name=str(product.name),
                unit_price=product.price,
                tax_rate=product.tax_rate,
                weight=getattr(product, "weight", None),
            )
            self.lines[product_id] = line
        else:
            line.sku = str(product.sku)
            line.name = str(product.name)
            line.weight = getattr(product, "weight", None)
            line.unit_price = money(product.price)
            line.tax_rate = money(product.tax_rate)
        return line

    def _line_for_product(self, product_id: int, track: dict, session: Session | None) -> CartLine | None:
        line = self.lines.get(product_id)
        if line is not None:
            return line
        if session is not None:
            try:
                product = get_product(session, product_id)
            except ProductNotFoundError:
                return None
            return self._ensure_line_from_product(product)
        price = track.get("price")
        tax_rate = track.get("tax_rate")
        if price is None or tax_rate is None:
            return None
        line = CartLine(
            product_id=product_id,
            sku=str(track.get("sku") or ""),
            name=str(track.get("name") or track.get("product_name") or ""),
            unit_price=Decimal(str(price)),
            tax_rate=Decimal(str(tax_rate)),
            weight=track.get("weight"),
        )
        self.lines[product_id] = line
        return line

    def _refresh_from_db(self, session: Session) -> None:
        for product_id, line in list(self.lines.items()):
            try:
                product = get_product(session, product_id)
            except ProductNotFoundError:
                continue
            line.sku = str(product.sku)
            line.name = str(product.name)
            line.weight = getattr(product, "weight", None)
            line.unit_price = money(product.price)
            line.tax_rate = money(product.tax_rate)


_cart: Cart | None = None


def get_cart() -> Cart:
    global _cart
    if _cart is None:
        _cart = Cart()
    return _cart


def reset_cart() -> Cart:
    global _cart
    _cart = Cart()
    return _cart
