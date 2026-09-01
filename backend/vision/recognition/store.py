from __future__ import annotations

import time
from dataclasses import dataclass

from sqlalchemy import select

from backend.app.database import get_session_factory, init_db
from backend.app.models.product import Product
from backend.app.schemas.product import serialize_money
from backend.app.services.product_service import ProductNotFoundError, get_product_by_class_id
from backend.app.services.seed import seed_products_from_registry


@dataclass(frozen=True)
class PricedProduct:
    product_id: int
    sku: str
    name: str
    price: int | float
    tax_rate: int | float
    class_id: int | None
    brand: str | None = None
    category: str | None = None

    def price_mapping(self) -> dict:
        return {
            "product_id": self.product_id,
            "sku": self.sku,
            "name": self.name,
            "price": self.price,
            "tax_rate": self.tax_rate,
        }


def _row_to_priced(product: Product) -> PricedProduct:
    price = serialize_money(product.price)
    tax = serialize_money(product.tax_rate)
    assert price is not None and tax is not None
    return PricedProduct(
        product_id=int(product.id),
        sku=str(product.sku),
        name=str(product.name),
        price=price,
        tax_rate=tax,
        class_id=product.yolo_class_id,
        brand=product.brand,
        category=product.category,
    )


class DatabaseProductStore:
    """Read-only priced catalog used by the vision pipeline.

    Detector code never contains product prices; this store queries SQLAlchemy.
    """

    def __init__(self, ttl_seconds: float = 5.0) -> None:
        self.ttl_seconds = ttl_seconds
        self._by_class: dict[int, PricedProduct] = {}
        self._loaded_at = 0.0
        self._ready = False

    def _refresh(self) -> None:
        init_db()
        session = get_session_factory()()
        try:
            seed_products_from_registry(session)
            session.commit()
            rows = session.scalars(select(Product).where(Product.is_active.is_(True))).all()
            mapping: dict[int, PricedProduct] = {}
            for row in rows:
                if row.yolo_class_id is None:
                    continue
                mapping[int(row.yolo_class_id)] = _row_to_priced(row)
            self._by_class = mapping
            self._loaded_at = time.monotonic()
            self._ready = True
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _ensure(self) -> None:
        if not self._ready or (time.monotonic() - self._loaded_at) >= self.ttl_seconds:
            self._refresh()

    def get_by_class_id(self, class_id: int) -> PricedProduct | None:
        self._ensure()
        cached = self._by_class.get(int(class_id))
        if cached is not None:
            return cached
        session = get_session_factory()()
        try:
            product = get_product_by_class_id(session, int(class_id))
        except ProductNotFoundError:
            product = None
        finally:
            session.close()
        if product is None:
            return None
        priced = _row_to_priced(product)
        self._by_class[int(class_id)] = priced
        return priced

    def invalidate(self) -> None:
        self._ready = False
        self._by_class.clear()
