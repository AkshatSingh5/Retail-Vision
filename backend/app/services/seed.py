from __future__ import annotations

from decimal import Decimal

import yaml
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.config import ROOT_DIR
from backend.app.models.product import Product

REGISTRY_PATH = ROOT_DIR / "products" / "registry.yaml"


def seed_products_from_registry(session: Session, path=REGISTRY_PATH) -> int:
    """Insert catalog rows from registry.yaml if the products table is empty.

    Prices live in the seed/registry file and the database — never in detector code.
    """
    if not path.exists():
        return 0
    existing = session.scalar(select(func.count()).select_from(Product)) or 0
    if existing:
        return 0

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = payload.get("products") or []
    created = 0
    for row in rows:
        if row.get("price") is None or not row.get("sku") or not row.get("name"):
            continue
        session.add(
            Product(
                sku=str(row["sku"]).strip().upper(),
                name=str(row["name"]),
                brand=row.get("brand"),
                category=row.get("category"),
                variant=row.get("variant"),
                weight=row.get("weight"),
                price=Decimal(str(row["price"])),
                tax_rate=Decimal(str(row.get("tax_rate", 0))),
                yolo_class_id=int(row["class_id"]) if row.get("class_id") is not None else None,
                image_path=row.get("image_path"),
                is_active=True,
            )
        )
        created += 1
    session.flush()
    return created
