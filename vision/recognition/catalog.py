from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from backend.app.config import ROOT_DIR

REGISTRY_PATH = ROOT_DIR / "products" / "registry.yaml"
PRODUCT_ID_BASE = 101


@dataclass(frozen=True)
class ProductIdentity:
    class_id: int
    product_id: int
    sku: str
    product_name: str
    price: int | float | None = None
    tax_rate: int | float | None = None
    brand: str | None = None
    category: str | None = None

    def as_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "product_id": self.product_id,
            "sku": self.sku,
            "product_name": self.product_name,
            "price": self.price,
            "tax_rate": self.tax_rate,
            "brand": self.brand,
            "category": self.category,
        }

    def price_mapping(self) -> dict | None:
        if self.price is None or self.tax_rate is None:
            return None
        return {
            "product_id": self.product_id,
            "sku": self.sku,
            "name": self.product_name,
            "price": self.price,
            "tax_rate": self.tax_rate,
        }


class ProductCatalog:
    """Maps YOLO class_id → product_id → SKU → product name."""

    def __init__(self, products: list[ProductIdentity]) -> None:
        if not products:
            raise ValueError("Product catalog is empty.")
        self.products = list(products)
        self.by_class_id = {item.class_id: item for item in self.products}
        self.by_product_id = {item.product_id: item for item in self.products}
        self.by_sku = {item.sku: item for item in self.products}

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> ProductCatalog:
        registry_path = path or REGISTRY_PATH
        with registry_path.open(encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        rows = payload.get("products") or []
        products: list[ProductIdentity] = []
        for row in rows:
            class_id = int(row["class_id"])
            products.append(
                ProductIdentity(
                    class_id=class_id,
                    product_id=int(row.get("product_id") or PRODUCT_ID_BASE + class_id),
                    sku=str(row["sku"]),
                    product_name=str(row["name"]),
                )
            )
        return cls(products)

    def resolve(self, class_id: int) -> ProductIdentity:
        identity = self.by_class_id.get(int(class_id))
        if identity is None:
            return ProductIdentity(
                class_id=int(class_id),
                product_id=PRODUCT_ID_BASE + int(class_id),
                sku=f"unknown-{class_id}",
                product_name=f"Unknown class {class_id}",
            )
        return identity

    def names(self) -> dict[int, str]:
        return {item.class_id: item.product_name for item in self.products}


@lru_cache(maxsize=1)
def load_catalog(path: str | None = None) -> ProductCatalog:
    return ProductCatalog.from_yaml(Path(path) if path else None)
