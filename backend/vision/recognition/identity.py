from __future__ import annotations

from typing import TypedDict

from vision.detection.types import Detection
from vision.recognition.catalog import ProductCatalog, ProductIdentity, load_catalog
from vision.recognition.embedding import EmbeddingRefiner
from vision.recognition.gallery import get_gallery
from vision.recognition.store import DatabaseProductStore, PricedProduct


class IdentifiedDetection(TypedDict):
    class_id: int
    confidence: float
    bbox: list[float]
    product_id: int
    sku: str
    product_name: str
    price: int | float | None
    tax_rate: int | float | None
    is_unknown: bool


class ProductIdentifier:
    """YOLO class_id → catalog identity; prices come only from the database store.

    Newly registered SKUs (no YOLO class) are matched from the crop gallery.
    """

    def __init__(
        self,
        catalog: ProductCatalog | None = None,
        refiner: EmbeddingRefiner | None = None,
        store: DatabaseProductStore | None = None,
        use_database: bool = True,
    ) -> None:
        self.catalog = catalog or load_catalog()
        self.refiner = refiner or EmbeddingRefiner(enabled=False)
        self.store = store
        if use_database and self.store is None:
            self.store = DatabaseProductStore()

    def identify(self, detection: Detection, crop=None) -> IdentifiedDetection:
        if crop is not None:
            # Threshold + margin enforced inside gallery.match(); never guess.
            match = get_gallery().match(crop)
            if match is not None and match.accepted:
                return {
                    "class_id": int(detection["class_id"]),
                    "confidence": float(match.score),
                    "bbox": list(detection["bbox"]),
                    "product_id": match.product_id,
                    "sku": match.sku,
                    "product_name": match.name,
                    "price": match.price,
                    "tax_rate": match.tax_rate,
                    "is_unknown": False,
                }
        identity = self._resolve(int(detection["class_id"]))
        if self.refiner.enabled and crop is not None:
            identity = self.refiner.refine(crop, identity)
        unknown = identity.sku.lower().startswith("unknown") or identity.price is None
        display_name = "New Product Detected" if identity.sku.lower().startswith("unknown") else identity.product_name
        return {
            "class_id": identity.class_id,
            "confidence": float(detection["confidence"]),
            "bbox": list(detection["bbox"]),
            "product_id": identity.product_id,
            "sku": identity.sku,
            "product_name": display_name,
            "price": identity.price,
            "tax_rate": identity.tax_rate,
            "is_unknown": unknown,
        }

    def identify_all(self, detections: list[Detection], crops: list | None = None) -> list[IdentifiedDetection]:
        results: list[IdentifiedDetection] = []
        for index, detection in enumerate(detections):
            crop = crops[index] if crops is not None and index < len(crops) else None
            results.append(self.identify(detection, crop=crop))
        return results

    def identity_for_class(self, class_id: int) -> ProductIdentity:
        return self._resolve(int(class_id))

    def price_mapping_for_class(self, class_id: int) -> dict | None:
        identity = self._resolve(int(class_id))
        return identity.price_mapping()

    def _resolve(self, class_id: int) -> ProductIdentity:
        priced = self._from_database(class_id)
        if priced is not None:
            return ProductIdentity(
                class_id=class_id,
                product_id=priced.product_id,
                sku=priced.sku,
                product_name=priced.name,
                price=priced.price,
                tax_rate=priced.tax_rate,
                brand=priced.brand,
                category=priced.category,
            )
        return self.catalog.resolve(class_id)

    def _from_database(self, class_id: int) -> PricedProduct | None:
        if self.store is None:
            return None
        try:
            return self.store.get_by_class_id(class_id)
        except Exception:
            return None
