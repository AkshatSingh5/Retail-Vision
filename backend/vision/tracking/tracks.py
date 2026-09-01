from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TypedDict


class TrackState(str, Enum):
    ENTERING = "entering"
    VISIBLE = "visible"
    LEAVING = "leaving"
    EXITED = "exited"


class TrackedProduct(TypedDict):
    track_id: int
    product_id: int
    class_id: int
    product_name: str
    sku: str
    price: int | float | None
    tax_rate: int | float | None
    is_unknown: bool
    confidence: float
    bbox: list[float]
    state: str
    confirmed: bool


@dataclass
class Track:
    track_id: int
    class_id: int
    product_id: int
    sku: str
    product_name: str
    bbox: list[float]
    confidence: float
    hits: int = 1
    consecutive_hits: int = 1
    misses: int = 0
    confirmed: bool = False
    state: TrackState = TrackState.ENTERING
    first_frame: int = 0
    last_frame: int = 0
    history: list[float] = field(default_factory=list)
    price: int | float | None = None
    tax_rate: int | float | None = None
    is_unknown: bool = False
    locked: bool = False

    def to_output(self) -> TrackedProduct:
        return {
            "track_id": self.track_id,
            "product_id": self.product_id,
            "class_id": self.class_id,
            "product_name": self.product_name,
            "sku": self.sku,
            "price": self.price,
            "tax_rate": self.tax_rate,
            "confidence": round(float(self.confidence), 4),
            "bbox": [round(float(v), 2) for v in self.bbox],
            "state": self.state.value,
            "confirmed": self.confirmed,
            "is_unknown": bool(self.is_unknown),
        }

    def public_json(self) -> dict:
        payload = {
            "track_id": self.track_id,
            "product_id": self.product_id,
            "class_id": self.class_id,
            "product_name": self.product_name,
            "sku": self.sku,
            "confidence": round(float(self.confidence), 4),
            "bbox": [round(float(v), 2) for v in self.bbox],
        }
        mapping = self.price_mapping()
        if mapping is not None:
            payload.update(mapping)
        return payload

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
