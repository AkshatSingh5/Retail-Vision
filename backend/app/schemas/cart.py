from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, field_serializer

from backend.app.utils.money import money_json


class CartAddRequest(BaseModel):
    product_id: int
    quantity: int = Field(default=1, ge=1)


class DiscountRequest(BaseModel):
    percent: Decimal = Field(ge=0, le=100)


class TrackIngest(BaseModel):
    track_id: int
    product_id: int
    sku: str
    name: str | None = None
    product_name: str | None = None
    weight: str | None = None
    unit_price: Decimal | None = None
    price: Decimal | None = None
    tax_rate: Decimal
    confirmed: bool = True


class TracksIngestRequest(BaseModel):
    tracks: list[TrackIngest]


class CartItemOut(BaseModel):
    product_id: int
    sku: str
    name: str
    weight: str | None = None
    quantity: int
    unit_price: int | float
    tax: int | float
    total: int | float
    tax_rate: int | float | None = None
    confirmed: bool | None = None
    track_ids: list[int] | None = None


class CartOut(BaseModel):
    transaction_id: str
    items: list[CartItemOut]
    subtotal: int | float
    tax: int | float
    discount: int | float
    discount_percent: int | float
    grand_total: int | float
    item_count: int
    alerts: list[dict] | None = None


class TransactionItemOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    product_id: int | None
    sku: str
    name: str
    weight: str | None = None
    quantity: int
    unit_price: Decimal
    tax: Decimal
    total: Decimal

    @field_serializer("unit_price", "tax", "total")
    def _money(self, value: Decimal) -> int | float:
        result = money_json(value)
        assert result is not None
        return result


class TransactionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    invoice_number: str
    status: str
    subtotal: Decimal
    tax: Decimal
    discount: Decimal
    discount_percent: Decimal
    grand_total: Decimal
    pdf_path: str | None
    created_at: datetime
    items: list[TransactionItemOut] = []

    @field_serializer("subtotal", "tax", "discount", "discount_percent", "grand_total")
    def _money(self, value: Decimal) -> int | float:
        result = money_json(value)
        assert result is not None
        return result


class CheckoutOut(BaseModel):
    invoice_number: str
    transaction_id: int
    pdf_url: str
    cart: CartOut
    bill: TransactionOut
