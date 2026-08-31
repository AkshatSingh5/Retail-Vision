from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator


def public_image_url(product_id: int | None, image_path: str | None) -> str | None:
    """Client-facing image route. Never expose a raw filesystem path as the URL."""
    if product_id and image_path:
        return f"/products/{int(product_id)}/image"
    return None


def serialize_money(value: Decimal | float | int | None) -> int | float | None:
    """JSON number: 40 instead of 40.0 when the amount is whole rupees."""
    if value is None:
        return None
    number = Decimal(str(value)).quantize(Decimal("0.01"))
    if number == number.to_integral_value():
        return int(number)
    return float(number)


class ProductCreate(BaseModel):
    sku: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=255)
    brand: str | None = None
    category: str | None = None
    variant: str | None = None
    weight: str | None = None
    price: Decimal = Field(ge=0)
    tax_rate: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    barcode: str | None = None
    description: str | None = Field(default=None, max_length=1024)
    yolo_class_id: int | None = Field(default=None, ge=0)
    image_path: str | None = None
    image_url: str | None = None
    is_active: bool = True

    @field_validator("sku")
    @classmethod
    def normalize_sku(cls, value: str) -> str:
        sku = value.strip().upper()
        if not sku:
            raise ValueError("sku cannot be empty")
        return sku


class ProductUpdate(BaseModel):
    sku: str | None = Field(default=None, min_length=1, max_length=64)
    name: str | None = Field(default=None, min_length=1, max_length=255)
    brand: str | None = None
    category: str | None = None
    variant: str | None = None
    weight: str | None = None
    price: Decimal | None = Field(default=None, ge=0)
    tax_rate: Decimal | None = Field(default=None, ge=0, le=100)
    barcode: str | None = None
    description: str | None = Field(default=None, max_length=1024)
    yolo_class_id: int | None = Field(default=None, ge=0)
    image_path: str | None = None
    image_url: str | None = None
    is_active: bool | None = None

    @field_validator("sku")
    @classmethod
    def normalize_sku(cls, value: str | None) -> str | None:
        if value is None:
            return None
        sku = value.strip().upper()
        if not sku:
            raise ValueError("sku cannot be empty")
        return sku


class ProductRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    sku: str
    name: str
    brand: str | None
    category: str | None
    variant: str | None = None
    weight: str | None = None
    price: Decimal
    tax_rate: Decimal
    barcode: str | None = None
    description: str | None = None
    yolo_class_id: int | None
    image_path: str | None
    image_url: str | None = None
    is_active: bool
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def _fill_image_url(self) -> ProductRead:
        if not self.image_url:
            self.image_url = public_image_url(self.id, self.image_path)
        return self

    @field_serializer("price", "tax_rate")
    def _money(self, value: Decimal) -> int | float:
        result = serialize_money(value)
        assert result is not None
        return result


class RegisteredProductSummary(BaseModel):
    id: int
    name: str
    sku: str
    price: int | float
    image_url: str | None = None


class ProductImageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    product_id: int
    image_path: str
    storage_key: str | None = None
    image_url: str | None = None
    image_type: str
    created_at: datetime


class ProductRegisterOut(ProductRead):
    cart_added: bool = False
    message: str = ""
    success: bool = True
    product: RegisteredProductSummary | None = None


def register_out_from_product(product, *, cart_added: bool, message: str) -> ProductRegisterOut:
    read = ProductRead.model_validate(product)
    payload = read.model_dump()
    return ProductRegisterOut(
        **payload,
        cart_added=cart_added,
        success=True,
        message=message,
        product=RegisteredProductSummary(
            id=read.id,
            name=read.name,
            sku=read.sku,
            price=serialize_money(read.price) or 0,
            image_url=read.image_url,
        ),
    )


class ScanMatchedProduct(BaseModel):
    product_id: int
    id: int | None = None
    name: str
    sku: str
    price: int | float
    tax_rate: int | float | None = None
    image_url: str | None = None
    confidence: float
    barcode: str | None = None
    category: str | None = None
    brand: str | None = None
    variant: str | None = None
    weight: str | None = None


class ScanRecognitionMeta(BaseModel):
    detection_confidence: float | None = None
    similarity: float | None = None
    best_similarity: float | None = None
    second_similarity: float | None = None
    margin: float | None = None
    threshold: float | None = None
    required_margin: float | None = None
    actual_margin: float | None = None
    crop_source: str | None = None
    crop_dimensions: dict[str, int] | None = None
    embedding_model: str | None = None
    embedding_dimensions: int | None = None
    normalized: bool | None = None
    failing_stage: str | None = None
    best_match: dict | None = None
    second_match: dict | None = None
    top_matches: list[dict] | None = None


class ScanDetectionItem(BaseModel):
    product_id: int | str | None = None
    product_name: str | None = None
    sku: str | None = None
    confidence: float = 0.0
    bbox: list[float] | None = None
    status: str = "unknown"


class ProductScanOut(BaseModel):
    found: bool
    status: str
    message: str
    confidence: float = 0.0
    threshold: float | None = None
    detections: int = 0
    success: bool = False
    items: list[ScanDetectionItem] | None = None
    scan_id: str | None = None
    preview_url: str | None = None
    reason: str | None = None
    match_type: str | None = None
    product: ScanMatchedProduct | None = None
    recognition: ScanRecognitionMeta | None = None
    best_match: dict | None = None
    second_match: dict | None = None
    candidates: list[dict] | None = None
    warnings: list[str] | None = None


class SimilarProductOut(BaseModel):
    status: str = "similar_product_found"
    message: str
    confidence: float
    product: RegisteredProductSummary



class ProductPriceMapping(BaseModel):
    """Detection-facing price payload. Amounts always come from the database."""

    product_id: int
    sku: str
    name: str
    price: int | float
    tax_rate: int | float


def to_price_mapping(product) -> ProductPriceMapping:
    price = serialize_money(product.price)
    tax = serialize_money(product.tax_rate)
    assert price is not None and tax is not None
    return ProductPriceMapping(
        product_id=int(product.id),
        sku=str(product.sku),
        name=str(product.name),
        price=price,
        tax_rate=tax,
    )
