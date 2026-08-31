from __future__ import annotations

import logging
import re
import traceback
from decimal import Decimal
from uuid import uuid4

import numpy as np
from sqlalchemy import or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.config import DUPLICATE_MATCH_THRESHOLD
from backend.app.models.product import Product
from backend.app.models.product_embedding import ProductEmbedding
from backend.app.models.product_image import ProductImage
from backend.app.schemas.product import ProductCreate, ProductUpdate, public_image_url, serialize_money, to_price_mapping
from backend.app.services.storage import StorageError, save_product_image
from vision.recognition.vector_bridge import current_model_name

logger = logging.getLogger(__name__)


class ProductNotFoundError(LookupError):
    pass


class ProductConflictError(ValueError):
    pass


class ProductValidationError(ValueError):
    pass


class SimilarProductError(ProductConflictError):
    """Raised when a visually similar product already exists."""

    def __init__(self, message: str, product: Product, confidence: float) -> None:
        super().__init__(message)
        self.product = product
        self.confidence = float(confidence)


def _invalidate_gallery() -> None:
    from vision.recognition.gallery import get_gallery

    get_gallery().invalidate()


def _apply_update(product: Product, payload: ProductUpdate) -> None:
    data = payload.model_dump(exclude_unset=True)
    for key, value in data.items():
        setattr(product, key, value)


def generate_sku(name: str) -> str:
    slug = re.sub(r"[^A-Z0-9]+", "", (name or "").upper())[:12] or "SKU"
    return f"{slug}{uuid4().hex[:6].upper()}"


def list_products(
    session: Session,
    include_inactive: bool = False,
    query: str | None = None,
) -> list[Product]:
    statement = select(Product).order_by(Product.id)
    if not include_inactive:
        statement = statement.where(Product.is_active.is_(True))
    if query:
        pattern = f"%{query.strip()}%"
        statement = statement.where(
            or_(
                Product.name.ilike(pattern),
                Product.sku.ilike(pattern),
                Product.brand.ilike(pattern),
                Product.category.ilike(pattern),
                Product.variant.ilike(pattern),
            )
        )
    return list(session.scalars(statement))


def get_product(session: Session, product_id: int) -> Product:
    product = session.get(Product, product_id)
    if product is None:
        raise ProductNotFoundError(f"Product {product_id} was not found.")
    return product


def get_product_by_sku(session: Session, sku: str) -> Product:
    product = session.scalar(select(Product).where(Product.sku == sku.strip().upper()))
    if product is None:
        raise ProductNotFoundError(f"SKU {sku} was not found.")
    return product


def get_product_by_class_id(session: Session, class_id: int) -> Product:
    product = session.scalar(
        select(Product).where(Product.yolo_class_id == int(class_id), Product.is_active.is_(True))
    )
    if product is None:
        raise ProductNotFoundError(f"No active product mapped to YOLO class_id={class_id}.")
    return product


def create_product(session: Session, payload: ProductCreate) -> Product:
    if not payload.name.strip():
        raise ProductValidationError("Product name is required.")
    existing = session.scalar(select(Product).where(Product.sku == payload.sku))
    if existing is not None:
        raise ProductConflictError("SKU already exists.\nPlease enter a different SKU.")
    product = Product(**payload.model_dump())
    session.add(product)
    try:
        session.flush()
    except IntegrityError as extra:
        session.rollback()
        raise ProductConflictError("SKU already exists.\nPlease enter a different SKU.") from extra
    return product


def update_product(session: Session, product_id: int, payload: ProductUpdate) -> Product:
    product = get_product(session, product_id)
    _apply_update(product, payload)
    try:
        session.flush()
    except IntegrityError as extra:
        session.rollback()
        raise ProductConflictError("SKU already exists.\nPlease enter a different SKU.") from extra
    _invalidate_gallery()
    return product


def delete_product(session: Session, product_id: int) -> Product:
    product = get_product(session, product_id)
    product.is_active = False
    session.flush()
    _invalidate_gallery()
    return product


def list_product_images(session: Session, product_id: int) -> list[ProductImage]:
    get_product(session, product_id)
    return list(
        session.scalars(
            select(ProductImage).where(ProductImage.product_id == product_id).order_by(ProductImage.id)
        )
    )


def add_product_image(
    session: Session,
    product_id: int,
    image_bytes: bytes,
    image_type: str = "reference",
) -> ProductImage:
    product = get_product(session, product_id)
    return _store_image(session, product, image_bytes, image_type=image_type)


def find_similar_product(
    session: Session,
    image_bytes: bytes | None = None,
    *,
    vector: list[float] | None = None,
    threshold: float | None = None,
) -> tuple[Product, float] | None:
    """Return (product, score) when a visually similar product already exists."""
    import numpy as np

    from vision.recognition.gallery import crop_embedding, get_gallery

    min_score = DUPLICATE_MATCH_THRESHOLD if threshold is None else float(threshold)
    query = vector
    if query is None:
        if not image_bytes:
            return None
        import cv2

        array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if frame is None:
            return None
        query = crop_embedding(frame)

    ranked = get_gallery().rank_products(
        np.zeros((64, 64, 3), dtype=np.uint8),
        top_k=3,
        session=session,
        query=query,
    )
    if not ranked or ranked[0].score < min_score:
        return None
    product = get_product(session, int(ranked[0].product_id))
    return product, float(ranked[0].score)


def register_product(
    session: Session,
    *,
    name: str,
    sku: str | None,
    price,
    tax_rate=18,
    brand: str | None = None,
    category: str | None = None,
    variant: str | None = None,
    weight: str | None = None,
    barcode: str | None = None,
    description: str | None = None,
    image_bytes: bytes | None = None,
    frame_bytes: bytes | None = None,
    image_type: str = "camera_capture",
    force_create: bool = False,
) -> Product:
    cleaned_name = (name or "").strip()
    if not cleaned_name:
        raise ProductValidationError("Product name is required.")
    try:
        amount = Decimal(str(price))
    except Exception as extra:
        raise ProductValidationError("Please enter a valid price.") from extra
    if amount < 0:
        raise ProductValidationError("Please enter a valid price.")
    if image_bytes is None or not image_bytes:
        raise ProductValidationError("Product image could not be saved.")

    cleaned_sku = (sku or "").strip().upper() or generate_sku(cleaned_name)

    existing_sku = session.scalar(select(Product).where(Product.sku == cleaned_sku))
    if existing_sku is not None:
        raise ProductConflictError("SKU already exists.\nPlease enter a different SKU.")

    if not force_create:
        similar = find_similar_product(session, image_bytes)
        if similar is not None:
            product, confidence = similar
            raise SimilarProductError(
                "SIMILAR PRODUCT FOUND",
                product=product,
                confidence=confidence,
            )

    payload = ProductCreate(
        sku=cleaned_sku,
        name=cleaned_name,
        brand=(brand or "").strip() or None,
        category=(category or "").strip() or None,
        variant=(variant or "").strip() or None,
        weight=(weight or "").strip() or None,
        price=amount,
        tax_rate=tax_rate,
        barcode=(barcode or "").strip() or None,
        description=(description or "").strip() or None,
        yolo_class_id=None,
        is_active=True,
    )
    print(
        f"[PRODUCT] Request received name={cleaned_name!r} sku={cleaned_sku!r} "
        f"price={amount} brand={brand!r} category={category!r} "
        f"variant={variant!r} weight={weight!r} image_bytes={len(image_bytes)}"
    )
    try:
        print(f"[PRODUCT] Creating product sku={payload.sku!r} name={cleaned_name!r}")
        product = create_product(session, payload)
        print(f"[PRODUCT] Product row flushed id={product.id}")
        _store_image(session, product, image_bytes, image_type=image_type)
        if frame_bytes and frame_bytes != image_bytes:
            try:
                _store_image(
                    session,
                    product,
                    frame_bytes,
                    image_type="scan_frame",
                    index_gallery=False,
                )
            except ProductValidationError:
                print("[IMAGE] Full-frame archive skipped (invalid frame bytes)")
        product.image_url = public_image_url(int(product.id), product.image_path)
        session.flush()
        emb_row = session.execute(
            select(ProductEmbedding.id).where(ProductEmbedding.product_id == int(product.id)).limit(1)
        ).first()
        if emb_row is None:
            raise ProductValidationError(
                "Product embedding was not stored. Registration rolled back."
            )
        print(
            f"[PRODUCT] Product registration complete id={product.id} sku={product.sku} "
            f"image_url={product.image_url} embedding_id={emb_row[0]}"
        )
    except (ProductConflictError, ProductValidationError, SimilarProductError):
        raise
    except Exception as extra:
        session.rollback()
        logger.exception("Product registration failed")
        print(f"[PRODUCT] Registration failed: {extra}")
        print(traceback.format_exc())
        # Keep API/UI message clean; full cause is in logs above.
        raise ProductValidationError("Product could not be saved. Please try again.") from extra
    return product


def _store_image(
    session: Session,
    product: Product,
    image_bytes: bytes,
    image_type: str,
    *,
    index_gallery: bool = True,
) -> ProductImage:
    from vision.recognition.gallery import (
        crop_embedding,
        embedding_to_json,
        embedding_variants,
        get_gallery,
    )

    print(
        f"[IMAGE] Received product_id={product.id} bytes={len(image_bytes)} "
        f"type={image_type!r}"
    )
    try:
        relative, decoded = save_product_image(int(product.id), image_bytes, image_type=image_type)
    except StorageError as extra:
        raise ProductValidationError(str(extra) or "Product image could not be saved.") from extra
    h, w = int(decoded.shape[0]), int(decoded.shape[1])
    print(f"[IMAGE] Image saved path={relative} shape={h}x{w}")

    vectors: list[list[float]] = []
    vector: list[float]
    if index_gallery:
        try:
            model_name = current_model_name()
            print(f"[EMBEDDING] Generating embedding model={model_name} input_shape={decoded.shape}")
            vectors = embedding_variants(decoded)
            vector = vectors[0] if vectors else crop_embedding(decoded)
            arr = np.asarray(vector, dtype=np.float32)
            norm = float(np.linalg.norm(arr)) if arr.size else 0.0
            print(
                f"[EMBEDDING] Generated dim={len(vector)} dtype=float32 "
                f"L2_norm={norm:.6f} variants={len(vectors)}"
            )
            if abs(norm - 1.0) > 0.05:
                print(f"[EMBEDDING] WARNING: embedding not unit-normalized (norm={norm:.6f})")
        except Exception as extra:
            logger.exception("Embedding generation failed")
            print(traceback.format_exc())
            raise ProductValidationError(
                f"Product embedding could not be generated: {extra}"
            ) from extra
    else:
        vector = []

    row = ProductImage(
        product_id=int(product.id),
        image_path=relative,
        storage_key=relative,
        image_type=image_type[:32],
        embedding=embedding_to_json(vector) if index_gallery and vector else None,
    )
    session.add(row)
    session.flush()
    row.image_url = f"/products/{int(product.id)}/images/{int(row.id)}/file"
    print(f"[IMAGE] product_images row id={row.id} product_id={row.product_id} url={row.image_url}")

    if not product.image_path:
        product.image_path = relative
        product.image_url = public_image_url(int(product.id), relative)
    if index_gallery and vector and not product.recognition_embedding:
        product.recognition_embedding = embedding_to_json(vector)

    if index_gallery and vectors:
        from backend.app.services.vector_search import embedding_payload, sync_pgvector

        model_name = current_model_name()
        # One DB row per (image, model) — unique constraint uq_product_embeddings_image_model.
        # Extra lighting variants still go into the in-memory gallery below.
        primary = vectors[0]
        embedding_row = ProductEmbedding(
            product_id=int(product.id),
            product_image_id=int(row.id),
            embedding=embedding_payload(primary, model_name),
            model_name=model_name,
        )
        session.add(embedding_row)
        session.flush()
        print(
            f"[EMBEDDING] product_embeddings row id={embedding_row.id} "
            f"product_id={product.id} product_image_id={row.id} "
            f"model={model_name} dim={len(primary)} "
            f"(stored 1 of {len(vectors)} variant(s) in DB)"
        )
        sync_pgvector(session, int(embedding_row.id), primary)

        price = serialize_money(product.price)
        tax = serialize_money(product.tax_rate)
        if price is not None and tax is not None:
            print("[INDEX] Adding product to recognition index")
            get_gallery().register_product(
                product_id=int(product.id),
                sku=str(product.sku),
                name=str(product.name),
                price=price,
                tax_rate=tax,
                vectors=vectors,
                brand=product.brand,
                category=product.category,
            )
    return row


def price_mapping_for_class(session: Session, class_id: int) -> dict:
    product = get_product_by_class_id(session, class_id)
    return to_price_mapping(product).model_dump()


def identity_from_product(product: Product) -> dict:
    return {
        "product_id": int(product.id),
        "sku": str(product.sku),
        "product_name": str(product.name),
        "price": serialize_money(product.price),
        "tax_rate": serialize_money(product.tax_rate),
        "brand": product.brand,
        "category": product.category,
        "class_id": product.yolo_class_id,
    }


def as_money(value):
    return serialize_money(value)
