from __future__ import annotations

import logging
import traceback
from decimal import Decimal, InvalidOperation
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse, Response
from sqlalchemy.orm import Session

from backend.app.config import RECOGNITION_DEBUG, ROOT_DIR
from backend.app.database import get_db

logger = logging.getLogger(__name__)
from backend.app.schemas.product import (
    ProductCreate,
    ProductImageRead,
    ProductPriceMapping,
    ProductRead,
    ProductRegisterOut,
    ProductScanOut,
    ProductUpdate,
    SimilarProductOut,
    public_image_url,
    register_out_from_product,
    serialize_money,
)
from backend.app.services.camera_hub import CameraControlError, get_camera_hub
from backend.app.services.cart_service import CartError, get_cart
from backend.app.services.product_service import (
    ProductConflictError,
    ProductNotFoundError,
    ProductValidationError,
    SimilarProductError,
    add_product_image,
    create_product,
    delete_product,
    get_product,
    get_product_by_sku,
    list_product_images,
    list_products,
    price_mapping_for_class,
    register_product,
    update_product,
)
from backend.app.services.scan_service import (
    ScanError,
    peek_pending_scan,
    pop_pending_scan_bundle,
    recognize_frame,
    validate_image_bytes,
)
from backend.app.services.storage import StorageError, get_product_image
from vision.recognition.store import DatabaseProductStore

router = APIRouter(prefix="/products", tags=["products"])


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, ProductNotFoundError):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    if isinstance(exc, ProductConflictError):
        detail = str(exc)
        if "SKU" in detail:
            return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail)
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail)
    if isinstance(exc, (ProductValidationError, ScanError, StorageError)):
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


def _refresh_recognition(*, wipe_gallery: bool = True) -> None:
    get_camera_hub()
    try:
        from vision.recognition.gallery import get_gallery

        if wipe_gallery:
            get_gallery().invalidate()
    except Exception:
        pass
    if get_camera_hub().pipeline is not None:
        store = getattr(get_camera_hub().pipeline.identifier, "store", None)
        if isinstance(store, DatabaseProductStore):
            store.invalidate()


@router.get("", response_model=list[ProductRead])
def api_list_products(
    include_inactive: bool = Query(False),
    q: str | None = Query(None, description="Search name / SKU / brand / category"),
    session: Session = Depends(get_db),
) -> list[ProductRead]:
    return list_products(session, include_inactive=include_inactive, query=q)


@router.get("/search", response_model=list[ProductRead])
def api_search_products(
    q: str = Query(..., min_length=1),
    session: Session = Depends(get_db),
) -> list[ProductRead]:
    """Catalog text search only — not used for visual recognition."""
    return list_products(session, query=q)


@router.get("/sku/{sku}", response_model=ProductRead)
def api_get_product_by_sku(sku: str, session: Session = Depends(get_db)) -> ProductRead:
    try:
        return get_product_by_sku(session, sku)
    except ProductNotFoundError as extra:
        raise _http_error(extra) from extra


@router.get("/class/{class_id}", response_model=ProductPriceMapping)
def api_get_product_by_class(class_id: int, session: Session = Depends(get_db)) -> ProductPriceMapping:
    """YOLO class_id → product_id → SKU → name → price (from the database)."""
    try:
        return ProductPriceMapping.model_validate(price_mapping_for_class(session, class_id))
    except ProductNotFoundError as extra:
        raise _http_error(extra) from extra


@router.post("/scan", response_model=ProductScanOut)
async def api_scan_product(
    image: UploadFile | None = File(None),
    use_camera: bool = Form(True),
    session: Session = Depends(get_db),
) -> ProductScanOut:
    """Capture or upload a frame → YOLO → recognition → product or not-found."""
    image_bytes = b""
    content_type = None
    if image is not None:
        image_bytes = await image.read()
        content_type = image.content_type
    if not image_bytes and use_camera:
        try:
            image_bytes = get_camera_hub().capture_frame()
        except CameraControlError as extra:
            raise HTTPException(status_code=400, detail=str(extra)) from extra
    print(f"[SCAN] Image received ({len(image_bytes)} bytes)")
    try:
        validate_image_bytes(image_bytes, content_type=content_type)
        result = recognize_frame(session, image_bytes)
    except ScanError as extra:
        raise _http_error(extra) from extra
    except Exception as extra:
        print(f"[SCAN] Recognition failed: {extra}")
        raise HTTPException(
            status_code=503,
            detail="Recognition service unavailable.",
        ) from extra
    # Valid scan with no match is 200 + product_not_found (not HTTP 404).
    return ProductScanOut.model_validate(result)


@router.get("/scan/{scan_id}/preview")
def api_scan_preview(scan_id: str) -> Response:
    payload = peek_pending_scan(scan_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Scan preview expired. Please scan again.")
    return Response(content=payload, media_type="image/jpeg")


@router.post("/register", response_model=ProductRegisterOut, status_code=status.HTTP_201_CREATED)
async def api_register_product(
    name: str = Form(...),
    sku: str | None = Form(None),
    price: str = Form(...),
    tax_rate: str = Form("18"),
    brand: str | None = Form(None),
    category: str | None = Form(None),
    variant: str | None = Form(None),
    weight: str | None = Form(None),
    barcode: str | None = Form(None),
    description: str | None = Form(None),
    track_id: int | None = Form(None),
    scan_id: str | None = Form(None),
    add_to_cart: bool = Form(True),
    force_create: bool = Form(False),
    image: UploadFile | None = File(None),
    session: Session = Depends(get_db),
) -> ProductRegisterOut:
    image_bytes = await image.read() if image is not None else b""
    mime = image.content_type if image else None
    print(
        f"[REGISTER] Request received name={name!r} price={price!r} "
        f"brand={brand!r} category={category!r} variant={variant!r} weight={weight!r} "
        f"scan_id={scan_id!r} force_create={force_create} "
        f"image_file={'yes' if image is not None else 'no'} "
        f"image_bytes={len(image_bytes)} mime={mime!r}"
    )
    frame_bytes: bytes | None = None
    if not image_bytes and scan_id:
        bundle = pop_pending_scan_bundle(scan_id) or {}
        image_bytes = bundle.get("crop") or bundle.get("image") or b""
        frame_bytes = bundle.get("image") or None
        print(f"[REGISTER] Loaded pending scan crop bytes={len(image_bytes)}")
    if not image_bytes and track_id is not None:
        image_bytes = get_camera_hub().crop_bytes(int(track_id)) or b""
        print(f"[REGISTER] Loaded track crop bytes={len(image_bytes)}")
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Product image could not be saved.")
    try:
        validate_image_bytes(image_bytes, content_type=image.content_type if image else "image/jpeg")
    except ScanError as extra:
        raise _http_error(extra) from extra
    try:
        amount = Decimal(str(price).replace("₹", "").replace(",", "").strip())
        tax = Decimal(str(tax_rate).replace("%", "").strip() or "18")
    except (InvalidOperation, ValueError) as extra:
        raise HTTPException(status_code=400, detail="Please enter a valid price.") from extra
    try:
        product = register_product(
            session,
            name=name,
            sku=sku,
            price=amount,
            tax_rate=tax,
            brand=brand,
            category=category,
            variant=variant,
            weight=weight,
            barcode=barcode,
            description=description,
            image_bytes=image_bytes,
            frame_bytes=frame_bytes,
            image_type="camera_capture",
            force_create=force_create,
        )
        session.flush()
        cart_added = False
        if add_to_cart:
            get_cart().add_from_registration(session, int(product.id), track_id)
            if track_id is not None:
                get_camera_hub().bind_registered_track(int(track_id), product)
            cart_added = True
        _refresh_recognition(wipe_gallery=False)
        print(f"[REGISTER] Success product_id={product.id} cart_added={cart_added}")
    except SimilarProductError as extra:
        product = extra.product
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "status": "similar_product_found",
                "message": str(extra),
                "confidence": round(float(extra.confidence), 4),
                "product": {
                    "id": int(product.id),
                    "name": str(product.name),
                    "sku": str(product.sku),
                    "price": serialize_money(product.price) or 0,
                    "image_url": public_image_url(int(product.id), product.image_path),
                },
            },
        ) from extra
    except (ProductConflictError, ProductValidationError, CartError, ProductNotFoundError) as extra:
        logger.exception("Register validation/conflict error: %s", extra)
        raise _http_error(extra) from extra
    except Exception as extra:
        logger.exception("Register unexpected failure")
        print(f"[REGISTER] Unexpected failure: {extra}")
        print(traceback.format_exc())
        detail = (
            f"Product could not be saved: {extra}"
            if RECOGNITION_DEBUG
            else "Product could not be saved. Please try again."
        )
        raise HTTPException(status_code=400, detail=detail) from extra
    return register_out_from_product(
        product,
        cart_added=cart_added,
        message=f"{product.name} added successfully.",
    )


@router.get("/{product_id}", response_model=ProductRead)
def api_get_product(product_id: int, session: Session = Depends(get_db)) -> ProductRead:
    try:
        product = get_product(session, product_id)
    except ProductNotFoundError as extra:
        raise _http_error(extra) from extra
    if not product.is_active:
        raise HTTPException(status_code=404, detail=f"Product {product_id} was not found.")
    return product


@router.get("/{product_id}/image")
def api_product_primary_image(product_id: int, session: Session = Depends(get_db)) -> FileResponse:
    try:
        product = get_product(session, product_id)
    except ProductNotFoundError as extra:
        raise _http_error(extra) from extra
    if not product.is_active:
        raise HTTPException(status_code=404, detail="Product image was not found.")

    candidates: list[str] = []
    if product.image_path:
        candidates.append(str(product.image_path))
    try:
        for row in list_product_images(session, product_id):
            if row.image_path and str(row.image_path) not in candidates:
                candidates.append(str(row.image_path))
    except ProductNotFoundError:
        pass

    last_error: Exception | None = None
    for key in candidates:
        try:
            path = get_product_image(key)
            return FileResponse(path, media_type="image/jpeg")
        except StorageError as extra:
            last_error = extra
            continue
        except Exception as extra:
            last_error = extra
            continue

    detail = str(last_error) if last_error else "Product image was not found."
    raise HTTPException(status_code=404, detail=detail)


@router.post("", response_model=ProductRead, status_code=status.HTTP_201_CREATED)
def api_create_product(payload: ProductCreate, session: Session = Depends(get_db)) -> ProductRead:
    try:
        return create_product(session, payload)
    except (ProductConflictError, ProductValidationError) as extra:
        raise _http_error(extra) from extra


@router.put("/{product_id}", response_model=ProductRead)
def api_update_product(
    product_id: int,
    payload: ProductUpdate,
    session: Session = Depends(get_db),
) -> ProductRead:
    try:
        return update_product(session, product_id, payload)
    except (ProductNotFoundError, ProductConflictError, ProductValidationError) as extra:
        raise _http_error(extra) from extra


@router.delete("/{product_id}", status_code=status.HTTP_204_NO_CONTENT)
def api_delete_product(product_id: int, session: Session = Depends(get_db)) -> Response:
    try:
        delete_product(session, product_id)
    except ProductNotFoundError as extra:
        raise _http_error(extra) from extra
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{product_id}/images", response_model=list[ProductImageRead])
def api_list_images(product_id: int, session: Session = Depends(get_db)) -> list[ProductImageRead]:
    try:
        return list_product_images(session, product_id)
    except ProductNotFoundError as extra:
        raise _http_error(extra) from extra


@router.post("/{product_id}/images", response_model=ProductImageRead, status_code=status.HTTP_201_CREATED)
async def api_add_image(
    product_id: int,
    image: UploadFile = File(...),
    image_type: str = Form("extra"),
    session: Session = Depends(get_db),
) -> ProductImageRead:
    payload = await image.read()
    try:
        validate_image_bytes(payload, content_type=image.content_type)
        row = add_product_image(session, product_id, payload, image_type=image_type)
    except (ProductNotFoundError, ProductValidationError, ScanError) as extra:
        raise _http_error(extra) from extra
    _refresh_recognition()
    return row


@router.get("/{product_id}/images/{image_id}/file")
def api_image_file(product_id: int, image_id: int, session: Session = Depends(get_db)) -> FileResponse:
    try:
        rows = list_product_images(session, product_id)
    except ProductNotFoundError as extra:
        raise _http_error(extra) from extra
    row = next((item for item in rows if int(item.id) == int(image_id)), None)
    if row is None:
        raise HTTPException(status_code=404, detail="Image was not found.")
    try:
        path = get_product_image(str(row.image_path))
    except StorageError:
        path = Path(row.image_path)
        if not path.is_absolute():
            path = ROOT_DIR / path
        if not path.exists():
            raise HTTPException(status_code=404, detail="Product image could not be saved.") from None
    return FileResponse(path, media_type="image/jpeg")
