from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.app.config import ROOT_DIR
from backend.app.database import get_db
from backend.app.schemas.cart import (
    CartAddRequest,
    CartOut,
    CheckoutOut,
    DiscountRequest,
    TracksIngestRequest,
    TransactionOut,
)
from backend.app.services.camera_hub import get_camera_hub
from backend.app.services.cart_service import CartError, get_cart
from backend.app.services.checkout import (
    CheckoutError,
    checkout,
    get_transaction,
    get_transaction_by_invoice,
    list_transactions,
)

router = APIRouter(tags=["cart"])


def _cart_error(exc: Exception) -> HTTPException:
    if isinstance(exc, CheckoutError) and "not found" in str(exc).lower():
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.get("/cart", response_model=CartOut)
def api_get_cart(session: Session = Depends(get_db)) -> dict:
    return get_cart().snapshot(session)


@router.post("/cart/items", response_model=CartOut)
def api_add_cart_item(payload: CartAddRequest, session: Session = Depends(get_db)) -> dict:
    try:
        return get_cart().add_product(session, payload.product_id, payload.quantity)
    except (CartError, LookupError) as extra:
        raise _cart_error(extra) from extra


@router.post("/cart/items/{product_id}/increase", response_model=CartOut)
def api_increase_item(product_id: int) -> dict:
    try:
        return get_cart().increase(product_id)
    except CartError as extra:
        raise _cart_error(extra) from extra


@router.post("/cart/items/{product_id}/decrease", response_model=CartOut)
def api_decrease_item(product_id: int) -> dict:
    try:
        return get_cart().decrease(product_id)
    except CartError as extra:
        raise _cart_error(extra) from extra


@router.post("/cart/items/{product_id}/confirm", response_model=CartOut)
def api_confirm_item(product_id: int) -> dict:
    try:
        return get_cart().confirm(product_id)
    except CartError as extra:
        raise _cart_error(extra) from extra


@router.delete("/cart/items/{product_id}", response_model=CartOut)
def api_remove_item(product_id: int) -> dict:
    try:
        return get_cart().remove(product_id)
    except CartError as extra:
        raise _cart_error(extra) from extra


@router.post("/cart/clear", response_model=CartOut)
def api_clear_cart() -> dict:
    return get_cart().clear()


@router.post("/cart/discount", response_model=CartOut)
def api_set_discount(payload: DiscountRequest) -> dict:
    try:
        return get_cart().set_discount_percent(payload.percent)
    except CartError as extra:
        raise _cart_error(extra) from extra


@router.post("/cart/tracks", response_model=CartOut)
def api_ingest_tracks(payload: TracksIngestRequest, session: Session = Depends(get_db)) -> dict:
    tracks = []
    for item in payload.tracks:
        tracks.append(
            {
                "track_id": item.track_id,
                "product_id": item.product_id,
                "sku": item.sku,
                "name": item.name or item.product_name,
                "product_name": item.product_name or item.name,
                "price": item.price if item.price is not None else item.unit_price,
                "tax_rate": item.tax_rate,
                "confirmed": item.confirmed,
            }
        )
    return get_cart().apply_tracks(tracks, session=session)


@router.post("/cart/new", response_model=CartOut)
def api_new_transaction() -> dict:
    get_camera_hub().reset_tracking()
    return get_cart().new_transaction()


@router.post("/checkout", response_model=CheckoutOut)
def api_checkout(session: Session = Depends(get_db)) -> dict:
    cart = get_cart()
    try:
        transaction = checkout(session, cart)
    except (CheckoutError, CartError) as extra:
        raise _cart_error(extra) from extra
    get_camera_hub().reset_tracking()
    session.refresh(transaction)
    return {
        "invoice_number": transaction.invoice_number,
        "transaction_id": transaction.id,
        "pdf_url": f"/invoices/{transaction.invoice_number}.pdf",
        "cart": cart.snapshot(session),
        "bill": transaction,
    }


@router.post("/bills/generate", response_model=CheckoutOut)
def api_generate_bill(session: Session = Depends(get_db)) -> dict:
    """Alias for checkout — bill totals always use database product prices."""
    return api_checkout(session)


@router.get("/transactions", response_model=list[TransactionOut])
def api_list_transactions(limit: int = Query(50, ge=1, le=200), session: Session = Depends(get_db)):
    return list_transactions(session, limit=limit)


@router.get("/transactions/{transaction_id}", response_model=TransactionOut)
def api_get_transaction(transaction_id: int, session: Session = Depends(get_db)):
    try:
        return get_transaction(session, transaction_id)
    except CheckoutError as extra:
        raise _cart_error(extra) from extra


@router.get("/invoices/{invoice_number}.pdf")
def api_get_invoice_pdf(invoice_number: str, session: Session = Depends(get_db)):
    try:
        transaction = get_transaction_by_invoice(session, invoice_number)
    except CheckoutError as extra:
        raise _cart_error(extra) from extra
    if not transaction.pdf_path:
        raise HTTPException(status_code=404, detail="Invoice PDF was not generated.")
    path = Path(transaction.pdf_path)
    if not path.is_absolute():
        path = ROOT_DIR / path
    if not path.exists():
        raise HTTPException(status_code=404, detail="Invoice PDF file is missing.")
    return FileResponse(path, media_type="application/pdf", filename=path.name)
