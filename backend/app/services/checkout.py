from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from backend.app.config import INVOICE_DIR, ROOT_DIR
from backend.app.models.transaction import Transaction, TransactionItem
from backend.app.services.cart_service import Cart, CartError
from backend.app.services.invoice import build_invoice_pdf
from backend.app.utils.money import money


class CheckoutError(CartError):
    pass


def next_invoice_number(session: Session, when: datetime | None = None) -> str:
    moment = when or datetime.now(timezone.utc)
    prefix = f"RV-{moment.strftime('%Y%m%d')}-"
    latest = session.scalar(
        select(func.max(Transaction.invoice_number)).where(Transaction.invoice_number.like(f"{prefix}%"))
    )
    sequence = 1
    if latest:
        try:
            sequence = int(str(latest).split("-")[-1]) + 1
        except ValueError:
            sequence = 1
    return f"{prefix}{sequence:04d}"


def checkout(
    session: Session,
    cart: Cart,
    invoice_dir: Path | None = None,
) -> Transaction:
    snapshot = cart.snapshot(session)
    if not snapshot["items"]:
        raise CheckoutError("Cart is empty.")

    invoice_number = next_invoice_number(session)
    created = datetime.now(timezone.utc)
    transaction = Transaction(
        invoice_number=invoice_number,
        status="completed",
        subtotal=money(snapshot["subtotal"]),
        tax=money(snapshot["tax"]),
        discount=money(snapshot["discount"]),
        discount_percent=money(snapshot["discount_percent"]),
        grand_total=money(snapshot["grand_total"]),
        created_at=created,
    )
    for item in snapshot["items"]:
        transaction.items.append(
            TransactionItem(
                product_id=item["product_id"],
                sku=item["sku"],
                name=item["name"],
                quantity=int(item["quantity"]),
                unit_price=money(item["unit_price"]),
                tax_rate=money(item.get("tax_rate") or 0),
                tax=money(item["tax"]),
                total=money(item["total"]),
            )
        )
    session.add(transaction)
    session.flush()

    directory = Path(invoice_dir) if invoice_dir is not None else INVOICE_DIR
    if not directory.is_absolute():
        directory = ROOT_DIR / directory
    pdf_path = directory / f"{invoice_number}.pdf"
    build_invoice_pdf(transaction, transaction.items, pdf_path)
    try:
        transaction.pdf_path = str(pdf_path.relative_to(ROOT_DIR))
    except ValueError:
        transaction.pdf_path = str(pdf_path)
    session.flush()

    cart.new_transaction()
    cart.last_invoice_number = invoice_number
    return transaction


def get_transaction(session: Session, transaction_id: int) -> Transaction:
    transaction = session.scalar(
        select(Transaction)
        .options(selectinload(Transaction.items))
        .where(Transaction.id == transaction_id)
    )
    if transaction is None:
        raise CheckoutError(f"Transaction {transaction_id} was not found.")
    return transaction


def get_transaction_by_invoice(session: Session, invoice_number: str) -> Transaction:
    transaction = session.scalar(
        select(Transaction)
        .options(selectinload(Transaction.items))
        .where(Transaction.invoice_number == invoice_number)
    )
    if transaction is None:
        raise CheckoutError(f"Invoice {invoice_number} was not found.")
    return transaction


def list_transactions(session: Session, limit: int = 50) -> list[Transaction]:
    statement = (
        select(Transaction)
        .options(selectinload(Transaction.items))
        .order_by(Transaction.id.desc())
        .limit(limit)
    )
    return list(session.scalars(statement))
