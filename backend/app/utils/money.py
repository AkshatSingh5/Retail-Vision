from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

TWOPLACE = Decimal("0.01")
HUNDRED = Decimal("100")


def money(value: Decimal | float | int | str) -> Decimal:
    return Decimal(str(value)).quantize(TWOPLACE, rounding=ROUND_HALF_UP)


def money_json(value: Decimal | float | int | None) -> int | float | None:
    if value is None:
        return None
    amount = money(value)
    if amount == amount.to_integral_value():
        return int(amount)
    return float(amount)


def line_subtotal(unit_price: Decimal, quantity: int) -> Decimal:
    return money(money(unit_price) * int(quantity))


def line_tax(unit_price: Decimal, quantity: int, tax_rate: Decimal) -> Decimal:
    """GST amount from the product's stored tax_rate (percent), not a hard-coded rule."""
    return money(line_subtotal(unit_price, quantity) * money(tax_rate) / HUNDRED)


def line_total(unit_price: Decimal, quantity: int, tax_rate: Decimal) -> Decimal:
    return money(line_subtotal(unit_price, quantity) + line_tax(unit_price, quantity, tax_rate))
