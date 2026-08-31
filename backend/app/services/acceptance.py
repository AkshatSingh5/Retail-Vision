from __future__ import annotations

from decimal import Decimal

from backend.app.config import MIN_CART_CONFIDENCE

UNKNOWN_SKU_PREFIXES = ("unknown", "UNKNOWN")

MESSAGES = {
    "low_confidence": "UNKNOWN PRODUCT\nConfidence too low.\nPlease verify manually.",
    "not_in_database": "UNKNOWN PRODUCT\nProduct not found in database.\nPlease verify manually.",
    "invalid_price": "UNKNOWN PRODUCT\nInvalid product price.\nPlease verify manually.",
    "database_unavailable": "Database unavailable.\nPlease verify manually.",
    "model_unavailable": "Model unavailable.\nPlease verify manually.",
    "camera_disconnected": "Camera disconnected.\nPlease verify manually.",
}


def _positive_price(value) -> bool:
    if value is None:
        return False
    try:
        return Decimal(str(value)) > 0
    except Exception:
        return False


def evaluate_for_cart(track: dict, *, in_database: bool | None = None) -> dict:
    """Decide whether a tracked detection may enter the bill.

    Prices never come from the detector. Missing catalog rows, weak scores,
    and invalid prices are rejected instead of being added silently.
    """
    sku = str(track.get("sku") or "")
    name = str(track.get("name") or track.get("product_name") or "")
    price = track.get("price")
    unknown = sku.lower().startswith("unknown") or name.lower().startswith("unknown")
    missing = in_database is False or unknown or track.get("product_id") in {None, 0}

    if not track.get("confirmed", True):
        return {"accepted": False, "reason": "unconfirmed", "message": None, "silent": True}
    if "confidence" in track and float(track.get("confidence") or 0.0) < MIN_CART_CONFIDENCE:
        return {"accepted": False, "reason": "low_confidence", "message": MESSAGES["low_confidence"], "silent": False}
    if missing:
        return {
            "accepted": False,
            "reason": "not_in_database",
            "message": MESSAGES["not_in_database"],
            "silent": False,
        }
    if in_database is not True and not _positive_price(price):
        return {
            "accepted": False,
            "reason": "invalid_price",
            "message": MESSAGES["invalid_price"],
            "silent": False,
        }
    return {"accepted": True, "reason": None, "message": None, "silent": True}
