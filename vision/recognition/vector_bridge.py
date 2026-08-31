"""Shared embedding serialization helpers."""

from __future__ import annotations

import json

from backend.app.config import DINOV2_MODEL, EMBEDDING_BACKEND


def current_model_name() -> str:
    backend = (EMBEDDING_BACKEND or "dinov2").strip().lower()
    if backend == "dinov2":
        # facebook/dinov2-small → dinov2-small
        return DINOV2_MODEL.split("/")[-1] if "/" in DINOV2_MODEL else DINOV2_MODEL
    return "v3_color_layout"


def embedding_to_json(values: list[float], model_name: str | None = None) -> str:
    name = model_name or current_model_name()
    return json.dumps({"v": name, "model": name, "vec": values})


def embedding_from_json(raw: str | None, *, require_model: str | None = None) -> list[float] | None:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        model = parsed.get("model") or parsed.get("v")
        if require_model and model != require_model:
            return None
        values = parsed.get("vec")
    elif isinstance(parsed, list):
        if require_model:
            return None
        values = parsed
    else:
        return None
    if not isinstance(values, list) or not values:
        return None
    return [float(item) for item in values]
