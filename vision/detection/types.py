from __future__ import annotations

from typing import NotRequired, TypedDict


class Detection(TypedDict):
    class_id: int
    confidence: float
    bbox: list[float]
    track_id: NotRequired[int | None]
