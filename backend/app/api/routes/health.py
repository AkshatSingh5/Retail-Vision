"""Liveness probe. Must not load YOLO, DINOv2, or Florence-2."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
