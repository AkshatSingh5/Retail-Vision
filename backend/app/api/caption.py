"""Image upload → Florence-2 detailed caption / prompt."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from backend.app.schemas.caption import CaptionOut
from backend.app.services.scan_service import ScanError, validate_image_bytes

router = APIRouter(tags=["caption"])


@router.post("/caption", response_model=CaptionOut)
async def api_generate_caption(image: UploadFile = File(...)) -> CaptionOut:
    """Upload an image and return a Florence-2 <MORE_DETAILED_CAPTION> prompt."""
    image_bytes = await image.read()
    try:
        validate_image_bytes(image_bytes, content_type=image.content_type)
    except ScanError as extra:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(extra)) from extra

    try:
        from vision.caption.florence import generate_caption, get_florence

        prompt = await asyncio.to_thread(generate_caption, image_bytes)
        captioner = get_florence()
    except HTTPException:
        raise
    except Exception as extra:
        print(f"[Florence-2] Caption failed: {extra}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Image-to-text is unavailable. Check that Florence-2 can be downloaded.",
        ) from extra

    if not prompt:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Florence-2 returned an empty prompt.",
        )
    return CaptionOut(prompt=prompt, model=captioner.model_id, device=captioner.device)
