from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.api.caption import router as caption_router
from backend.app.api.cart import router as cart_router
from backend.app.api.pos import router as pos_router
from backend.app.api.products import router as products_router
from backend.app.config import (
    INVOICE_DIR,
    PRELOAD_DINOV2,
    PRELOAD_FLORENCE,
    PRELOAD_YOLO,
    PROJECT_NAME,
    ROOT_DIR,
)
from backend.app.database import get_session_factory, init_db
from backend.app.services.seed import seed_products_from_registry

STATIC_DIR = ROOT_DIR / "backend" / "app" / "static"
POS_INDEX = STATIC_DIR / "pos" / "index.html"
FAVICON = STATIC_DIR / "pos" / "favicon.png"

_runtime: dict[str, str | bool | None] = {
    "yolo": "not_loaded",
    "dinov2": "not_loaded",
    "yolo_error": None,
    "dinov2_error": None,
}


def _print_backend_banner() -> None:
    import torch

    from vision.device import cuda_available, device_banner_name, gpu_name, resolve_device

    device = resolve_device("cuda", warn=False)
    cuda = cuda_available()
    gpu = gpu_name() or "n/a"
    print("========================================")
    print("Retail Vision Backend")
    print("========================================")
    print(f"Device: {device_banner_name(device)}")
    print(f"GPU: {gpu}")
    print(f"CUDA: {cuda}")
    print(f"PyTorch: {torch.__version__}")
    if not cuda:
        print("WARNING: CUDA unavailable. Running on CPU.")
    print()
    print(f"YOLO26m: {_runtime['yolo']}")
    if _runtime["yolo_error"]:
        print(f"  error: {_runtime['yolo_error']}")
    print(f"DINOv2: {_runtime['dinov2']}")
    if _runtime["dinov2_error"]:
        print(f"  error: {_runtime['dinov2_error']}")
    print()
    print("Backend ready")
    print("========================================")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    INVOICE_DIR.mkdir(parents=True, exist_ok=True)
    session = get_session_factory()()
    try:
        seed_products_from_registry(session)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    if PRELOAD_YOLO:
        try:
            from vision.detection.yolo_detector import get_yolo_detector

            detector = get_yolo_detector()
            _runtime["yolo"] = f"Loaded ({detector.device})"
            print("YOLO26m: Loaded")
        except Exception as extra:
            _runtime["yolo"] = "FAILED"
            _runtime["yolo_error"] = str(extra)
            print(f"[YOLO26m] Startup load failed: {extra}")

    if PRELOAD_DINOV2:
        try:
            from vision.recognition.dinov2 import get_dinov2

            embedder = get_dinov2()
            _runtime["dinov2"] = f"Loaded ({embedder.device})"
            print("DINOv2: Loaded")
        except Exception as extra:
            _runtime["dinov2"] = "FAILED"
            _runtime["dinov2_error"] = str(extra)
            print(f"[DINOv2] Startup preload failed: {extra}")
    if PRELOAD_FLORENCE:
        try:
            from vision.caption.florence import get_florence

            get_florence()
        except Exception as extra:
            print(f"[Florence-2] Startup preload failed: {extra}")

    _print_backend_banner()
    yield


app = FastAPI(title=PROJECT_NAME, lifespan=lifespan)
# Existing POS paths (no /api prefix) — keep for the UI.
app.include_router(products_router)
app.include_router(cart_router)
app.include_router(pos_router)
app.include_router(caption_router)
# Spec aliases: /api/products/*, /api/cart/*, /api/bills/generate, /api/caption
app.include_router(products_router, prefix="/api")
app.include_router(cart_router, prefix="/api")
app.include_router(caption_router, prefix="/api")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def root() -> FileResponse:
    return FileResponse(POS_INDEX, media_type="text/html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    return FileResponse(FAVICON, media_type="image/png")


@app.get("/api/health")
def health() -> dict:
    import torch

    from vision.device import cuda_available, gpu_name, resolve_device

    device = resolve_device("cuda", warn=False)
    return {
        "project": PROJECT_NAME,
        "status": "running",
        "device": device,
        "cuda": cuda_available(),
        "gpu": gpu_name(),
        "pytorch": torch.__version__,
        "yolo26m": _runtime["yolo"],
        "dinov2": _runtime["dinov2"],
    }
