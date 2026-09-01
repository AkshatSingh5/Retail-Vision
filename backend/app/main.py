from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _APP_DIR.parent
_ROOT_DIR = _BACKEND_DIR.parent
for _path in (str(_ROOT_DIR), str(_BACKEND_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from backend.app.api.caption import router as caption_router
from backend.app.api.cart import router as cart_router
from backend.app.api.pos import router as pos_router
from backend.app.api.products import router as products_router
from backend.app.api.routes.health import router as health_router
from backend.app.config import (
    FRONTEND_DIR,
    INVOICE_DIR,
    PRELOAD_DINOV2,
    PRELOAD_FLORENCE,
    PRELOAD_YOLO,
    PROJECT_NAME,
    ROOT_DIR,
    cors_allow_origins,
)
from backend.app.database import get_session_factory, init_db
from backend.app.services.seed import seed_products_from_registry

STATIC_DIR = ROOT_DIR / "backend" / "app" / "static"
POS_INDEX = FRONTEND_DIR / "index.html"
if not POS_INDEX.is_file():
    POS_INDEX = STATIC_DIR / "pos" / "index.html"
FAVICON = FRONTEND_DIR / "src" / "favicon.png"
if not FAVICON.is_file():
    FAVICON = STATIC_DIR / "pos" / "favicon.png"

_runtime: dict[str, str | bool | None] = {
    "yolo": "not_loaded",
    "dinov2": "not_loaded",
    "yolo_error": None,
    "dinov2_error": None,
}


def _print_backend_banner() -> None:
    from vision.device import cuda_available, device_banner_name, gpu_name, resolve_device, torch_version

    device = resolve_device("cuda", warn=False)
    cuda = cuda_available()
    gpu = gpu_name() or "n/a"
    print("========================================")
    print("Retail Vision Backend")
    print("========================================")
    print(f"Device: {device_banner_name(device)}")
    print(f"GPU: {gpu}")
    print(f"CUDA: {cuda}")
    print(f"PyTorch: {torch_version() or 'not installed'}")
    if torch_version() is None:
        print("WARNING: PyTorch not installed. Vision scan uses color embeddings only.")
    elif not cuda:
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
    try:
        init_db()
        try:
            INVOICE_DIR.mkdir(parents=True, exist_ok=True)
        except OSError:
            Path("/tmp/retail-vision/invoices").mkdir(parents=True, exist_ok=True)
        session = get_session_factory()()
        try:
            seed_products_from_registry(session)
            session.commit()
        except Exception as extra:
            session.rollback()
            print(f"[startup] catalog seed skipped: {extra}")
        finally:
            session.close()
    except Exception as extra:
        print(f"[startup] database init skipped: {extra}")

    if PRELOAD_YOLO:
        try:
            from vision.detection.yolo_detector import get_yolo_detector

            detector = get_yolo_detector()
            _runtime["yolo"] = f"Loaded ({detector.device})"
            print(f"YOLO device: {detector.device}")
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
            print(f"DINO device: {embedder.device}")
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

    try:
        _print_backend_banner()
    except Exception as extra:
        print(f"[startup] banner skipped: {extra}")
    yield


app = FastAPI(title=PROJECT_NAME, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allow_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
app.include_router(health_router)
app.include_router(products_router)
app.include_router(cart_router)
app.include_router(pos_router)
app.include_router(caption_router)
app.include_router(products_router, prefix="/api")
app.include_router(cart_router, prefix="/api")
app.include_router(caption_router, prefix="/api")

frontend_src = FRONTEND_DIR / "src"
if frontend_src.is_dir():
    app.mount("/src", StaticFiles(directory=frontend_src), name="frontend_src")
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def root():
    if POS_INDEX.is_file():
        return FileResponse(POS_INDEX, media_type="text/html")
    return {"project": PROJECT_NAME, "status": "running", "ui": "not_bundled"}


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    if FAVICON.is_file():
        return FileResponse(FAVICON, media_type="image/png")
    return Response(status_code=204)


@app.get("/api/health")
def api_health() -> dict:
    """Diagnostic status. Does not load YOLO/DINOv2; reports already-loaded runtime."""
    return {
        "project": PROJECT_NAME,
        "status": "ok",
        "yolo26m": _runtime["yolo"],
        "dinov2": _runtime["dinov2"],
    }
