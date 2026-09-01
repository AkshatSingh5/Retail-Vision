import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

# Vercel sets VERCEL=1 on every serverless build and request.
ON_VERCEL = os.getenv("VERCEL") == "1"

PROJECT_NAME = "Retail Vision"

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./retail_vision.db")

# Prefer YOLO_MODEL_PATH; fall back to MODEL_PATH for older .env files.
_raw_model_path = Path(
    os.getenv("YOLO_MODEL_PATH") or os.getenv("MODEL_PATH", "vision/models/yolo26m.pt")
)
MODEL_PATH = str(_raw_model_path if _raw_model_path.is_absolute() else ROOT_DIR / _raw_model_path)

CONFIDENCE_THRESHOLD = float(
    os.getenv("YOLO_CONFIDENCE_THRESHOLD") or os.getenv("CONFIDENCE_THRESHOLD", "0.50")
)
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "cuda").strip()
DINO_DEVICE = os.getenv("DINO_DEVICE", "cuda").strip()
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
WINDOW_NAME = os.getenv("WINDOW_NAME", "Retail Vision - YOLO26m")
STABLE_FRAMES = int(os.getenv("STABLE_FRAMES", "5"))
TRACK_MAX_MISSING = int(os.getenv("TRACK_MAX_MISSING", "20"))
TRACK_IOU_THRESHOLD = float(os.getenv("TRACK_IOU_THRESHOLD", "0.30"))
ENABLE_EMBEDDING_REFINEMENT = os.getenv("ENABLE_EMBEDDING_REFINEMENT", "false").lower() in {"1", "true", "yes"}
USE_BYTETRACK = os.getenv("TRACKER", "bytetrack").lower() != "iou"
DISCOUNT_PERCENT = float(os.getenv("DISCOUNT_PERCENT", "0"))
STORE_NAME = os.getenv("STORE_NAME", "Retail Vision")
STORE_ADDRESS = os.getenv("STORE_ADDRESS", "AI Checkout Counter")
_raw_invoice_dir = Path(os.getenv("INVOICE_DIR", "invoices"))
INVOICE_DIR = _raw_invoice_dir if _raw_invoice_dir.is_absolute() else ROOT_DIR / _raw_invoice_dir
MIN_CART_CONFIDENCE = float(os.getenv("MIN_CART_CONFIDENCE", str(CONFIDENCE_THRESHOLD)))
DETECT_CONFIDENCE = float(os.getenv("DETECT_CONFIDENCE", "0.15"))
GALLERY_MATCH_THRESHOLD = float(os.getenv("GALLERY_MATCH_THRESHOLD", "0.86"))
# Visual match gates. PRODUCT_MATCH_* is canonical; SCAN_MATCH_* / DINO_SIMILARITY_THRESHOLD remain aliases.
PRODUCT_MATCH_THRESHOLD = float(
    os.getenv("DINO_SIMILARITY_THRESHOLD")
    or os.getenv("PRODUCT_MATCH_THRESHOLD")
    or os.getenv("SCAN_MATCH_THRESHOLD", "0.82")
)
PRODUCT_MATCH_MARGIN = float(
    os.getenv("PRODUCT_MATCH_MARGIN") or os.getenv("SCAN_MATCH_MARGIN", "0.04")
)
SCAN_MATCH_THRESHOLD = PRODUCT_MATCH_THRESHOLD
SCAN_MATCH_MARGIN = PRODUCT_MATCH_MARGIN
# Visual duplicate warning when registering a new product (not a silent block).
DUPLICATE_MATCH_THRESHOLD = float(os.getenv("DUPLICATE_MATCH_THRESHOLD", "0.93"))
# dinov2 (production) or color (fast tests / fallback).
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "dinov2").strip().lower()
DINOV2_MODEL = os.getenv("DINOV2_MODEL", "facebook/dinov2-small").strip()
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))
# When true, load DINOv2 during API startup (slower boot, faster first scan).
PRELOAD_DINOV2 = os.getenv("PRELOAD_DINOV2", "true").lower() in {"1", "true", "yes"}
# When true, load YOLO26m during API startup (required for GPU scan path).
PRELOAD_YOLO = os.getenv("PRELOAD_YOLO", "true").lower() in {"1", "true", "yes"}
FLORENCE_MODEL = os.getenv("FLORENCE_MODEL", "microsoft/Florence-2-base").strip()
# When true, load Florence-2 during API startup (slower boot, faster first caption).
PRELOAD_FLORENCE = os.getenv("PRELOAD_FLORENCE", "false").lower() in {"1", "true", "yes"}
VECTOR_SEARCH_TOP_K = int(os.getenv("VECTOR_SEARCH_TOP_K", "40"))
PGVECTOR_INDEX = os.getenv("PGVECTOR_INDEX", "hnsw").strip().lower()
# Image quality gates (OpenCV). Calibrate on real camera captures.
MIN_IMAGE_SIDE = int(os.getenv("MIN_IMAGE_SIDE", "64"))
BLUR_VARIANCE_MIN = float(os.getenv("BLUR_VARIANCE_MIN", "45"))
MIN_BRIGHTNESS = float(os.getenv("MIN_BRIGHTNESS", "18"))
MAX_BRIGHTNESS = float(os.getenv("MAX_BRIGHTNESS", "245"))
MIN_CONTRAST_STD = float(os.getenv("MIN_CONTRAST_STD", "8"))
# When true, write original / bbox / crop JPEGs under storage/debug/scans/.
SCAN_DEBUG_CROPS = os.getenv("SCAN_DEBUG_CROPS", "false").lower() in {"1", "true", "yes"}
# Include top-match diagnostics in scan API responses (dev / local debugging).
RECOGNITION_DEBUG = os.getenv("RECOGNITION_DEBUG", "false").lower() in {"1", "true", "yes"}
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))
ALLOWED_IMAGE_MIME = {
    item.strip().lower()
    for item in os.getenv("ALLOWED_IMAGE_MIME", "image/jpeg,image/png,image/webp,image/jpg").split(",")
    if item.strip()
}
# When true, opening the camera also starts continuous YOLO cart updates.
# Default false: preview only; user clicks Scan Product for one-shot recognition.
AUTO_DETECT_ON_CAMERA = os.getenv("AUTO_DETECT_ON_CAMERA", "false").lower() in {"1", "true", "yes"}
_raw_product_dir = Path(os.getenv("PRODUCT_IMAGE_DIR", "products/images"))
PRODUCT_IMAGE_DIR = _raw_product_dir if _raw_product_dir.is_absolute() else ROOT_DIR / _raw_product_dir
_raw_storage_dir = Path(os.getenv("STORAGE_DIR", "storage"))
STORAGE_DIR = _raw_storage_dir if _raw_storage_dir.is_absolute() else ROOT_DIR / _raw_storage_dir
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "local").strip().lower()
S3_BUCKET = os.getenv("S3_BUCKET", "").strip()
AWS_REGION = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1")).strip()
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "").strip() or None
S3_PREFIX = os.getenv("S3_PREFIX", "products").strip().strip("/")
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "0"))
_raw_imgsz = os.getenv("INFER_IMGSZ", "").strip()
INFER_IMGSZ = int(_raw_imgsz) if _raw_imgsz.isdigit() and int(_raw_imgsz) > 0 else None

# Serverless has no NVIDIA GPU and a read-only project filesystem.
# Skip model preload and write SQLite/invoices/images under /tmp.
if ON_VERCEL:
    YOLO_DEVICE = "cpu"
    DINO_DEVICE = "cpu"
    PRELOAD_YOLO = False
    PRELOAD_DINOV2 = False
    PRELOAD_FLORENCE = False
    _vercel_tmp = Path("/tmp/retail-vision")
    INVOICE_DIR = _vercel_tmp / "invoices"
    PRODUCT_IMAGE_DIR = _vercel_tmp / "products" / "images"
    STORAGE_DIR = _vercel_tmp / "storage"
