# Retail Vision

AI-powered retail product detection and automatic billing.

Retail Vision will use computer vision to detect products through a camera, identify them, look up prices, calculate quantities, and generate an automatic bill.

**Current phase:** Phase 8 — testing, R&D, and production-readiness evaluation. The system is **not production-ready**.

---

## Project objective

Build a scalable system that:

1. Captures video from a camera
2. Detects retail products with **YOLO26m**
3. Matches detections to a product catalog
4. Calculates quantities and totals
5. Generates an invoice

This repository currently provides a live POS checkout: **YOLO detection → DINOv2 visual embeddings → PostgreSQL/pgvector similarity → cart → PDF bill**.

Identity is **visual only** (no OCR, barcode, or QR). YOLO is used only to detect and crop a single product; final SKU identity comes from DINOv2 + vector search.

---

## Technology stack

| Area | Choice |
| --- | --- |
| Language | Python 3.12 |
| Object detection | Ultralytics YOLO (`yolo26m.pt` / retail weights) — **crop only** |
| Visual recognition | DINOv2 (`facebook/dinov2-small`) embeddings |
| Vector search | PostgreSQL + pgvector (JSON cosine fallback on SQLite / if extension missing) |
| Vision I/O | OpenCV, NumPy, Pillow |
| Backend API | FastAPI, Uvicorn, Pydantic |
| Database | SQLAlchemy; PostgreSQL preferred (`psycopg2-binary`), SQLite for tests |
| Invoices | ReportLab (PDF) |
| Configuration | `.env` loaded with `python-dotenv` |
| Image storage | Local filesystem (dev) or S3 (`STORAGE_BACKEND=s3`) |

### YOLO26m (verified)

YOLO26m is **not** a separate pip package. It is provided by the official **`ultralytics`** package.

| Item | Value |
| --- | --- |
| Official package | `ultralytics` **8.4.123** |
| Model identifier | `yolo26m.pt` |
| Recommended install | `pip install ultralytics` |
| Documentation | [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26) |

Older YOLO packages (YOLOv5, YOLOv8-only pins, etc.) do **not** automatically include YOLO26m. Use a current `ultralytics` release (8.4.x or newer) that lists `yolo26m.pt` among its official assets.

---

## Installation

### Prerequisites

- Python 3.12+
- pip
- Git
- Windows, Linux, or macOS

### Virtual environment setup

From the project root (`Retail Vision/`):

```bash
python -m venv .venv
```

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**

```bat
.venv\Scripts\activate.bat
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

Verify:

```bash
python --version
pip --version
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

### Package installation

```bash
pip install -r requirements.txt
```

Copy `.env` values as needed. Never commit secrets. The provided `.env` uses SQLite and vision settings:

```text
DATABASE_URL=sqlite:///./retail_vision.db
MODEL_PATH=vision/models/retail_yolo26m_v2.pt
CONFIDENCE_THRESHOLD=0.50
IOU_THRESHOLD=0.45
CAMERA_INDEX=0
STABLE_FRAMES=5
TRACK_MAX_MISSING=20
TRACK_IOU_THRESHOLD=0.30
ENABLE_EMBEDDING_REFINEMENT=false
TRACKER=bytetrack
DISCOUNT_PERCENT=0
STORE_NAME=Retail Vision
STORE_ADDRESS=AI Checkout Counter
INVOICE_DIR=invoices
```

`CONFIDENCE_THRESHOLD` and `STABLE_FRAMES` are read at runtime. Do not hard-code them. If `yolo26m.pt` is missing and `MODEL_PATH` still points at it, the detector downloads the official pretrained weights into `vision/models/`.

---

## How to run the backend

From the project root, with the virtual environment activated:

```bash
uvicorn backend.app.main:app --reload
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) for the POS checkout.

Health check: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

```json
{
  "project": "Retail Vision",
  "status": "running"
}
```

Interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

On startup the API creates the `products` table (SQLite by default) and seeds it from `products/registry.yaml` if it is empty. Prices live in the database. Switch to PostgreSQL later with:

```text
DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/retail_vision
```

Product catalog:

```text
GET    /products
GET    /products/{id}
GET    /products/sku/{sku}
GET    /products/class/{class_id}
POST   /products
PUT    /products/{id}
DELETE /products/{id}
```

`GET /products/class/0` returns the detection price mapping:

```json
{
  "product_id": 1,
  "sku": "COKE500",
  "name": "Coca-Cola 500ml",
  "price": 40,
  "tax_rate": 18
}
```

Seed or re-seed from the registry (skipped when rows already exist):

```powershell
python scripts/seed_products.py
python scripts/test_products.py
python scripts/test_cart.py
python scripts/run_phase8.py
```

Phase 8 report: `reports/phase8_rnd_report.md`. Uncertain detections are **not** added to the bill; the POS shows `UNKNOWN PRODUCT` / `Please verify manually`.

---

```text
GET    /cart
POST   /cart/items
POST   /cart/items/{id}/increase
POST   /cart/items/{id}/decrease
POST   /cart/items/{id}/confirm
DELETE /cart/items/{id}
POST   /cart/clear
POST   /cart/new
POST   /checkout
GET    /transactions
GET    /invoices/{invoice_number}.pdf
```

Tax/GST is taken from each product's `tax_rate` in the database. `DISCOUNT_PERCENT` is configurable in `.env` (default 0).

POS shortcut:

```powershell
python scripts/run_pos.py
```

---

## How to run the detector

From the project root, with the virtual environment activated:

```powershell
python -m vision.inference
```

Equivalent:

```powershell
python scripts/run_detector.py
```

The live window shows:

```text
Camera Feed
     ↓
YOLO26m
     ↓
Detected Objects
     ↓
Bounding Boxes
     ↓
Confidence + FPS
```

Each box is labeled with the COCO class name, confidence, and class ID, for example:

```text
bottle
Confidence: 0.94
Class ID: 39
```

The overlay also shows FPS and inference latency. Press **Q** or **ESC** to quit.

If the webcam fails to open, set `CAMERA_INDEX` in `.env` to the correct device (often `0` or `1`).

This phase uses the custom-trained detector when `MODEL_PATH` points at `vision/models/retail_yolo26m_v2.pt`.

---

## How to run product tracking

From the project root:

```powershell
python scripts/run_tracker.py
```

Pipeline:

```text
Camera
 ↓
YOLO26m
 ↓
class_id → product_id → SKU → name → price
 ↓
ByteTrack / IoU tracker
 ↓
Stable tracks + cart quantities
```

Each confirmed object is printed as JSON. When the database is available, price mapping is included:

```json
{
  "product_id": 1,
  "sku": "COKE500",
  "name": "Coca-Cola 500ml",
  "price": 40,
  "tax_rate": 18
}
```

The overlay shows `track_id`, product name, entry/visible/leaving state, and a cart that counts **unique tracks**, not frames. Two Coke bottles become Track 1 and Track 2 (`Coke × 2`). One bottle visible for 100 frames stays Track 1 (`Coke × 1`).

A product must be detected for `STABLE_FRAMES` consecutive frames (default 5) before it is added to the cart. Set `TRACKER=iou` in `.env` to use the IoU tracker instead of Ultralytics ByteTrack.

Tests (no camera required):

```powershell
python scripts/test_tracking.py
python scripts/test_yolo_tracking.py
python scripts/benchmark_identity.py
python scripts/test_products.py
```

---

## Custom product dataset

Eight retail SKUs are registered in `products/registry.yaml` and `data.yaml`:

| class_id | DB id | SKU | Product | Price |
| --- | --- | --- | --- | --- |
| 0 | 1 | COKE500 | Coca-Cola 500ml | 40 |
| 1 | 2 | LAYSCLASSIC | Lays Classic | 20 |
| 2 | 3 | MAGGI | Maggi Noodles | 14 |
| 3 | 4 | DAIRYMILK | Dairy Milk | 45 |
| 4 | 5 | PEPSI500 | Pepsi 500ml | 40 |
| 5 | 6 | SPRITE500 | Sprite 500ml | 40 |
| 6 | 7 | KITKAT | KitKat | 30 |
| 7 | 8 | KURKURE | Kurkure Masala Munch | 20 |

The long-term target is **1000+ real visual variations per product** (angles, lighting, distance, occlusion, camera height, and mixed scenes). This phase starts the first training experiment with a **real-photo seed set**, not an augmentation-only set.

### Dataset layout

```text
dataset/
├── data.yaml
├── images/{train,val,test}/
├── labels/{train,val,test}/
└── raw/                      # session-grouped captures before split
```

Labels use YOLO format, one file per image:

```text
class_id x_center y_center width height
```

All box coordinates are normalized to `[0, 1]`.

The split is **70% / 20% / 10% by capture session** (barcode or photo source), not by shuffling frames. Photos of the same pack never appear in both train and test.

### Dataset commands

From the project root, with `.venv` activated:

```powershell
python scripts/capture_products.py
python scripts/annotate_products.py
python scripts/collect_seed_images.py
python scripts/auto_label_raw.py
python scripts/split_dataset.py
python scripts/validate_dataset.py
python scripts/visualize_dataset.py --show
```

Webcam capture keys: `0-7` select class, `n` starts a new session, `SPACE` saves, `q` quits.

Optional **train-only** extras (never written to val/test):

```powershell
python scripts/augment_train.py
```

Use this after more real photos exist. Do not rely on it to invent the dataset.

---

## Train YOLO26m

From the project root:

```powershell
python scripts/train.py
```

Useful flags:

```text
--model     pretrained weights (default: vision/models/yolo26m.pt)
--data      dataset YAML (default: data.yaml)
--imgsz     image size
--batch     batch size
--epochs    epochs
--device    0 for CUDA, cpu otherwise (auto-detected if omitted)
--workers   dataloader workers (auto: 0 on Windows)
--project   output root (default: runs/)
--name      experiment name (auto: retail_yolo26m_v1, v2, ...)
```

Runs are never overwritten. Each experiment gets a new folder:

```text
runs/retail_yolo26m_v1/
runs/retail_yolo26m_v2/
```

CPU-friendly first-run example used on this machine:

```powershell
python scripts/train.py --name retail_yolo26m_v2 --imgsz 416 --epochs 20 --batch 8 --patience 20 --mosaic 0.0 --mixup 0.0 --freeze 10
```

Augmentation (rotation, scale, translate, perspective, HSV brightness/contrast, mosaic/mixup, random erase for occlusion) is configured in `scripts/train.py`. Shear is left at 0 to avoid unrealistic warps. On a 48-image seed set, mosaic/mixup caused v1 to stall; v2 disables them until more real photos exist.

### Current best model

| Item | Path |
| --- | --- |
| Serving weights | `vision/models/retail_yolo26m_v2.pt` |
| Training run | `runs/retail_yolo26m_v2/` |
| Config | `runs/retail_yolo26m_v2/train_config.json` |
| Metrics | `runs/retail_yolo26m_v2/evaluation.md` |
| Robustness | `runs/retail_yolo26m_v2/robustness.json` |
| `.env` | `MODEL_PATH=vision/models/retail_yolo26m_v2.pt` |

Base pretrained checkpoint remains at `vision/models/yolo26m.pt`.

### Evaluate

```powershell
python scripts/evaluate.py --run runs/retail_yolo26m_v2 --imgsz 416
python scripts/evaluate_robustness.py --weights vision/models/retail_yolo26m_v2.pt --imgsz 416
python scripts/test_realtime.py
```

`evaluate_robustness.py` applies 120 lighting/angle/scale variants to each held-out test image (1080 checks) to measure recognition under many views. That is not the same as adding 1000 fake photos to training.

### v2 test-set metrics

| Metric | Value |
| --- | --- |
| Precision | 0.644 |
| Recall | 0.375 |
| F1 | 0.474 |
| mAP50 | 0.463 |
| mAP50-95 | 0.221 |

Best test classes: Coca-Cola 500ml (F1 0.79), Dairy Milk (0.45), Lays Classic (0.36). Weakest: Maggi, Pepsi, Sprite, KitKat, Kurkure (need more real photos).

Confusion on the 9-image test split: Maggi Noodles → Kurkure Masala Munch (1). No Coke↔Pepsi or Lays↔Kurkure swaps in that split. Most errors are **misses** (false negatives), not look-alike swaps.

1000+ variation protocol: **7 / 1080** correct class hits (0.65%). The dedicated test set is still too small for 1000 real views per SKU. Capture more webcam photos with `scripts/capture_products.py` before production.

Webcam smoke test (8 frames, no catalog products in view): camera opened, ~160–300 ms/frame on CPU, 0 detections at `CONFIDENCE_THRESHOLD=0.50`.

v1 (`runs/retail_yolo26m_v1`) early-stopped at epoch 1 (mosaic on 48 images). Keep it for comparison; use v2.

---

## Project structure

```text
Retail-Vision/
├── .venv/                      # Local virtual environment (not committed)
├── backend/
│   ├── app/
│   │   ├── main.py             # FastAPI entrypoint
│   │   ├── config.py           # Environment-based settings
│   │   ├── api/                # HTTP routes (later)
│   │   ├── services/           # Business logic (later)
│   │   ├── models/             # Database models (later)
│   │   ├── schemas/            # Request/response schemas (later)
│   │   └── utils/
│   └── tests/
├── vision/
│   ├── models/                 # yolo26m.pt and retail_yolo26m_v2.pt
│   ├── inference/              # Camera capture and live loop
│   ├── detection/              # YOLO26m detector
│   ├── tracking/               # Object tracking (later)
│   └── preprocessing/          # Frame preprocessing (later)
├── dataset/
│   ├── data.yaml
│   ├── images/{train,val,test}/
│   ├── labels/{train,val,test}/
│   └── raw/                    # session-grouped photos before split
├── products/
│   └── registry.yaml           # class IDs and product names
├── invoices/                   # Generated PDFs (gitignored)
├── reports/                    # Phase 8 R&D report
├── scripts/
├── runs/                       # Training experiments (gitignored)
├── logs/                       # Runtime logs (gitignored)
├── tests/
├── .gitignore
├── requirements.txt
├── README.md
└── data.yaml                   # YOLO dataset config
```

---

## Current development phase

**Phase 8 — Testing, R&D, optimization, and production readiness**

Completed:

- Failure gates: unknown SKU, low confidence, missing DB row, invalid price, camera/model/database faults
- Duplicate-track quantity checks (100 frames = 1; two physical items = 2)
- Named-condition robustness sweep and 5/10/20/30-object stress measurements
- R&D report with measured accuracy, FPS, latency, and an explicit non-production verdict

**Verdict:** YOLO26m alone is not sufficient. Do not treat this checkout as production-ready until a much larger real-photo dataset exists and live-threshold recall is high.

---

Completed:

- Phase 1–6 detection, tracking, catalog, and database prices
- Cart service: unique tracks → quantity, unit price, tax, and line total from the database
- POS UI at `/` with live camera, cart, and GENERATE BILL
- Manual controls: add, remove, increase, decrease, confirm, clear, new transaction
- Subtotal, GST from each product's stored `tax_rate`, configurable discount, grand total
- PDF invoices and `transactions` / `transaction_items` persistence
- After billing: save transaction, clear cart, start a new transaction

Explicitly **not** implemented yet:

- Payments / tender types
- Second-stage embedding model (architecture hook only)

---

## Future phases

1. **Grow the dataset** — capture 1000+ real variations per SKU, then retrain
2. **Optional embedding refinement** — only if similar SKUs still collide after a stronger detector
3. **Payments and receipts** — tender types and printed copies

---

## Hardware notes (this machine)

| Item | Status |
| --- | --- |
| OS | Windows 11 |
| GPU | NVIDIA GeForce GTX 1650 (4 GB) |
| NVIDIA driver | 457.49 (reports CUDA 11.1) |
| CUDA toolkit (`nvcc`) | Not installed |
| PyTorch | `2.13.0+cpu` — CPU wheel |

The installed PyTorch build is **CPU-only**. The GTX 1650 driver (CUDA 11.1) is too old for current CUDA PyTorch wheels. Live detection therefore runs on CPU; FPS will be lower than a CUDA build. Update the NVIDIA driver if you want GPU inference later.

---

## License

Project source is private unless a license file is added. Ultralytics YOLO26 is distributed under AGPL-3.0 or an Ultralytics Enterprise license.
