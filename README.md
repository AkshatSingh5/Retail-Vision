# Retail Vision

AI-powered retail product recognition and billing.

**Vercel should deploy frontend only.**

**YOLO26m + DINOv2 + CUDA should run on the GPU backend, not as a Vercel serverless function.**

```text
                    USER
                     │
                     ↓
              ┌──────────────┐
              │   Vercel     │
              │  Frontend    │
              └──────┬───────┘
                     │ HTTPS
                     ↓
              ┌──────────────┐
              │ FastAPI      │
              │ GPU Backend  │
              └──────┬───────┘
                     │
          ┌──────────┼──────────┐
          ↓          ↓          ↓
       YOLO26m    DINOv2    Florence-2
          │          │
          └──────┬───┘
                 ↓
        PostgreSQL + pgvector
                 │
                 ↓
            Product DB

        Images → local disk or S3
```

Camera / image → YOLO26m (detect + crop) → DINOv2 embedding → pgvector similarity → known or unknown product → cart → bill.

Identity is visual only. YOLO is not used as the SKU classifier. Unmatched items stay unknown; they are never forced onto the nearest catalog product.

---

## 1. Project overview

Retail Vision is a POS checkout: live camera or upload, product detection, visual matching, cart, and PDF invoice. Florence-2 provides optional image-to-text captions.

The UI is **vanilla HTML/CSS/JavaScript** (not Next.js or Vite). FastAPI serves it locally. Production splits:

| Piece | Where it runs |
| --- | --- |
| Frontend | Vercel (static files in `frontend/`) |
| Backend | GPU-capable Linux server (FastAPI) |
| Database | Cloud PostgreSQL + pgvector (local Postgres for development) |
| Images | Local filesystem in development, S3 in production |

---

## 2. Architecture

```text
Retail Vision
     │
     ├── frontend/     → Vercel (static POS UI)
     └── backend/      → GPU server (FastAPI + YOLO + DINOv2 + Florence-2)
              │
              ├── PostgreSQL + pgvector
              └── local disk or S3
```

The frontend calls the backend with `API_BASE_URL` (never a hardcoded production host). CORS allows local UI origins plus `FRONTEND_URL`.

`GET /health` returns `{"status":"ok"}` and does **not** load YOLO, DINOv2, or Florence-2.

---

## 3. Repository structure

```text
Retail-Vision/
├── frontend/                 # Vercel root directory
│   ├── index.html
│   ├── src/                  # pos.js, pos.css, api.js, config.js
│   ├── package.json
│   ├── .env.example
│   └── README.md
├── backend/
│   ├── app/
│   │   ├── main.py           # FastAPI entrypoint
│   │   ├── config.py
│   │   ├── database.py
│   │   ├── api/routes/       # health + re-exports of existing routers
│   │   ├── core/             # config re-export
│   │   ├── db/               # database/model re-exports
│   │   └── services/         # scan, products, storage, cart, …
│   ├── vision/               # YOLO, DINOv2, Florence-2, tracking
│   │   └── models/           # yolo26m.pt (not committed; see below)
│   ├── Dockerfile
│   ├── requirements.txt
│   └── .env.example
├── scripts/                  # local setup, training, tests
├── products/registry.yaml
├── docker-compose.yml        # local PostgreSQL + pgvector only
├── .env.example
└── README.md
```

A thin `vision/` shim at the repo root keeps `from vision…` working in existing scripts.

---

## 4. Local setup

Prerequisites: Python 3.12+, Git, PostgreSQL with pgvector (or Docker).

**Windows (PowerShell):**

```powershell
.\scripts\setup_local.ps1
copy .env.example .env   # if the script did not already create it
# edit DATABASE_URL
.venv\Scripts\Activate.ps1
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

**Linux / macOS:**

```bash
chmod +x scripts/setup_local.sh
./scripts/setup_local.sh
cp .env.example .env
source .venv/bin/activate
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

Open http://127.0.0.1:8000 for the POS. Docs: http://127.0.0.1:8000/docs

---

## 5. Frontend setup

The POS is static files. Locally, FastAPI serves `frontend/index.html` at `/`.

To run the UI on port 3000 against the API on 8000:

```bash
cd frontend
npm install
# API_BASE_URL=http://localhost:8000
npm run build
npm start
```

Set `FRONTEND_URL=http://localhost:3000` on the backend so CORS allows that origin.

Do not put `DATABASE_URL`, AWS keys, or any secret in frontend env vars.

---

## 6. Backend setup

From the **repository root** (required so `backend.app` and `vision` import):

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

Equivalent: `python scripts/run_pos.py`

CUDA wheels (GPU server):

```bash
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r backend/requirements.txt
```

---

## 7. PostgreSQL setup

Local container:

```bash
docker compose up -d db
```

Default compose credentials (development only): user `postgres`, password `postgres`, database `retail_vision`.

Then in `.env`:

```text
DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/retail_vision
```

Production: set `DATABASE_URL` to your cloud provider URL. The backend does not assume `DB_HOST=localhost`.

---

## 8. pgvector setup

```bash
python scripts/setup_pgvector.py
```

The API also tries to enable the extension on startup. The database role needs permission to `CREATE EXTENSION vector`.

---

## 9. GPU requirements

- NVIDIA GPU + driver that reports CUDA 12.6+ for the pinned PyTorch **cu126** wheels
- `YOLO_DEVICE=cuda` and `DINO_DEVICE=cuda`
- If CUDA is unavailable, the process **logs a warning** and uses CPU. It does not pretend CUDA exists.

Startup logs look like:

```text
YOLO device: cuda
DINO device: cuda
```

or:

```text
YOLO device: cpu
DINO device: cpu
WARNING: CUDA unavailable. Running on CPU.
```

---

## 10. YOLO26m setup

Weights path: `backend/vision/models/yolo26m.pt` (also accepted: repo-root `vision/models/…` via the shim).

Files are **not committed** (`*.pt` is gitignored). Official `yolo26m.pt` is downloaded by Ultralytics on first load if the file is missing.

Custom retail weights (`retail_yolo26m_v2.pt`, ~44 MB each) stay on the GPU machine. Copy them next to `yolo26m.pt` and point `YOLO_MODEL_PATH` at them if you use the fine-tuned detector.

Do not upload private weights to a random public host. Use Git LFS only if you explicitly decide to version a public-licensed checkpoint, or copy weights from private object storage during server provision.

---

## 11. DINOv2 setup

`DINOV2_MODEL=facebook/dinov2-small` is downloaded from Hugging Face on first load (`PRELOAD_DINOV2=true` at API startup). Cache lives under the Hugging Face home directory (gitignored).

---

## 12. Florence-2 setup

`FLORENCE_MODEL=microsoft/Florence-2-base`. Default `PRELOAD_FLORENCE=false` — loaded on the first `/caption` request. First run can take several minutes.

---

## 13. Environment variables

See `.env.example`, `backend/.env.example`, and `frontend/.env.example`. Copy to `.env` and fill in real values. **Never commit `.env`.**

| Variable | Role |
| --- | --- |
| `DATABASE_URL` | SQLAlchemy URL (PostgreSQL in production) |
| `FRONTEND_URL` | Production Vercel origin for CORS |
| `API_BASE_URL` | Frontend → backend origin (Vercel env) |
| `YOLO_DEVICE` / `DINO_DEVICE` | `cuda` or `cpu` |
| `PRELOAD_YOLO` / `PRELOAD_DINOV2` | Load models once per process |
| `PRODUCT_MATCH_THRESHOLD` / `PRODUCT_MATCH_MARGIN` | Unknown-product gates |
| `STORAGE_BACKEND` | `local` or `s3` |

Matching aliases kept: `DINO_SIMILARITY_THRESHOLD`, `SCAN_MATCH_THRESHOLD`, `SCAN_MATCH_MARGIN`.

---

## 14. Vercel deployment

**Root Directory: `frontend`**

- Build: `npm run build`
- Env: `API_BASE_URL=https://YOUR-BACKEND-DOMAIN` (no trailing slash)

Do **not** set Vercel Root Directory to the repo root Python app. Do **not** use a Python serverless entrypoint. The previous `500 INTERNAL_SERVER_ERROR` / `FUNCTION_INVOCATION_FAILED` came from running FastAPI + (attempted) vision on Vercel: the CUDA stack exceeded the function size limit, OpenCV lacked libGL, and lifespan/DB init is not a serverless workload.

---

## 15. Backend deployment

On a GPU Linux host:

```bash
git clone <repo>
cd Retail-Vision
python -m venv .venv
source .venv/bin/activate
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r backend/requirements.txt
cp backend/.env.example .env
# set DATABASE_URL, FRONTEND_URL, STORAGE_BACKEND, …
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

Docker (from repo root):

```bash
docker build -f backend/Dockerfile -t retail-vision-backend .
docker run --gpus all -p 8000:8000 --env-file .env retail-vision-backend
```

Put the server behind HTTPS (Caddy, nginx, or a load balancer) and point `API_BASE_URL` / `FRONTEND_URL` at those public URLs.

---

## 16. S3 configuration

```text
STORAGE_BACKEND=s3
S3_BUCKET=your-bucket
AWS_REGION=ap-south-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

`StorageService` (`save_image` / `get_image` / `delete_image`) writes objects and stores keys/URLs in PostgreSQL, not image blobs.

Local development: `STORAGE_BACKEND=local` (files under `storage/` and `products/images/`, gitignored).

---

## 17. GitHub workflow

1. Confirm `.env` is not staged (`git status`).
2. Do not add `*.pt`, `node_modules`, `.venv`, `storage/`, Hugging Face caches.
3. Commit and push when you are ready:

```bash
git add .
git commit -m "Reorganize Retail Vision for production deployment"
git push origin main
```

---

## 18. Troubleshooting

| Symptom | What to check |
| --- | --- |
| Vercel `FUNCTION_INVOCATION_FAILED` | The Python API is still configured as a Vercel function. Point the project at `frontend/` only. |
| UI loads, catalog empty / CORS error | Set `API_BASE_URL` on Vercel and `FRONTEND_URL` on the GPU server. |
| `GET /health` ok but scan 503 | Models failed to load. Read backend logs for YOLO/DINOv2 errors. |
| CUDA warning + CPU | Driver too old or CPU-only PyTorch wheel. Install cu126 wheels. |
| pgvector skipped | Extension not allowed. Run `scripts/setup_pgvector.py` as a superuser. |
| Unknown product always | Intended when score &lt; threshold or margin is too small. Do not lower gates just to force a match. |
| Missing `yolo26m.pt` | First request downloads official weights, or copy the file into `backend/vision/models/`. |

Existing API routes are unchanged, including `POST /products/scan`, cart, checkout, and `/caption`.

Health: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health) and [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)
