# Retail Vision backend (FastAPI + YOLO26m + DINOv2 + Florence-2)

The GPU inference API. Do not deploy this folder to Vercel.

## Run locally (from the repository root)

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
pip install -r backend/requirements.txt
copy backend\.env.example .env   # then edit DATABASE_URL
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

Health (does not load models):

```text
GET http://127.0.0.1:8000/health
→ {"status":"ok"}
```

## Database migrations (Alembic)

Schema changes are versioned with **Alembic** (migrations in `backend/alembic/`).
The application runs `alembic upgrade head` automatically on startup
(`init_db`), so a normal `uvicorn` boot keeps the schema current for both
SQLite and PostgreSQL. Existing pre-Alembic databases are detected
(no `alembic_version` table) and stamped at head automatically on first boot.

Run the CLI from the repository root:

```bash
# Apply pending migrations to the DATABASE_URL in .env
.venv/bin/alembic -c backend/alembic.ini upgrade head        # Linux/macOS
.venv\Scripts\alembic -c backend\alembic.ini upgrade head    # Windows

# Generate a new migration after editing backend/app/models/*.py
alembic -c backend/alembic.ini revision --autogenerate -m "describe change"

# Show applied / pending revisions and verify no drift
alembic -c backend/alembic.ini current
alembic -c backend/alembic.ini check
```

Notes:

- `backend/alembic/env.py` resolves URLs through
  `backend.app.database.resolve_database_url`, so relative SQLite paths anchor
  to the repo root just like the application.
- The pgvector column `product_embeddings.embedding_vec` (and its HNSW/IVFFlat
  index) is managed out-of-band by `ensure_pgvector_schema` and is excluded from
  autogenerate diffs.
- `init_db("sqlite:///...")` and in-memory SQLite (`sqlite:///:memory:`) work
  with migrations; the app hands Alembic its engine explicitly.

## GPU server

- NVIDIA driver with CUDA 12.6+ (for the pinned PyTorch cu126 wheels)
- Place `yolo26m.pt` at `backend/vision/models/yolo26m.pt` (or let Ultralytics download it)
- Set `YOLO_DEVICE=cuda` and `DINO_DEVICE=cuda`
- If CUDA is missing, the process logs a warning and falls back to CPU

## Docker

From the repository root:

```bash
docker build -f backend/Dockerfile -t retail-vision-backend .
docker run --gpus all -p 8000:8000 --env-file .env retail-vision-backend
```
