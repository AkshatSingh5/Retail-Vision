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
