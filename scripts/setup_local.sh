#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r backend/requirements.txt

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Wrote .env from .env.example — edit DATABASE_URL before starting the API."
fi

echo
echo "PostgreSQL + pgvector (optional local container):"
echo "  docker compose up -d db"
echo "Then: python scripts/setup_pgvector.py"
echo
echo "Start backend (from repo root):"
echo "  source .venv/bin/activate"
echo "  uvicorn backend.app.main:app --host 0.0.0.0 --port 8000"
echo
echo "Frontend is served by FastAPI at http://127.0.0.1:8000"
echo "Or: cd frontend && npm install && npm run build && npm start"
