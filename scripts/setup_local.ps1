$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

python -m venv .venv
& "$Root\.venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip
pip install -r backend/requirements.txt

if (-not (Test-Path "$Root\.env")) {
  Copy-Item "$Root\.env.example" "$Root\.env"
  Write-Host "Wrote .env from .env.example — edit DATABASE_URL before starting the API."
}

Write-Host ""
Write-Host "PostgreSQL + pgvector (optional local container):"
Write-Host "  docker compose up -d db"
Write-Host "Then: python scripts/setup_pgvector.py"
Write-Host ""
Write-Host "Start backend (from repo root):"
Write-Host "  .venv\Scripts\Activate.ps1"
Write-Host "  uvicorn backend.app.main:app --host 0.0.0.0 --port 8000"
Write-Host ""
Write-Host "Frontend is served by FastAPI at http://127.0.0.1:8000"
Write-Host "Or: cd frontend; npm install; npm run build; npm start"
