# Retail Vision frontend (POS UI)

Vanilla HTML/CSS/JavaScript. There is no React/Next/Vite app.

**Vercel should deploy this folder only.** The Python AI stack must not run on Vercel.

## Local (same origin)

Start the FastAPI backend from the repo root. It serves this UI at http://127.0.0.1:8000

## Local (separate origin)

```bash
cd frontend
npm install
# set API_BASE_URL=http://localhost:8000 then:
npm run build
npm start
```

## Vercel

1. Root Directory: `frontend`
2. Build command: `npm run build`
3. Environment variable: `API_BASE_URL=https://YOUR-BACKEND-DOMAIN` (no trailing slash)

Do not set `DATABASE_URL`, AWS keys, or any secret in frontend env vars.
