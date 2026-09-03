"""Launch the Retail Vision POS (camera + cart + billing).

Run from the project root:

    python scripts/run_pos.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import uvicorn


def main() -> int:
    print("Retail Vision POS")
    print("  Local scan:  http://127.0.0.1:8000")
    print("  Vercel is UI only — Scan Product needs this FastAPI process.")
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
