"""Repository / backend path bootstrap.

Keeps `from backend.app…` and `from vision…` working whether the process is
started from the repo root (`uvicorn backend.app.main:app`) or from
`backend/` (`uvicorn app.main:app`).
"""

from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
ROOT_DIR = BACKEND_DIR.parent

for _path in (str(ROOT_DIR), str(BACKEND_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
