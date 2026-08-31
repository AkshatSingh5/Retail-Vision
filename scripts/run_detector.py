"""Launch the real-time YOLO26m camera detector.

Run from the project root:

    python scripts/run_detector.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from vision.inference.run import run

if __name__ == "__main__":
    raise SystemExit(run())
