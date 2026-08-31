"""Download and warm-load DINOv2 so the first POS scan is not blocked.

Usage (from Retail Vision project root, venv active):

  python scripts/preload_dinov2.py

Caches facebook/dinov2-small under the Hugging Face hub cache, then runs a
dummy embedding to verify CPU inference works.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.config import DINOV2_MODEL, EMBEDDING_BACKEND
from vision.recognition.dinov2 import get_dinov2, reset_dinov2


def main() -> int:
    print(f"EMBEDDING_BACKEND={EMBEDDING_BACKEND}")
    print(f"DINOV2_MODEL={DINOV2_MODEL}")
    if EMBEDDING_BACKEND != "dinov2":
        print("Warning: EMBEDDING_BACKEND is not dinov2 — production scans use color/fallback.")
    reset_dinov2()
    started = time.perf_counter()
    embedder = get_dinov2()
    load_ms = (time.perf_counter() - started) * 1000.0
    print(f"Loaded on {embedder.device} in {load_ms:.0f} ms (dim={embedder.dim})")

    dummy = np.zeros((224, 224, 3), dtype=np.uint8)
    dummy[:, :] = (40, 90, 200)
    started = time.perf_counter()
    vector = embedder.embed_bgr(dummy)
    infer_ms = (time.perf_counter() - started) * 1000.0
    print(f"Dummy embedding ok: len={len(vector)} in {infer_ms:.0f} ms")
    print("DINOv2 is cached. First Scan Product should skip the Hugging Face download.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
