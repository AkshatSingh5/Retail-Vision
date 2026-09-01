"""Compatibility shim so `import vision` still works from the repo root.

The package lives in `backend/vision/` (GPU backend). This module redirects
the package search path so existing scripts keep working.
"""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent.parent / "backend" / "vision")]
