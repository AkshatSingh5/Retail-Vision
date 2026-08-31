"""Seed the products table from products/registry.yaml.

Run from the project root:

    python scripts/seed_products.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.database import get_session_factory, init_db
from backend.app.services.seed import seed_products_from_registry


def main() -> int:
    init_db()
    session = get_session_factory()()
    try:
        created = seed_products_from_registry(session)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
    if created:
        print(f"Seeded {created} products.")
    else:
        print("Products table already has rows; seed skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
