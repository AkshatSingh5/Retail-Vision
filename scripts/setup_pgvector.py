"""Enable PostgreSQL pgvector for Retail Vision product embeddings.

Run once against your DATABASE_URL:

  python scripts/setup_pgvector.py

Requires: PostgreSQL with the pgvector extension available
  (e.g. CREATE EXTENSION needs superuser or rds_superuser).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.database import get_engine, init_db
from backend.app.services.vector_search import ensure_pgvector_schema


def main() -> int:
    engine = init_db()
    if engine.dialect.name != "postgresql":
        print(f"Current database is {engine.dialect.name}, not PostgreSQL.")
        print("Set DATABASE_URL to a PostgreSQL URL, then rerun.")
        return 1
    ensure_pgvector_schema(engine)
    print("pgvector extension + embedding_vec column + index ensured.")
    print("Product embeddings will sync to embedding_vec on register/scan index.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
