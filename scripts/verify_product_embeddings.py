"""Development verification: products ↔ images ↔ embeddings readiness.

Usage (from Retail Vision root, with venv active):

    python scripts/verify_product_embeddings.py
    python scripts/verify_product_embeddings.py --product-id 12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import func, select

from backend.app.database import get_session_factory, init_db
from backend.app.models.product import Product
from backend.app.models.product_embedding import ProductEmbedding
from backend.app.models.product_image import ProductImage
from backend.app.services.vector_search import parse_embedding_json


def _embedding_dim(raw: str | None) -> int | None:
    values = parse_embedding_json(raw)
    return len(values) if values else None


def verify(product_id: int | None = None) -> int:
    init_db()
    session = get_session_factory()()
    try:
        statement = select(Product).where(Product.is_active.is_(True)).order_by(Product.id)
        if product_id is not None:
            statement = statement.where(Product.id == int(product_id))
        products = list(session.scalars(statement))
        if not products:
            print("No products found.")
            return 1

        missing = 0
        for product in products:
            images = list(
                session.scalars(
                    select(ProductImage).where(ProductImage.product_id == int(product.id))
                )
            )
            embeddings = list(
                session.scalars(
                    select(ProductEmbedding).where(ProductEmbedding.product_id == int(product.id))
                )
            )
            dims = {_embedding_dim(row.embedding) for row in embeddings}
            dims.discard(None)
            models = {str(row.model_name) for row in embeddings if row.model_name}
            if not images:
                status = "IMAGE MISSING"
                missing += 1
            elif not embeddings:
                status = "EMBEDDING MISSING"
                missing += 1
            else:
                status = "READY"

            dim_text = ",".join(str(d) for d in sorted(dims)) if dims else "n/a"
            model_text = ",".join(sorted(models)) if models else "n/a"
            print(f"Product ID: {product.id}")
            print(f"Name: {product.name}")
            print(f"Images: {len(images)}")
            print(f"Embeddings: {len(embeddings)}")
            print(f"Dimension: {dim_text}")
            print(f"Model: {model_text}")
            print(f"Status: {status}")
            print("-" * 40)

        print(f"Checked {len(products)} product(s); incomplete={missing}")
        return 0 if missing == 0 else 2
    finally:
        session.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify product image/embedding readiness")
    parser.add_argument("--product-id", type=int, default=None)
    args = parser.parse_args()
    raise SystemExit(verify(args.product_id))


if __name__ == "__main__":
    main()
