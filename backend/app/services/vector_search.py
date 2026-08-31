"""Similarity search over stored product embeddings.

PostgreSQL: pgvector cosine distance when embedding_vec is available.
SQLite / fallback: JSON embeddings + NumPy cosine (no image reloads).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from backend.app.config import EMBEDDING_DIM, VECTOR_SEARCH_TOP_K
from backend.app.models.product import Product
from backend.app.models.product_embedding import ProductEmbedding
from backend.app.schemas.product import public_image_url, serialize_money
from vision.recognition.matching import ImageHit

logger = logging.getLogger(__name__)


def format_vector(values: list[float]) -> str:
    return "[" + ",".join(f"{float(v):.8f}" for v in values) + "]"


def parse_embedding_json(raw: str | None) -> list[float] | None:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        values = parsed.get("vec")
    elif isinstance(parsed, list):
        values = parsed
    else:
        return None
    if not isinstance(values, list) or not values:
        return None
    return [float(item) for item in values]


def embedding_payload(values: list[float], model_name: str) -> str:
    return json.dumps({"v": model_name, "model": model_name, "vec": values})


def cosine_similarity(left: list[float], right: list[float]) -> float:
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def is_postgres(session: Session) -> bool:
    bind = session.get_bind()
    return bool(bind is not None and bind.dialect.name == "postgresql")


def _vector_type_available(session: Session) -> bool:
    """True only when the PostgreSQL `vector` type exists (pgvector installed)."""
    if not is_postgres(session):
        return False
    try:
        row = session.execute(text("SELECT 1 FROM pg_type WHERE typname = 'vector'")).first()
        return row is not None
    except Exception:
        return False


def _embedding_vec_column_available(session: Session) -> bool:
    if not is_postgres(session):
        return False
    try:
        row = session.execute(
            text(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_name = 'product_embeddings' AND column_name = 'embedding_vec'"
            )
        ).first()
        return row is not None
    except Exception:
        return False


def _pgvector_available(session: Session) -> bool:
    """pgvector search/sync needs both the vector type and embedding_vec column."""
    return _vector_type_available(session) and _embedding_vec_column_available(session)


def sync_pgvector(session: Session, embedding_id: int, values: list[float]) -> None:
    """Best-effort copy of JSON embedding into embedding_vec.

    Must NEVER abort the surrounding product-registration transaction when
    pgvector is missing. Uses a SAVEPOINT so Postgres InFailedSqlTransaction
    cannot poison later INSERTs.
    """
    if not is_postgres(session):
        return
    if not _pgvector_available(session):
        logger.debug(
            "Skipping pgvector sync for embedding %s (extension/column unavailable); JSON embedding retained",
            embedding_id,
        )
        return
    if len(values) != int(EMBEDDING_DIM):
        logger.warning(
            "Skipping pgvector sync for embedding %s: dim %s != configured %s",
            embedding_id,
            len(values),
            EMBEDDING_DIM,
        )
        return
    try:
        # Nested transaction = SAVEPOINT. Failure rolls back only this sync.
        with session.begin_nested():
            session.execute(
                text(
                    "UPDATE product_embeddings "
                    "SET embedding_vec = CAST(:vec AS vector) "
                    "WHERE id = :id"
                ),
                {"vec": format_vector(values), "id": int(embedding_id)},
            )
        logger.info("[PGVECTOR] Synced embedding_id=%s dim=%s", embedding_id, len(values))
    except Exception as extra:
        logger.warning(
            "pgvector sync failed for embedding %s (JSON embedding kept): %s",
            embedding_id,
            extra,
        )


def search_similar(
    session: Session,
    query: list[float],
    *,
    model_name: str,
    top_k: int | None = None,
) -> list[ImageHit]:
    limit = int(top_k or VECTOR_SEARCH_TOP_K)
    if not query:
        return []
    if _pgvector_available(session):
        try:
            return _search_pgvector(session, query, model_name=model_name, top_k=limit)
        except Exception as extra:
            logger.warning("pgvector search failed; falling back to JSON: %s", extra)
    return _search_json(session, query, model_name=model_name, top_k=limit)


def _search_pgvector(
    session: Session,
    query: list[float],
    *,
    model_name: str,
    top_k: int,
) -> list[ImageHit]:
    rows = session.execute(
        text(
            """
            SELECT
                pe.product_id,
                pe.product_image_id,
                1 - (pe.embedding_vec <=> CAST(:q AS vector)) AS similarity,
                p.name,
                p.sku,
                p.price,
                p.tax_rate,
                p.brand,
                p.category,
                p.image_path
            FROM product_embeddings pe
            JOIN products p ON p.id = pe.product_id
            WHERE p.is_active IS TRUE
              AND pe.model_name = :model
              AND pe.embedding_vec IS NOT NULL
            ORDER BY pe.embedding_vec <=> CAST(:q AS vector)
            LIMIT :k
            """
        ),
        {"q": format_vector(query), "model": model_name, "k": int(top_k)},
    ).mappings().all()
    hits: list[ImageHit] = []
    for row in rows:
        price = serialize_money(row["price"])
        tax = serialize_money(row["tax_rate"])
        if price is None or tax is None:
            continue
        hits.append(
            ImageHit(
                product_id=int(row["product_id"]),
                product_image_id=int(row["product_image_id"]) if row["product_image_id"] is not None else None,
                similarity=float(row["similarity"]),
                name=str(row["name"]),
                sku=str(row["sku"]),
                price=price,
                tax_rate=tax,
                brand=row["brand"],
                category=row["category"],
                image_url=public_image_url(int(row["product_id"]), row["image_path"]),
            )
        )
    return hits


def _search_json(
    session: Session,
    query: list[float],
    *,
    model_name: str,
    top_k: int,
) -> list[ImageHit]:
    statement = (
        select(ProductEmbedding, Product)
        .join(Product, Product.id == ProductEmbedding.product_id)
        .where(Product.is_active.is_(True), ProductEmbedding.model_name == model_name)
    )
    scored: list[tuple[float, Any, Product]] = []
    for embedding_row, product in session.execute(statement).all():
        vector = parse_embedding_json(embedding_row.embedding)
        if not vector:
            continue
        score = cosine_similarity(query, vector)
        scored.append((score, embedding_row, product))
    scored.sort(key=lambda item: item[0], reverse=True)
    hits: list[ImageHit] = []
    for score, embedding_row, product in scored[: max(1, int(top_k))]:
        price = serialize_money(product.price)
        tax = serialize_money(product.tax_rate)
        if price is None or tax is None:
            continue
        hits.append(
            ImageHit(
                product_id=int(product.id),
                product_image_id=int(embedding_row.product_image_id)
                if embedding_row.product_image_id is not None
                else None,
                similarity=float(score),
                name=str(product.name),
                sku=str(product.sku),
                price=price,
                tax_rate=tax,
                brand=product.brand,
                category=product.category,
                image_url=public_image_url(int(product.id), product.image_path),
            )
        )
    return hits


def ensure_pgvector_schema(engine) -> None:
    """Enable pgvector and add embedding_vec + HNSW index when using PostgreSQL."""
    if engine.dialect.name != "postgresql":
        return
    dim = int(EMBEDDING_DIM)
    from backend.app.config import PGVECTOR_INDEX

    with engine.begin() as connection:
        try:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        except Exception as extra:
            logger.warning("Could not enable pgvector extension: %s", extra)
            return
        connection.execute(
            text(
                f"ALTER TABLE product_embeddings "
                f"ADD COLUMN IF NOT EXISTS embedding_vec vector({dim})"
            )
        )
        if PGVECTOR_INDEX == "hnsw":
            connection.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS ix_product_embeddings_hnsw
                    ON product_embeddings
                    USING hnsw (embedding_vec vector_cosine_ops)
                    """
                )
            )
        elif PGVECTOR_INDEX == "ivfflat":
            # IVFFlat needs data to train; create only when rows exist.
            count = connection.execute(text("SELECT COUNT(*) FROM product_embeddings")).scalar() or 0
            if int(count) >= 20:
                lists = max(10, int(count) // 20)
                connection.execute(
                    text(
                        f"""
                        CREATE INDEX IF NOT EXISTS ix_product_embeddings_ivfflat
                        ON product_embeddings
                        USING ivfflat (embedding_vec vector_cosine_ops)
                        WITH (lists = {lists})
                        """
                    )
                )
        # Backfill vector column from JSON {"vec":[...]} embeddings when missing.
        connection.execute(
            text(
                """
                UPDATE product_embeddings
                SET embedding_vec = CAST(
                    regexp_replace((embedding::jsonb -> 'vec')::text, '\\s', '', 'g')
                    AS vector
                )
                WHERE embedding_vec IS NULL
                  AND embedding IS NOT NULL
                  AND embedding LIKE '{%'
                  AND embedding::jsonb ? 'vec'
                """
            )
        )
