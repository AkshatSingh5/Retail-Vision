from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ProductEmbedding(Base):
    """DINOv2 (or test) embedding for one product reference image.

    `embedding` is JSON text so SQLite tests work. PostgreSQL also stores
    `embedding_vec` (pgvector) via database migration — not as an ORM column,
    so create_all stays dialect-safe.
    """

    __tablename__ = "product_embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    product_id: Mapped[int] = mapped_column(
        ForeignKey("products.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    product_image_id: Mapped[int | None] = mapped_column(
        ForeignKey("product_images.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    embedding: Mapped[str] = mapped_column(Text, nullable=False)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    product = relationship("Product", backref="embeddings")
    product_image = relationship("ProductImage", backref="embeddings")
