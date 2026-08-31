"""Product visual embeddings + in-memory gallery cache.

Production path: DINOv2 → product_embeddings (PostgreSQL / SQLite JSON).
Tests / fallback: color+layout embedding when EMBEDDING_BACKEND=color.

Matching rules (also applied in matching.decide_match via scan_service):
  1. Best product score >= PRODUCT_MATCH_THRESHOLD
  2. Best − second-best >= PRODUCT_MATCH_MARGIN
Otherwise ambiguous / not found (never guess).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy import select

from backend.app.config import (
    EMBEDDING_BACKEND,
    GALLERY_MATCH_THRESHOLD,
    PRODUCT_MATCH_MARGIN,
    PRODUCT_MATCH_THRESHOLD,
    ROOT_DIR,
)
from vision.recognition.vector_bridge import current_model_name
from backend.app.database import get_session_factory, init_db
from backend.app.models.product import Product
from backend.app.models.product_image import ProductImage
from vision.recognition.preprocess import prepare_product_image

logger = logging.getLogger(__name__)

_NON_PRODUCT_IMAGE_TYPES = frozenset({"scan_frame", "full_frame", "frame"})


def embedding_model_name() -> str:
    return current_model_name()


# Back-compat alias used by scan logs / older imports.
EMBEDDING_VERSION = embedding_model_name()


def refresh_embedding_version() -> str:
    """Re-read backend after tests mutate EMBEDDING_BACKEND at runtime."""
    global EMBEDDING_VERSION
    EMBEDDING_VERSION = embedding_model_name()
    return EMBEDDING_VERSION


def _color_layout_embedding(image: np.ndarray) -> list[float]:
    """Fast discriminative color + layout embedding (tests / offline fallback)."""
    prepared = prepare_product_image(image)
    size = prepared.shape[0]

    hsv = cv2.cvtColor(prepared, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(prepared, cv2.COLOR_BGR2LAB)

    hist_hsv = cv2.calcHist([hsv], [0, 1, 2], None, [18, 8, 8], [0, 180, 0, 256, 0, 256])
    hist_hsv = cv2.normalize(hist_hsv, hist_hsv).flatten().astype(np.float32)

    hist_lab = cv2.calcHist([lab], [1, 2], None, [12, 12], [0, 256, 0, 256])
    hist_lab = cv2.normalize(hist_lab, hist_lab).flatten().astype(np.float32)

    cells: list[float] = []
    grid = 4
    cell = size // grid
    for gy in range(grid):
        for gx in range(grid):
            patch = hsv[gy * cell : (gy + 1) * cell, gx * cell : (gx + 1) * cell]
            means = patch.reshape(-1, 3).mean(axis=0)
            cells.extend(
                [
                    float(means[0] / 180.0),
                    float(means[1] / 255.0),
                    float(means[2] / 255.0),
                    float(patch[:, :, 1].std() / 255.0),
                ]
            )
    spatial = np.asarray(cells, dtype=np.float32)

    gray = cv2.cvtColor(prepared, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 140).astype(np.float32) / 255.0
    edge_vec = cv2.resize(edges, (16, 16), interpolation=cv2.INTER_AREA).flatten()

    hue_hist = cv2.calcHist([hsv], [0], None, [24], [0, 180]).flatten().astype(np.float32)
    hue_hist = hue_hist / (float(hue_hist.sum()) + 1e-6)

    vector = np.concatenate(
        [
            hist_hsv * 2.2,
            hist_lab * 1.8,
            spatial * 2.0,
            edge_vec * 1.2,
            hue_hist * 2.5,
        ]
    )
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector = vector / norm
    return [float(value) for value in vector.tolist()]


def crop_embedding(image: np.ndarray) -> list[float]:
    """Visual embedding for one product crop (DINOv2 or color backend)."""
    if EMBEDDING_BACKEND == "dinov2":
        from vision.recognition.dinov2 import dinov2_embed

        return dinov2_embed(image)
    return _color_layout_embedding(image)


def embedding_variants(image: np.ndarray) -> list[list[float]]:
    """Reference views for one image. DINOv2: single vector (same preprocess as scan)."""
    if image is None or image.size == 0:
        return []
    if EMBEDDING_BACKEND == "dinov2":
        try:
            return [crop_embedding(image)]
        except Exception:
            return []

    variants: list[np.ndarray] = [image]
    height, width = image.shape[:2]
    if min(height, width) >= 360 and max(height, width) / max(1, min(height, width)) >= 1.25:
        for ratio in (0.45, 0.55, 0.65):
            ch = max(32, int(height * ratio))
            cw = max(32, int(width * ratio))
            y0 = max(0, (height - ch) // 2)
            x0 = max(0, (width - cw) // 2)
            variants.append(image[y0 : y0 + ch, x0 : x0 + cw].copy())
    pad = max(8, min(height, width) // 8)
    variants.append(cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE))
    if height > 48 and width > 48:
        y0, y1 = height // 12, height - height // 12
        x0, x1 = width // 12, width - width // 12
        variants.append(image[y0:y1, x0:x1].copy())
    brighter = cv2.convertScaleAbs(image, alpha=1.12, beta=8)
    darker = cv2.convertScaleAbs(image, alpha=0.88, beta=-8)
    variants.extend([brighter, darker])

    vectors: list[list[float]] = []
    for sample in variants:
        try:
            vectors.append(crop_embedding(sample))
        except Exception:
            continue
    return vectors


def embedding_to_json(values: list[float]) -> str:
    model = embedding_model_name()
    return json.dumps({"v": model, "model": model, "vec": values})


def embedding_from_json(raw: str | None) -> list[float] | None:
    from vision.recognition.vector_bridge import embedding_from_json as _parse

    return _parse(raw, require_model=None)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    a = np.array(left, dtype=np.float32)
    b = np.array(right, dtype=np.float32)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _load_image(path_value: str | None) -> np.ndarray | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT_DIR / path
    if not path.exists():
        return None
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


@dataclass(frozen=True)
class GalleryMatch:
    product_id: int
    sku: str
    name: str
    price: int | float
    tax_rate: int | float
    score: float
    brand: str | None = None
    category: str | None = None
    second_score: float = 0.0
    margin: float = 0.0
    accepted: bool = False
    reason: str = ""


@dataclass(frozen=True)
class RankedProductScore:
    product_id: int
    sku: str
    name: str
    price: int | float
    tax_rate: int | float
    score: float
    brand: str | None = None
    category: str | None = None


class ProductGallery:
    """In-memory cache of registered product embeddings (many images per product)."""

    def __init__(self, threshold: float | None = None, margin: float | None = None) -> None:
        self.threshold = PRODUCT_MATCH_THRESHOLD if threshold is None else float(threshold)
        self.margin = PRODUCT_MATCH_MARGIN if margin is None else float(margin)
        if threshold is None and GALLERY_MATCH_THRESHOLD:
            # Keep gallery live-track threshold available; scan uses PRODUCT_MATCH_*.
            pass
        self._entries: list[tuple[list[float], GalleryMatch]] = []
        self._ready = False

    def invalidate(self) -> None:
        self._ready = False
        self._entries = []

    def reload(self) -> None:
        init_db()
        session = get_session_factory()()
        entries: list[tuple[list[float], GalleryMatch]] = []
        try:
            from backend.app.models.product_embedding import ProductEmbedding
            from backend.app.schemas.product import serialize_money
            from backend.app.services.vector_search import parse_embedding_json

            model = embedding_model_name()
            products = {
                int(p.id): p
                for p in session.scalars(select(Product).where(Product.is_active.is_(True))).all()
            }
            emb_rows = session.scalars(
                select(ProductEmbedding).where(ProductEmbedding.model_name == model)
            ).all()
            for row in emb_rows:
                product = products.get(int(row.product_id))
                if product is None:
                    continue
                vector = parse_embedding_json(row.embedding)
                if not vector:
                    continue
                price = serialize_money(product.price)
                tax = serialize_money(product.tax_rate)
                if price is None or tax is None:
                    continue
                meta = GalleryMatch(
                    product_id=int(product.id),
                    sku=str(product.sku),
                    name=str(product.name),
                    price=price,
                    tax_rate=tax,
                    score=0.0,
                    brand=product.brand,
                    category=product.category,
                )
                entries.append((vector, meta))

            # Rebuild from disk when embedding table is empty (first migration).
            if not entries:
                images = session.scalars(select(ProductImage)).all()
                images_by_product: dict[int, list[ProductImage]] = {}
                for image in images:
                    images_by_product.setdefault(int(image.product_id), []).append(image)
                for product in products.values():
                    price = serialize_money(product.price)
                    tax = serialize_money(product.tax_rate)
                    if price is None or tax is None:
                        continue
                    meta = GalleryMatch(
                        product_id=int(product.id),
                        sku=str(product.sku),
                        name=str(product.name),
                        price=price,
                        tax_rate=tax,
                        score=0.0,
                        brand=product.brand,
                        category=product.category,
                    )
                    for image in images_by_product.get(int(product.id), []):
                        image_type = str(image.image_type or "").strip().lower()
                        if image_type in _NON_PRODUCT_IMAGE_TYPES:
                            continue
                        disk = _load_image(image.image_path)
                        if disk is None:
                            continue
                        try:
                            for vector in embedding_variants(disk):
                                entries.append((vector, meta))
                        except Exception:
                            continue

            self._entries = entries
            self._ready = True
            print(f"[INDEX] Recognition gallery loaded: {len(self._entries)} embedding(s) ({model})")
        finally:
            session.close()

    def register_product(
        self,
        *,
        product_id: int,
        sku: str,
        name: str,
        price: int | float,
        tax_rate: int | float,
        vectors: list[list[float]],
        brand: str | None = None,
        category: str | None = None,
    ) -> None:
        """Add embeddings immediately so a newly saved SKU is searchable without restart."""
        if not self._ready:
            self.reload()
        meta = GalleryMatch(
            product_id=int(product_id),
            sku=str(sku),
            name=str(name),
            price=price,
            tax_rate=tax_rate,
            score=0.0,
            brand=brand,
            category=category,
        )
        added = 0
        for vector in vectors:
            if not vector:
                continue
            self._entries.append((vector, meta))
            added += 1
        self._ready = True
        print(
            f"[INDEX] Added product_id={product_id} sku={sku} "
            f"with {added} embedding(s); gallery size={len(self._entries)}"
        )

    def rank_products(
        self,
        crop: np.ndarray,
        top_k: int = 5,
        *,
        session=None,
        query: list[float] | None = None,
    ) -> list[RankedProductScore]:
        """Group image hits by product_id and return product-level scores."""
        if query is None:
            if crop is None or getattr(crop, "size", 0) == 0:
                return []
            try:
                query = crop_embedding(crop)
            except Exception:
                return []

        if session is not None:
            from backend.app.services.vector_search import search_similar
            from vision.recognition.matching import group_by_product

            hits = search_similar(session, query, model_name=embedding_model_name())
            grouped = group_by_product(hits)
            return [
                RankedProductScore(
                    product_id=row.product_id,
                    sku=row.sku,
                    name=row.name,
                    price=row.price,
                    tax_rate=row.tax_rate,
                    score=float(row.score),
                    brand=row.brand,
                    category=row.category,
                )
                for row in grouped[: max(1, int(top_k))]
            ]

        if not self._ready:
            self.reload()
        if not self._entries:
            return []
        best_by_product: dict[int, RankedProductScore] = {}
        for vector, meta in self._entries:
            score = cosine_similarity(query, vector)
            existing = best_by_product.get(meta.product_id)
            if existing is None or score > existing.score:
                best_by_product[meta.product_id] = RankedProductScore(
                    product_id=meta.product_id,
                    sku=meta.sku,
                    name=meta.name,
                    price=meta.price,
                    tax_rate=meta.tax_rate,
                    score=score,
                    brand=meta.brand,
                    category=meta.category,
                )
        ranked = sorted(best_by_product.values(), key=lambda row: row.score, reverse=True)
        return ranked[: max(1, int(top_k))]

    def match(
        self,
        crop: np.ndarray,
        *,
        threshold: float | None = None,
        margin: float | None = None,
    ) -> GalleryMatch | None:
        decision = self.match_decision(crop, threshold=threshold, margin=margin)
        if decision is None or not decision.accepted:
            return None
        return decision

    def match_decision(
        self,
        crop: np.ndarray,
        *,
        threshold: float | None = None,
        margin: float | None = None,
        session=None,
        query: list[float] | None = None,
    ) -> GalleryMatch | None:
        min_score = self.threshold if threshold is None else float(threshold)
        min_margin = self.margin if margin is None else float(margin)
        ranked = self.rank_products(crop, top_k=5, session=session, query=query)
        if not ranked:
            return None

        best = ranked[0]
        if len(ranked) > 1:
            second_score = float(ranked[1].score)
        else:
            second_score = float(min_score)
        gap = float(best.score - second_score)
        accepted = best.score >= min_score and gap >= min_margin
        if best.score < min_score:
            reason = "below_threshold"
        elif gap < min_margin:
            reason = "ambiguous_margin"
        else:
            reason = "accepted"

        return GalleryMatch(
            product_id=best.product_id,
            sku=best.sku,
            name=best.name,
            price=best.price,
            tax_rate=best.tax_rate,
            score=float(best.score),
            brand=best.brand,
            category=best.category,
            second_score=float(second_score),
            margin=gap,
            accepted=accepted,
            reason=reason,
        )


_gallery: ProductGallery | None = None


def get_gallery() -> ProductGallery:
    global _gallery
    if _gallery is None:
        _gallery = ProductGallery()
    return _gallery
