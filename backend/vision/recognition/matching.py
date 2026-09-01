"""Product-level visual matching: group image hits, score, threshold + margin."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class ImageHit:
    product_id: int
    product_image_id: int | None
    similarity: float
    name: str = ""
    sku: str = ""
    price: float | int = 0
    tax_rate: float | int = 0
    brand: str | None = None
    category: str | None = None
    image_url: str | None = None


@dataclass(frozen=True)
class ProductCandidate:
    product_id: int
    name: str
    sku: str
    price: float | int
    tax_rate: float | int
    best_similarity: float
    avg_top_similarity: float
    score: float
    match_count: int
    brand: str | None = None
    category: str | None = None
    image_url: str | None = None


@dataclass(frozen=True)
class MatchDecision:
    accepted: bool
    reason: str
    score: float
    second_score: float
    margin: float
    product_id: int | None = None
    candidate: ProductCandidate | None = None
    ranked: tuple[ProductCandidate, ...] = ()


def product_level_score(similarities: list[float]) -> tuple[float, float, float]:
    """Return (best, avg_top3, combined_score). Prefer NOT FOUND over wrong SKU."""
    if not similarities:
        return 0.0, 0.0, 0.0
    ordered = sorted((float(value) for value in similarities), reverse=True)
    best = ordered[0]
    top = ordered[: min(3, len(ordered))]
    avg_top = sum(top) / len(top)
    # Emphasize best match, but reward products with multiple strong views.
    score = (0.65 * best) + (0.35 * avg_top)
    return best, avg_top, score


def group_by_product(hits: list[ImageHit]) -> list[ProductCandidate]:
    buckets: dict[int, list[ImageHit]] = defaultdict(list)
    for hit in hits:
        buckets[int(hit.product_id)].append(hit)
    candidates: list[ProductCandidate] = []
    for product_id, rows in buckets.items():
        sims = [float(row.similarity) for row in rows]
        best, avg_top, score = product_level_score(sims)
        sample = max(rows, key=lambda row: row.similarity)
        candidates.append(
            ProductCandidate(
                product_id=product_id,
                name=str(sample.name),
                sku=str(sample.sku),
                price=sample.price,
                tax_rate=sample.tax_rate,
                best_similarity=best,
                avg_top_similarity=avg_top,
                score=score,
                match_count=len(rows),
                brand=sample.brand,
                category=sample.category,
                image_url=sample.image_url,
            )
        )
    candidates.sort(key=lambda row: row.score, reverse=True)
    return candidates


def decide_match(
    ranked: list[ProductCandidate],
    *,
    threshold: float,
    margin: float,
) -> MatchDecision:
    if not ranked:
        return MatchDecision(
            accepted=False,
            reason="below_threshold",
            score=0.0,
            second_score=0.0,
            margin=0.0,
            ranked=(),
        )

    best = ranked[0]
    if len(ranked) > 1:
        second_score = float(ranked[1].score)
    else:
        # No competing product — threshold alone decides (margin is for pairwise ambiguity).
        second_score = 0.0
    gap = float(best.score - second_score)

    if best.score < float(threshold):
        reason = "below_threshold"
        accepted = False
    elif gap < float(margin):
        reason = "ambiguous_margin"
        accepted = False
    else:
        reason = "accepted"
        accepted = True

    return MatchDecision(
        accepted=accepted,
        reason=reason,
        score=float(best.score),
        second_score=second_score,
        margin=gap,
        product_id=int(best.product_id) if accepted else None,
        candidate=best if accepted else None,
        ranked=tuple(ranked),
    )
