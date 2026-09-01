from __future__ import annotations

from vision.recognition.catalog import ProductIdentity


class EmbeddingRefiner:
    """Legacy hook. Production recognition uses DINOv2 via gallery / scan_service.

    Preferred pipeline:

        Camera → YOLO (detect/crop) → DINOv2 → pgvector → product match
    """

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = bool(enabled)
        if self.enabled:
            raise RuntimeError(
                "EmbeddingRefiner is unused. Visual identity runs through "
                "vision.recognition.gallery (DINOv2 + vector search)."
            )

    def refine(self, crop, identity: ProductIdentity) -> ProductIdentity:
        del crop
        return identity
