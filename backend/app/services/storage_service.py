"""Storage service re-export. Implementation lives in `backend.app.services.storage`."""

from backend.app.services.storage import (
    StorageError,
    StorageService,
    delete_product_image,
    get_product_image,
    get_storage_service,
    save_product_image,
)

__all__ = [
    "StorageError",
    "StorageService",
    "delete_product_image",
    "get_product_image",
    "get_storage_service",
    "save_product_image",
]
