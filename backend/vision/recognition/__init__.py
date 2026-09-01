"""Vision recognition package.

Import concrete modules directly (e.g. `vision.recognition.gallery`) to avoid
circular imports with backend services.
"""

from vision.recognition.catalog import ProductCatalog, ProductIdentity, load_catalog

__all__ = [
    "ProductCatalog",
    "ProductIdentity",
    "load_catalog",
]
