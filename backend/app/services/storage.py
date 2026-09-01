"""Product image storage abstraction.

Development: local filesystem under products/images/
Production: set STORAGE_BACKEND=s3 with S3_BUCKET / AWS credentials
"""

from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4

import numpy as np

from backend.app.config import (
    AWS_ACCESS_KEY_ID,
    AWS_REGION,
    AWS_SECRET_ACCESS_KEY,
    PRODUCT_IMAGE_DIR,
    ROOT_DIR,
    S3_BUCKET,
    S3_ENDPOINT_URL,
    S3_PREFIX,
    STORAGE_BACKEND,
    STORAGE_DIR,
)


class StorageError(RuntimeError):
    """Raised when product images cannot be saved or retrieved."""


_SAFE_KEY = re.compile(r"[^a-zA-Z0-9._/-]+")


def _relative_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT_DIR.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def resolve_storage_path(storage_key: str) -> Path:
    raw = (storage_key or "").strip().replace("\\", "/")
    if not raw or ".." in raw.split("/"):
        raise StorageError("Image retrieval failed.")
    if raw.startswith("s3:"):
        raise StorageError("Image retrieval failed.")
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT_DIR / path
    resolved = path.resolve()
    allowed_roots = (
        PRODUCT_IMAGE_DIR.resolve(),
        (STORAGE_DIR / "products").resolve(),
        ROOT_DIR.resolve(),
    )
    if not any(str(resolved).startswith(str(root)) for root in allowed_roots):
        raise StorageError("Image retrieval failed.")
    return resolved


def _s3_client():
    try:
        import boto3
    except ImportError as extra:
        raise StorageError("S3 storage requires boto3. Install boto3 or use STORAGE_BACKEND=local.") from extra
    kwargs = {"region_name": AWS_REGION or "us-east-1"}
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        kwargs["aws_access_key_id"] = AWS_ACCESS_KEY_ID
        kwargs["aws_secret_access_key"] = AWS_SECRET_ACCESS_KEY
    if S3_ENDPOINT_URL:
        kwargs["endpoint_url"] = S3_ENDPOINT_URL
    return boto3.client("s3", **kwargs)


def save_product_image(
    product_id: int,
    image_bytes: bytes,
    *,
    image_type: str = "original",
) -> tuple[str, np.ndarray]:
    """Save JPEG bytes and return (storage_key, decoded BGR)."""
    from vision.image_io import decode_bgr, encode_jpeg, write_jpeg

    if not image_bytes:
        raise StorageError("Image upload failed.")
    decoded = decode_bgr(image_bytes)
    if decoded is None:
        raise StorageError("Invalid image.")
    safe_type = _SAFE_KEY.sub("_", (image_type or "original")[:32]) or "original"
    filename = f"{safe_type}_{uuid4().hex[:12]}.jpg"
    product_folder = f"product_{int(product_id):03d}"

    if STORAGE_BACKEND == "s3":
        if not S3_BUCKET:
            raise StorageError("S3_BUCKET is not configured.")
        key = f"{S3_PREFIX}/{product_folder}/{filename}".lstrip("/")
        try:
            encoded = encode_jpeg(decoded, quality=92)
        except Exception as extra:
            raise StorageError("Image save failed.") from extra
        try:
            _s3_client().put_object(
                Bucket=S3_BUCKET,
                Key=key,
                Body=encoded,
                ContentType="image/jpeg",
            )
        except Exception as extra:
            raise StorageError("Image save failed.") from extra
        # Local cache for FileResponse serving.
        cache = STORAGE_DIR / "products" / product_folder
        cache.mkdir(parents=True, exist_ok=True)
        absolute = cache / filename
        absolute.write_bytes(encoded)
        return f"s3:{key}", decoded

    # Preferred layout: storage/products/product_<id>/… (also keep PRODUCT_IMAGE_DIR writable).
    product_folder = STORAGE_DIR / "products" / f"product_{int(product_id):03d}"
    product_folder.mkdir(parents=True, exist_ok=True)
    absolute = product_folder / filename
    try:
        write_jpeg(absolute, decoded, quality=92)
    except Exception as extra:
        raise StorageError("Image save failed.") from extra
    # Mirror under PRODUCT_IMAGE_DIR for older path lookups.
    legacy = PRODUCT_IMAGE_DIR / str(int(product_id))
    try:
        legacy.mkdir(parents=True, exist_ok=True)
        legacy_path = legacy / filename
        if not legacy_path.exists():
            legacy_path.write_bytes(absolute.read_bytes())
    except Exception:
        pass
    return _relative_to_root(absolute).replace("\\", "/"), decoded


def get_product_image(storage_key: str) -> Path:
    raw = (storage_key or "").strip()
    if raw.startswith("s3:"):
        key = raw[3:]
        if not S3_BUCKET:
            raise StorageError("Image retrieval failed.")
        filename = Path(key).name
        cache = STORAGE_DIR / "products" / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        absolute = cache / filename
        if not absolute.exists():
            try:
                _s3_client().download_file(S3_BUCKET, key, str(absolute))
            except Exception as extra:
                raise StorageError("Image retrieval failed.") from extra
        return absolute

    path = resolve_storage_path(storage_key)
    if not path.exists() or not path.is_file():
        raise StorageError("Image retrieval failed.")
    return path


def delete_product_image(storage_key: str) -> None:
    raw = (storage_key or "").strip()
    if raw.startswith("s3:"):
        key = raw[3:]
        if S3_BUCKET:
            try:
                _s3_client().delete_object(Bucket=S3_BUCKET, Key=key)
            except Exception:
                pass
        return
    path = resolve_storage_path(storage_key)
    if path.exists() and path.is_file():
        path.unlink()
