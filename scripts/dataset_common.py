from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from backend.app.config import ROOT_DIR

REGISTRY_PATH = ROOT_DIR / "products" / "registry.yaml"
DATASET_DIR = ROOT_DIR / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"
RAW_DIR = DATASET_DIR / "raw"
MANIFEST_PATH = RAW_DIR / "manifest.json"
DATA_YAML_PATH = ROOT_DIR / "data.yaml"
DATASET_DATA_YAML_PATH = DATASET_DIR / "data.yaml"
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_IMAGE_SIDE = 64
HASH_SIZE = 16
DUPLICATE_HAMMING = 6


def detect_device(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    return "0" if torch.cuda.is_available() else "cpu"


def load_registry(path: Path = REGISTRY_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    products = payload.get("products") or []
    if not products:
        raise ValueError(f"No products found in {path}")
    products = sorted(products, key=lambda item: int(item["class_id"]))
    ids = [int(item["class_id"]) for item in products]
    if ids != list(range(len(ids))):
        raise ValueError("class_id values must be unique and contiguous from 0")
    return products


def class_names(products: list[dict] | None = None) -> dict[int, str]:
    products = products if products is not None else load_registry()
    return {int(item["class_id"]): str(item["name"]) for item in products}


def known_class_ids(products: list[dict] | None = None) -> set[int]:
    return set(class_names(products))


def write_data_yaml(products: list[dict] | None = None) -> None:
    products = products if products is not None else load_registry()
    names = {int(item["class_id"]): str(item["name"]) for item in products}
    payload = {
        "path": "dataset",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(names),
        "names": names,
    }
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    DATA_YAML_PATH.write_text(text, encoding="utf-8")
    DATASET_DATA_YAML_PATH.write_text(text, encoding="utf-8")


def ensure_dataset_dirs() -> None:
    for split in SPLITS:
        (IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
        (LABELS_DIR / split).mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "images").mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "labels").mkdir(parents=True, exist_ok=True)


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and path.name != ".gitkeep"
    )


def label_path_for(image_path: Path, labels_dir: Path | None = None) -> Path:
    if labels_dir is None:
        labels_dir = image_path.parent.parent.parent / "labels" / image_path.parent.name
        if image_path.parent.parent.name == "raw":
            labels_dir = RAW_DIR / "labels"
    return labels_dir / f"{image_path.stem}.txt"


def read_yolo_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    if not path.exists():
        return []
    rows: list[tuple[int, float, float, float, float]] = []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return rows
    for line_no, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_no} expected 5 values, got {len(parts)}")
        class_id = int(float(parts[0]))
        xc, yc, width, height = (float(value) for value in parts[1:])
        rows.append((class_id, xc, yc, width, height))
    return rows


def write_yolo_labels(path: Path, rows: list[tuple[int, float, float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{class_id} {xc:.6f} {yc:.6f} {width:.6f} {height:.6f}"
        for class_id, xc, yc, width, height in rows
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def xyxy_to_yolo(
    x1: float, y1: float, x2: float, y2: float, image_width: int, image_height: int
) -> tuple[float, float, float, float]:
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    xc = ((x1 + x2) / 2.0) / image_width
    yc = ((y1 + y2) / 2.0) / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height
    return _clip_box(xc, yc, width, height)


def yolo_to_xyxy(
    xc: float, yc: float, width: float, height: float, image_width: int, image_height: int
) -> tuple[int, int, int, int]:
    x1 = (xc - width / 2.0) * image_width
    y1 = (yc - height / 2.0) * image_height
    x2 = (xc + width / 2.0) * image_width
    y2 = (yc + height / 2.0) * image_height
    return (
        int(round(max(0, x1))),
        int(round(max(0, y1))),
        int(round(min(image_width - 1, x2))),
        int(round(min(image_height - 1, y2))),
    )


def _clip_box(xc: float, yc: float, width: float, height: float) -> tuple[float, float, float, float]:
    width = min(max(width, 1e-6), 1.0)
    height = min(max(height, 1e-6), 1.0)
    xc = min(max(xc, width / 2.0), 1.0 - width / 2.0)
    yc = min(max(yc, height / 2.0), 1.0 - height / 2.0)
    return xc, yc, width, height


def box_is_valid(xc: float, yc: float, width: float, height: float) -> bool:
    if not all(0.0 <= value <= 1.0 for value in (xc, yc, width, height)):
        return False
    if width <= 0.0 or height <= 0.0:
        return False
    x1 = xc - width / 2.0
    y1 = yc - height / 2.0
    x2 = xc + width / 2.0
    y2 = yc + height / 2.0
    return x1 >= -1e-3 and y1 >= -1e-3 and x2 <= 1.0 + 1e-3 and y2 <= 1.0 + 1e-3


def load_image(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError(f"Could not encode {path}")
    encoded.tofile(str(path))


def average_hash(image: np.ndarray, hash_size: int = HASH_SIZE) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    bits = (resized > resized.mean()).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        return []
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def save_manifest(entries: list[dict]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def color_for_class(class_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(class_id + 21)
    b, g, r = (int(value) for value in rng.integers(40, 255, size=3))
    return b, g, r
