"""Preview random labeled images with bounding boxes and class names."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.config import ROOT_DIR as PROJECT_ROOT
from scripts.dataset_common import (
    IMAGES_DIR,
    LABELS_DIR,
    SPLITS,
    class_names,
    color_for_class,
    label_path_for,
    list_images,
    load_image,
    read_yolo_labels,
    yolo_to_xyxy,
)


def _draw(image: np.ndarray, image_path: Path, names: dict[int, str]) -> np.ndarray:
    canvas = image.copy()
    split = image_path.parent.name
    label_file = label_path_for(image_path, LABELS_DIR / split)
    height, width = canvas.shape[:2]
    rows = read_yolo_labels(label_file) if label_file.exists() else []
    for class_id, xc, yc, box_w, box_h in rows:
        x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, box_w, box_h, width, height)
        color = color_for_class(class_id)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        caption = names.get(class_id, f"class {class_id}")
        cv2.putText(
            canvas,
            caption,
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    header = f"{split}/{image_path.name}"
    cv2.putText(
        canvas,
        header,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return canvas


def _mosaic(images: list[np.ndarray], cols: int = 3, cell: int = 360) -> np.ndarray:
    if not images:
        return np.zeros((cell, cell, 3), dtype=np.uint8)
    rows = int(np.ceil(len(images) / cols))
    mosaic = np.zeros((rows * cell, cols * cell, 3), dtype=np.uint8)
    for index, image in enumerate(images):
        resized = cv2.resize(image, (cell, cell))
        row, col = divmod(index, cols)
        mosaic[row * cell : (row + 1) * cell, col * cell : (col + 1) * cell] = resized
    return mosaic


def visualize(count: int, seed: int, show: bool, save_path: Path) -> int:
    names = class_names()
    pool: list[Path] = []
    for split in SPLITS:
        pool.extend(list_images(IMAGES_DIR / split))
    if not pool:
        print("No labeled split images found.", file=sys.stderr)
        return 1

    rng = random.Random(seed)
    chosen = pool if len(pool) <= count else rng.sample(pool, count)
    rendered = [_draw(load_image(path), path, names) for path in chosen if load_image(path) is not None]
    preview = _mosaic(rendered)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), preview)
    print(f"Wrote preview mosaic: {save_path} ({len(rendered)} samples)")

    if show:
        for image, path in zip(rendered, chosen):
            cv2.imshow("Retail Vision dataset", image)
            print(path)
            key = cv2.waitKey(0) & 0xFF
            if key in {ord("q"), 27}:
                break
        cv2.destroyAllWindows()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize dataset annotations.")
    parser.add_argument("--count", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show", action="store_true", help="Open an interactive window")
    parser.add_argument(
        "--save",
        type=Path,
        default=PROJECT_ROOT / "logs" / "dataset_preview.jpg",
    )
    args = parser.parse_args()
    return visualize(args.count, args.seed, args.show, args.save)


if __name__ == "__main__":
    raise SystemExit(main())
