"""Light, train-only geometric/photometric augmentation.

Does not replace real photographs. New files stay in train/ and keep
session IDs out of val/test.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.dataset_common import (
    IMAGES_DIR,
    LABELS_DIR,
    list_images,
    load_image,
    read_yolo_labels,
    save_image,
    write_yolo_labels,
    xyxy_to_yolo,
    yolo_to_xyxy,
)


def _rotate(image: np.ndarray, rows: list[tuple[int, float, float, float, float]], angle: float):
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT_101)
    converted: list[tuple[int, float, float, float, float]] = []
    for class_id, xc, yc, box_w, box_h in rows:
        x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, box_w, box_h, width, height)
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        ones = np.ones((4, 1), dtype=np.float32)
        transformed = matrix @ np.hstack([corners, ones]).T
        xs, ys = transformed[0], transformed[1]
        converted.append((class_id, *xyxy_to_yolo(xs.min(), ys.min(), xs.max(), ys.max(), width, height)))
    return rotated, converted


def _brightness(image: np.ndarray, delta: int) -> np.ndarray:
    return cv2.convertScaleAbs(image, alpha=1.0, beta=delta)


def augment_train() -> None:
    images = [
        path
        for path in list_images(IMAGES_DIR / "train")
        if not path.stem.startswith("aug_")
    ]
    created = 0
    for image_path in images:
        image = load_image(image_path)
        if image is None:
            continue
        label_file = LABELS_DIR / "train" / f"{image_path.stem}.txt"
        rows = read_yolo_labels(label_file)
        if not rows:
            continue
        ops = [
            ("rot15", lambda img, lab: _rotate(img, lab, 15)),
            ("rot30", lambda img, lab: _rotate(img, lab, 30)),
            ("dark", lambda img, lab: (_brightness(img, -40), lab)),
            ("bright", lambda img, lab: (_brightness(img, 40), lab)),
        ]
        for suffix, func in ops:
            out_img, out_rows = func(image, rows)
            stem = f"aug_{suffix}_{image_path.stem}"
            save_image(IMAGES_DIR / "train" / f"{stem}.jpg", out_img)
            write_yolo_labels(LABELS_DIR / "train" / f"{stem}.txt", out_rows)
            created += 1
    print(f"Created {created} train-only augmented images")


def main() -> int:
    parser = argparse.ArgumentParser(description="Controlled train-only augmentation.")
    parser.parse_args()
    augment_train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
