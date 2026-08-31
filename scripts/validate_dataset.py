"""Quality checks for the YOLO retail product dataset."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.dataset_common import (
    DUPLICATE_HAMMING,
    IMAGES_DIR,
    LABELS_DIR,
    MIN_IMAGE_SIDE,
    SPLITS,
    average_hash,
    box_is_valid,
    file_md5,
    hamming_distance,
    known_class_ids,
    label_path_for,
    list_images,
    load_image,
    read_yolo_labels,
)


def validate(fail_on_empty: bool = True) -> int:
    errors: list[str] = []
    warnings: list[str] = []
    valid_ids = known_class_ids()
    hashes: list[tuple[int, str, str]] = []
    md5s: dict[str, str] = {}
    counts: dict[str, int] = {}
    class_counts: dict[str, dict[int, int]] = {split: defaultdict(int) for split in SPLITS}

    for split in SPLITS:
        images = list_images(IMAGES_DIR / split)
        labels_dir = LABELS_DIR / split
        counts[split] = len(images)
        label_stems = {
            path.stem
            for path in labels_dir.glob("*.txt")
            if path.name != ".gitkeep"
        }

        for image_path in images:
            image = load_image(image_path)
            if image is None:
                errors.append(f"{split}/{image_path.name}: corrupted or unreadable image")
                continue
            height, width = image.shape[:2]
            if height < MIN_IMAGE_SIDE or width < MIN_IMAGE_SIDE:
                errors.append(
                    f"{split}/{image_path.name}: invalid dimensions {width}x{height}"
                )
            if image.ndim != 3 or image.shape[2] != 3:
                errors.append(f"{split}/{image_path.name}: expected 3-channel BGR image")

            digest = file_md5(image_path)
            previous = md5s.get(digest)
            if previous:
                errors.append(
                    f"duplicate file bytes: {previous} and {split}/{image_path.name}"
                )
            else:
                md5s[digest] = f"{split}/{image_path.name}"

            hashes.append((average_hash(image), split, image_path.name))

            label_file = label_path_for(image_path, labels_dir)
            if not label_file.exists():
                errors.append(f"{split}/{image_path.name}: missing label file")
                continue
            try:
                rows = read_yolo_labels(label_file)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if not rows:
                message = f"{split}/{image_path.name}: empty annotation"
                if fail_on_empty:
                    errors.append(message)
                else:
                    warnings.append(message)
                continue
            for class_id, xc, yc, width_n, height_n in rows:
                if class_id not in valid_ids:
                    errors.append(
                        f"{split}/{image_path.name}: unknown class_id {class_id}"
                    )
                if not box_is_valid(xc, yc, width_n, height_n):
                    errors.append(
                        f"{split}/{image_path.name}: invalid box "
                        f"{class_id} {xc:.4f} {yc:.4f} {width_n:.4f} {height_n:.4f}"
                    )
                else:
                    class_counts[split][class_id] += 1

        orphan_labels = label_stems - {path.stem for path in images}
        for stem in sorted(orphan_labels):
            warnings.append(f"{split}: label without image ({stem}.txt)")

    for index, (hash_a, split_a, name_a) in enumerate(hashes):
        for hash_b, split_b, name_b in hashes[index + 1 :]:
            distance = hamming_distance(hash_a, hash_b)
            if distance > DUPLICATE_HAMMING:
                continue
            message = (
                f"near-duplicate images (hamming={distance}): "
                f"{split_a}/{name_a} ~ {split_b}/{name_b}"
            )
            if distance <= 2 and split_a != split_b:
                errors.append("split leak: " + message)
            elif split_a != split_b:
                warnings.append("similar across splits: " + message)
            else:
                warnings.append(message)

    print("Image counts:", counts)
    print("Class instance counts:")
    for split in SPLITS:
        print(f"  {split}: {dict(class_counts[split])}")
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(" -", warning)
    if errors:
        print("\nErrors:")
        for error in errors:
            print(" -", error)
        print(f"\nFAILED: {len(errors)} error(s), {len(warnings)} warning(s)")
        return 1

    if sum(counts.values()) == 0:
        print("FAILED: dataset is empty")
        return 1

    print(f"\nPASSED: {sum(counts.values())} images, {len(warnings)} warning(s)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Retail Vision dataset.")
    parser.add_argument("--allow-empty-labels", action="store_true")
    args = parser.parse_args()
    return validate(fail_on_empty=not args.allow_empty_labels)


if __name__ == "__main__":
    raise SystemExit(main())
