"""Split labeled raw images into train/val/test by capture session.

Sessions never cross splits. That keeps near-duplicate pack shots, including
multiple photos of the same barcode, out of both train and test.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.dataset_common import (
    IMAGES_DIR,
    LABELS_DIR,
    RAW_DIR,
    SPLITS,
    ensure_dataset_dirs,
    load_manifest,
    write_data_yaml,
)


def _clear_split_dirs() -> None:
    for split in SPLITS:
        for folder in (IMAGES_DIR / split, LABELS_DIR / split):
            folder.mkdir(parents=True, exist_ok=True)
            for path in folder.iterdir():
                if path.name == ".gitkeep":
                    continue
                if path.is_file():
                    path.unlink()


def _assign_sessions(sessions: list[str], rng: random.Random) -> dict[str, str]:
    ordered = list(sessions)
    rng.shuffle(ordered)
    count = len(ordered)
    if count == 1:
        return {ordered[0]: "train"}
    if count == 2:
        return {ordered[0]: "train", ordered[1]: "val"}

    n_test = max(1, round(count * 0.10))
    n_val = max(1, round(count * 0.20))
    n_train = count - n_val - n_test
    if n_train < 1:
        n_train = 1
        n_test = 1
        n_val = count - 2

    mapping: dict[str, str] = {}
    for session in ordered[:n_train]:
        mapping[session] = "train"
    for session in ordered[n_train : n_train + n_val]:
        mapping[session] = "val"
    for session in ordered[n_train + n_val :]:
        mapping[session] = "test"
    return mapping


def split_dataset(seed: int) -> None:
    write_data_yaml()
    ensure_dataset_dirs()
    manifest = [entry for entry in load_manifest() if entry.get("labeled")]
    if not manifest:
        raise SystemExit("No labeled raw images. Run scripts/auto_label_raw.py first.")

    by_class: dict[int, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for entry in manifest:
        by_class[int(entry["class_id"])][str(entry["session_id"])].append(entry)

    rng = random.Random(seed)
    _clear_split_dirs()
    counts = {split: 0 for split in SPLITS}

    for class_id, sessions in sorted(by_class.items()):
        assignment = _assign_sessions(list(sessions), rng)
        print(f"class {class_id}: {len(sessions)} sessions")
        split_summary: dict[str, int] = defaultdict(int)
        for session_id, entries in sessions.items():
            split = assignment[session_id]
            split_summary[split] += 1
            for entry in entries:
                source_image = RAW_DIR / "images" / entry["file"]
                source_label = RAW_DIR / "labels" / f"{Path(entry['file']).stem}.txt"
                if not source_image.exists() or not source_label.exists():
                    continue
                dest_name = entry["file"]
                shutil.copy2(source_image, IMAGES_DIR / split / dest_name)
                shutil.copy2(source_label, LABELS_DIR / split / f"{Path(dest_name).stem}.txt")
                counts[split] += 1
        print(f"  session split {dict(split_summary)}")

    print(f"Copied images: {counts}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Session-aware 70/20/10 dataset split.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split_dataset(args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
