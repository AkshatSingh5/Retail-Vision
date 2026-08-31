"""Interactive YOLO bounding-box annotator for dataset/raw images.

Drag a box with the mouse. Keys:
  0-7   set class for the next box
  u     undo last box
  s     save and next
  d     skip image
  q     quit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.dataset_common import (
    RAW_DIR,
    class_names,
    color_for_class,
    load_image,
    load_manifest,
    save_manifest,
    write_yolo_labels,
    xyxy_to_yolo,
)


class Annotator:
    def __init__(self, image: np.ndarray, class_id: int, names: dict[int, str]) -> None:
        self.base = image
        self.class_id = class_id
        self.names = names
        self.boxes: list[tuple[int, int, int, int, int]] = []
        self.dragging = False
        self.start = (0, 0)
        self.current = (0, 0)

    def on_mouse(self, event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.start = (x, y)
            self.current = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            x1, y1 = self.start
            x2, y2 = x, y
            if abs(x2 - x1) > 4 and abs(y2 - y1) > 4:
                self.boxes.append((self.class_id, x1, y1, x2, y2))

    def render(self) -> np.ndarray:
        canvas = self.base.copy()
        height, width = canvas.shape[:2]
        for class_id, x1, y1, x2, y2 in self.boxes:
            color = color_for_class(class_id)
            cv2.rectangle(canvas, (min(x1, x2), min(y1, y2)), (max(x1, x2), max(y1, y2)), color, 2)
            cv2.putText(
                canvas,
                self.names.get(class_id, str(class_id)),
                (min(x1, x2), max(16, min(y1, y2) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )
        if self.dragging:
            cv2.rectangle(canvas, self.start, self.current, (0, 255, 255), 1)
        cv2.putText(
            canvas,
            f"Class {self.class_id}: {self.names.get(self.class_id, '?')} | drag box | s save | u undo | d skip",
            (10, height - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        return canvas

    def to_yolo(self) -> list[tuple[int, float, float, float, float]]:
        height, width = self.base.shape[:2]
        rows = []
        for class_id, x1, y1, x2, y2 in self.boxes:
            rows.append((class_id, *xyxy_to_yolo(x1, y1, x2, y2, width, height)))
        return rows


def annotate() -> int:
    names = class_names()
    manifest = load_manifest()
    pending = [entry for entry in manifest if not entry.get("labeled")]
    if not pending:
        print("No unlabeled raw images.")
        return 0

    window = "Retail Vision annotate"
    cv2.namedWindow(window)
    for entry in pending:
        image = load_image(RAW_DIR / "images" / entry["file"])
        if image is None:
            continue
        state = Annotator(image, int(entry["class_id"]), names)
        cv2.setMouseCallback(window, state.on_mouse)
        while True:
            cv2.imshow(window, state.render())
            key = cv2.waitKey(16) & 0xFF
            if key in {ord("q"), 27}:
                cv2.destroyAllWindows()
                save_manifest(manifest)
                return 0
            if ord("0") <= key <= ord("7"):
                candidate = key - ord("0")
                if candidate in names:
                    state.class_id = candidate
            elif key == ord("u"):
                if state.boxes:
                    state.boxes.pop()
            elif key == ord("d"):
                break
            elif key == ord("s"):
                rows = state.to_yolo()
                if not rows:
                    print("Draw at least one box before saving.")
                    continue
                write_yolo_labels(RAW_DIR / "labels" / f"{Path(entry['file']).stem}.txt", rows)
                entry["labeled"] = True
                save_manifest(manifest)
                print(f"Saved labels for {entry['file']}")
                break
    cv2.destroyAllWindows()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Manually annotate raw product images.")
    parser.parse_args()
    return annotate()


if __name__ == "__main__":
    raise SystemExit(main())
