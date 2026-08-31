"""Capture real webcam photos into dataset/raw, organized by session.

Keys:
  0-7   select product class
  n     start a new capture session (prevents train/test leakage)
  SPACE save frame
  q/ESC quit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.dataset_common import (
    RAW_DIR,
    class_names,
    ensure_dataset_dirs,
    load_manifest,
    save_image,
    save_manifest,
)
from vision.inference.camera import Camera, CameraError


def _overlay(frame, class_id: int, names: dict[int, str], session_id: str, saved: int):
    canvas = frame.copy()
    lines = [
        f"Class {class_id}: {names.get(class_id, '?')}",
        f"Session: {session_id}",
        f"Saved: {saved}",
        "0-7 class | n new session | SPACE save | q quit",
    ]
    y = 28
    for line in lines:
        cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y += 28
    return canvas


def capture(start_class: int) -> int:
    ensure_dataset_dirs()
    names = class_names()
    class_id = start_class if start_class in names else 0
    session_id = f"cam_{int(time.time())}"
    manifest = load_manifest()
    saved = 0

    camera = Camera(window_name="Retail Vision capture")
    try:
        camera.open()
    except CameraError as exc:
        print(exc, file=sys.stderr)
        return 1

    try:
        while True:
            frame = camera.read()
            camera.show(_overlay(frame, class_id, names, session_id, saved))
            key = camera.wait_key(1)
            if camera.should_quit(key):
                break
            if ord("0") <= key <= ord("7"):
                candidate = key - ord("0")
                if candidate in names:
                    class_id = candidate
            elif key in {ord("n"), ord("N")}:
                session_id = f"cam_{int(time.time())}"
                print(f"New session {session_id}")
            elif key == 32:
                filename = f"c{class_id}_{session_id}_{saved:03d}.jpg"
                dest = RAW_DIR / "images" / filename
                save_image(dest, frame)
                manifest.append(
                    {
                        "file": filename,
                        "class_id": class_id,
                        "class_name": names[class_id],
                        "session_id": f"{class_id}_{session_id}",
                        "source": "webcam",
                        "labeled": False,
                    }
                )
                save_manifest(manifest)
                saved += 1
                print(f"Saved {filename}")
    finally:
        camera.release()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture real product photos from the webcam.")
    parser.add_argument("--class-id", type=int, default=0)
    args = parser.parse_args()
    return capture(args.class_id)


if __name__ == "__main__":
    raise SystemExit(main())
