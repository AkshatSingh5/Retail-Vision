"""Webcam test for a trained retail detector.

Default: interactive live window (Q/ESC to quit).
Use --frames N to capture a short recorded sample without waiting.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.config import ROOT_DIR as PROJECT_ROOT
from scripts.dataset_common import class_names
from vision.detection.yolo_detector import YOLODetector
from vision.inference.camera import Camera, CameraError


def test_camera(model_path: Path | None, frames: int, save_dir: Path) -> int:
    names = class_names()
    detector = YOLODetector(model_path=model_path) if model_path else YOLODetector()
    print(f"Model: {detector.model_path}")
    print(f"Device: {detector.device}")
    camera = Camera(window_name="Retail Vision — trained YOLO26m")
    try:
        camera.open()
    except CameraError as exc:
        print(f"Camera error: {exc}", file=sys.stderr)
        return 1

    save_dir.mkdir(parents=True, exist_ok=True)
    records = []
    fps = 0.0
    previous = time.perf_counter()
    captured = 0
    try:
        while True:
            frame = camera.read()
            detections, latency_ms = detector.detect(frame)
            annotated = detector.annotate(frame, detections, fps=fps, latency_ms=latency_ms)
            now = time.perf_counter()
            elapsed = now - previous
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            previous = now
            camera.show(annotated)
            if frames > 0:
                image_path = save_dir / f"frame_{captured:03d}.jpg"
                cv2.imwrite(str(image_path), annotated)
                records.append(
                    {
                        "file": image_path.name,
                        "latency_ms": round(latency_ms, 1),
                        "fps": round(fps, 2),
                        "detections": [
                            {
                                **det,
                                "class_name": names.get(det["class_id"], detector.names.get(det["class_id"], str(det["class_id"]))),
                            }
                            for det in detections
                        ],
                    }
                )
                captured += 1
                if captured >= frames:
                    break
            if camera.should_quit(camera.wait_key(1)):
                break
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        camera.release()
    if records:
        (save_dir / "realtime_results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"Saved {len(records)} frames to {save_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Real-world webcam test of the trained model.")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--frames", type=int, default=0, help="Capture N frames then exit. 0 = interactive.")
    parser.add_argument("--save-dir", type=Path, default=PROJECT_ROOT / "logs" / "realtime_test")
    args = parser.parse_args()
    return test_camera(args.model, args.frames, args.save_dir)


if __name__ == "__main__":
    raise SystemExit(main())
