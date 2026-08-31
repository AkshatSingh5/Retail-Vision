from __future__ import annotations

import sys
import time

from vision.detection.yolo_detector import YOLODetector
from vision.inference.camera import Camera, CameraError


def run() -> int:
    print("Retail Vision — loading YOLO26m...")
    detector = YOLODetector()
    print(f"Model: {detector.model_path}")
    print(f"Device: {detector.device}")
    print(f"Confidence threshold: {detector.confidence_threshold:.2f}")
    print("Opening camera... press Q or ESC to quit.")

    camera = Camera()
    try:
        camera.open()
    except CameraError as exc:
        print(f"Camera error: {exc}", file=sys.stderr)
        return 1

    fps = 0.0
    previous = time.perf_counter()
    try:
        while True:
            try:
                frame = camera.read()
            except CameraError as exc:
                print(f"Camera error: {exc}", file=sys.stderr)
                return 1

            detections, latency_ms = detector.detect(frame)
            annotated = detector.annotate(
                frame,
                detections,
                fps=fps,
                latency_ms=latency_ms,
            )

            now = time.perf_counter()
            elapsed = now - previous
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            previous = now

            camera.show(annotated)
            if camera.should_quit(camera.wait_key(1)):
                break
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        camera.release()
        print("Camera released.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
