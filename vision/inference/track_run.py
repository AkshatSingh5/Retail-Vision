from __future__ import annotations

import json
import sys
import time

from backend.app.config import (
    ENABLE_EMBEDDING_REFINEMENT,
    STABLE_FRAMES,
    TRACK_IOU_THRESHOLD,
    TRACK_MAX_MISSING,
    USE_BYTETRACK,
)
from vision.inference.camera import Camera, CameraError
from vision.tracking.pipeline import TrackingPipeline


def run() -> int:
    print("Retail Vision — product tracking")
    print(f"Stable frames before cart: {STABLE_FRAMES}")
    print(f"Lost after missed frames: {TRACK_MAX_MISSING}")
    print(f"Tracker: {'ByteTrack' if USE_BYTETRACK else 'IoU'}")
    print(f"Embedding refinement: {ENABLE_EMBEDDING_REFINEMENT}")
    print("Opening camera... press Q or ESC to quit.")

    pipeline = TrackingPipeline(
        use_ultralytics_track=USE_BYTETRACK,
        stable_frames=STABLE_FRAMES,
        max_missing=TRACK_MAX_MISSING,
        track_iou_threshold=TRACK_IOU_THRESHOLD,
        enable_embedding=ENABLE_EMBEDDING_REFINEMENT,
    )
    print(f"Model: {pipeline.detector.model_path}")
    print(f"Device: {pipeline.detector.device}")

    camera = Camera()
    try:
        camera.open()
    except CameraError as exc:
        print(f"Camera error: {exc}", file=sys.stderr)
        return 1

    fps = 0.0
    previous = time.perf_counter()
    last_print = 0
    try:
        while True:
            try:
                frame = camera.read()
            except CameraError as exc:
                print(f"Camera error: {exc}", file=sys.stderr)
                return 1

            result = pipeline.process(frame)
            annotated = pipeline.annotate(frame, result, fps=fps)

            now = time.perf_counter()
            elapsed = now - previous
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            previous = now

            if result["prices"] and now - last_print > 1.0:
                print(json.dumps(result["prices"], separators=(",", ":")))
                last_print = now
            elif result["public"] and now - last_print > 1.0:
                print(json.dumps(result["public"], separators=(",", ":")))
                last_print = now

            camera.show(annotated)
            if camera.should_quit(camera.wait_key(1)):
                break
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        camera.release()
        print("Camera released.")
        cart = pipeline.manager.cart_by_name()
        print("Session cart:", json.dumps(cart, indent=2) if cart else "{}")
    return 0


if __name__ == "__main__":
    sys.exit(run())
