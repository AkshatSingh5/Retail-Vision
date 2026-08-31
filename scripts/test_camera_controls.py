"""Camera auto-detection on CAMERA ON. Run from the project root."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.camera_hub import CameraHub
from vision.detection.product_filter import (
    filter_retail_detections,
    is_non_product_class,
    is_retail_trained_model,
)

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        extra = f" -- {detail}" if detail else ""
        print(f"  FAIL  {name}{extra}")


def main() -> int:
    print("Retail Vision - camera auto-detection")
    hub = CameraHub()
    check("initial camera off", hub.camera_active is False)
    check("initial detection off", hub.detection_active is False)

    info = hub.start_camera()
    check("camera start sets detection_loading", info.get("detection_loading") is True)
    check("camera start keeps detection_active false until model ready", info.get("detection_active") is False)
    hub.stop_camera()

    retail_names = {
        0: "Coca-Cola 500ml",
        1: "Lays Classic",
    }
    check("retail model recognized", is_retail_trained_model(retail_names))
    check("person class flagged non-product", is_non_product_class("person"))
    check("hand class flagged non-product", is_non_product_class("hand"))

    coco_names = {0: "person", 1: "bottle", 2: "chair"}
    person_only = [{"class_id": 0, "confidence": 0.94, "bbox": [0, 0, 10, 10]}]
    filtered = filter_retail_detections(person_only, coco_names)
    check("person detections filtered out", len(filtered) == 0)

    with TestClient(app) as client:
        status = client.get("/pos/status").json()
        check("status camera_active false initially", status.get("camera_active") is False)
        check("status detection_active false initially", status.get("detection_active") is False)

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
