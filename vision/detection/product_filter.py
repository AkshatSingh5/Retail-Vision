from __future__ import annotations

import re
from functools import lru_cache

from vision.detection.types import Detection

# Human / scene classes that must never enter the retail pipeline.
_NON_PRODUCT_EXACT = frozenset(
    {
        "person",
        "face",
        "hand",
        "arm",
        "head",
        "human",
        "body",
        "foot",
        "leg",
        "finger",
        "hair",
        "eye",
        "nose",
        "mouth",
        "torso",
        "neck",
        "shoulder",
        "knee",
        "elbow",
        "wrist",
        "background",
        "wall",
        "floor",
        "ceiling",
        "sky",
        "road",
        "tree",
        "plant",
        "grass",
    }
)

# Generic COCO-style objects — not retail SKUs; never show ADD NEW PRODUCT for these.
_GENERIC_NON_RETAIL = frozenset(
    {
        "chair",
        "couch",
        "bed",
        "table",
        "desk",
        "laptop",
        "mouse",
        "keyboard",
        "cell phone",
        "tv",
        "monitor",
        "remote",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "car",
        "truck",
        "bus",
        "bicycle",
        "motorcycle",
        "train",
        "dog",
        "cat",
        "bird",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
    }
)

_HUMAN_PATTERN = re.compile(
    r"\b(person|people|human|face|hand|hands|arm|arms|head|body|foot|feet|leg|legs)\b",
    re.IGNORECASE,
)


def class_name_for(class_names: dict[int, str], class_id: int) -> str:
    return str(class_names.get(int(class_id), "object"))


def is_retail_trained_model(class_names: dict[int, str]) -> bool:
    """True when weights look like a Retail Vision product detector, not COCO."""
    if not class_names:
        return False
    lowered = {str(name).lower() for name in class_names.values()}
    if "person" in lowered:
        return False
    # COCO and similar general detectors expose dozens of classes.
    if len(class_names) > 20:
        return False
    return True


@lru_cache(maxsize=1)
def _catalog_class_ids() -> frozenset[int]:
    try:
        from vision.recognition.catalog import load_catalog

        return frozenset(load_catalog().by_class_id.keys())
    except Exception:
        return frozenset()


def is_non_product_class(class_name: str) -> bool:
    normalized = class_name.strip().lower().replace("_", " ")
    if normalized in _NON_PRODUCT_EXACT:
        return True
    if _HUMAN_PATTERN.search(normalized):
        return True
    return False


def is_retail_product_detection(
    detection: Detection,
    class_names: dict[int, str],
    *,
    log: bool = True,
    log_process: bool = False,
) -> bool:
    """Return True only when a detection may enter recognition / cart logic."""
    class_id = int(detection["class_id"])
    class_name = class_name_for(class_names, class_id)
    confidence = float(detection["confidence"])

    if is_non_product_class(class_name):
        if log:
            print(
                f"Detected: class_id={class_id} class_name={class_name} "
                f"confidence={confidence:.2f} → IGNORE: non-product (human/scene)"
            )
        return False

    if is_retail_trained_model(class_names):
        if log and log_process:
            print(
                f"Detected: class_id={class_id} class_name={class_name} "
                f"confidence={confidence:.2f} → PROCESS: retail product"
            )
        return True

    # Generic pretrained model — only allow classes mapped in the retail catalog.
    if class_name.strip().lower() in _GENERIC_NON_RETAIL:
        if log:
            print(
                f"Detected: class_id={class_id} class_name={class_name} "
                f"confidence={confidence:.2f} → IGNORE: generic non-retail object"
            )
        return False

    if class_id in _catalog_class_ids():
        if log and log_process:
            print(
                f"Detected: class_id={class_id} class_name={class_name} "
                f"confidence={confidence:.2f} → PROCESS: catalog-mapped class"
            )
        return True

    if log:
        print(
            f"Detected: class_id={class_id} class_name={class_name} "
            f"confidence={confidence:.2f} → IGNORE: not a retail product class"
        )
    return False


def filter_retail_detections(
    detections: list[Detection],
    class_names: dict[int, str],
    *,
    log: bool = True,
    log_process: bool = False,
) -> list[Detection]:
    filtered = [
        detection
        for detection in detections
        if is_retail_product_detection(
            detection,
            class_names,
            log=log,
            log_process=log_process,
        )
    ]
    if log and log_process and filtered:
        names = ", ".join(
            class_name_for(class_names, int(item["class_id"])) for item in filtered[:4]
        )
        extra = f" (+{len(filtered) - 4} more)" if len(filtered) > 4 else ""
        print(f"Retail pipeline: {len(filtered)} product detection(s): {names}{extra}")
    return filtered


# COCO / generic classes that often wrap retail packaging — keep for scan crops.
# Exclude bags/remotes/etc.: COCO often tags the clerk's handbag or a remote-sized
# fragment while the actual pack sits elsewhere in frame.
_SCAN_CROP_ALLOW = frozenset(
    {
        "bottle",
        "cup",
        "bowl",
        "wine glass",
        "book",
        "cell phone",
        "vase",
        "sports ball",
        "teddy bear",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
    }
)

# Minimum box size so tiny false positives (e.g. 48px "bottle") are not used as crops.
_MIN_SCAN_BOX_AREA_FRAC = 0.03
_MIN_SCAN_BOX_SIDE = 40


def detection_area_fraction(detection: Detection, frame_shape: tuple[int, ...] | None = None) -> float:
    bbox = detection.get("bbox") or []
    if len(bbox) != 4:
        return 0.0
    x1, y1, x2, y2 = (float(v) for v in bbox)
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if frame_shape is None or len(frame_shape) < 2:
        return area
    height, width = int(frame_shape[0]), int(frame_shape[1])
    total = float(max(1, height * width))
    return area / total


def is_usable_scan_box(
    detection: Detection,
    frame_shape: tuple[int, ...] | None = None,
    *,
    log: bool = False,
) -> bool:
    """Reject tiny / degenerate boxes that destroy recognition."""
    bbox = detection.get("bbox") or []
    if len(bbox) != 4:
        return False
    x1, y1, x2, y2 = (float(v) for v in bbox)
    bw, bh = abs(x2 - x1), abs(y2 - y1)
    if bw < _MIN_SCAN_BOX_SIDE or bh < _MIN_SCAN_BOX_SIDE:
        if log:
            print(f"[YOLO] bbox too small ({bw:.0f}x{bh:.0f}) → IGNORE crop")
        return False
    frac = detection_area_fraction(detection, frame_shape)
    if frame_shape is not None and frac < _MIN_SCAN_BOX_AREA_FRAC:
        if log:
            print(f"[YOLO] bbox area {frac:.3%} of frame → IGNORE crop (too small)")
        return False
    return True


def scan_crop_priority(detection: Detection, frame_shape: tuple[int, ...] | None = None) -> float:
    """Prefer large, confident packaging boxes over tiny high-conf fragments."""
    conf = float(detection.get("confidence") or 0.0)
    frac = detection_area_fraction(detection, frame_shape)
    return conf * (0.35 + 0.65 * min(1.0, frac / 0.25))


def is_scan_crop_detection(
    detection: Detection,
    class_names: dict[int, str],
    *,
    log: bool = False,
    frame_shape: tuple[int, ...] | None = None,
) -> bool:
    """Looser filter for one-shot Scan Product: keep packaging-like boxes for cropping."""
    class_id = int(detection["class_id"])
    class_name = class_name_for(class_names, class_id)
    confidence = float(detection["confidence"])

    if is_non_product_class(class_name):
        if log:
            print(
                f"[YOLO] class_id={class_id} {class_name} conf={confidence:.2f} → IGNORE crop (human/scene)"
            )
        return False

    if is_retail_trained_model(class_names):
        return is_usable_scan_box(detection, frame_shape, log=log)

    if not is_usable_scan_box(detection, frame_shape, log=log):
        return False

    lowered = class_name.strip().lower()
    if lowered in _SCAN_CROP_ALLOW or class_id in _catalog_class_ids():
        if log:
            print(
                f"[YOLO] class_id={class_id} {class_name} conf={confidence:.2f} → KEEP crop for recognition"
            )
        return True

    if lowered in _GENERIC_NON_RETAIL:
        if log:
            print(
                f"[YOLO] class_id={class_id} {class_name} conf={confidence:.2f} → IGNORE crop (non-retail)"
            )
        return False

    # Unknown class name — still useful as a product crop candidate if large enough.
    return True


def filter_scan_detections(
    detections: list[Detection],
    class_names: dict[int, str],
    *,
    log: bool = True,
    frame_shape: tuple[int, ...] | None = None,
) -> list[Detection]:
    filtered = [
        detection
        for detection in detections
        if is_scan_crop_detection(detection, class_names, log=log, frame_shape=frame_shape)
    ]
    filtered.sort(key=lambda item: scan_crop_priority(item, frame_shape), reverse=True)
    return filtered

