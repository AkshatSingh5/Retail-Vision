from __future__ import annotations

from vision.detection.yolo_detector import Detection
from vision.recognition.identity import IdentifiedDetection


def box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class IoUTracker:
    """YOLO-compatible multi-object tracker using IoU + class identity.

    Two physical items of the same SKU get different track_ids. The same
    item across frames keeps one track_id as long as boxes overlap.
    """

    def __init__(self, iou_threshold: float = 0.30) -> None:
        self.iou_threshold = iou_threshold
        self._next_id = 1
        self._last_boxes: dict[int, tuple[int, list[float]]] = {}

    def reset(self) -> None:
        self._next_id = 1
        self._last_boxes.clear()

    def update(
        self,
        detections: list[Detection] | list[IdentifiedDetection],
        active_track_ids: list[int] | None = None,
    ) -> list[tuple[int, Detection | IdentifiedDetection]]:
        """Assign a track_id to each current-frame detection."""
        assigned: list[tuple[int, Detection | IdentifiedDetection]] = []
        used_tracks: set[int] = set()
        candidates = list(self._last_boxes.items())
        if active_track_ids is not None:
            allowed = set(active_track_ids)
            candidates = [(tid, payload) for tid, payload in candidates if tid in allowed]

        unmatched = list(enumerate(detections))
        pairs: list[tuple[float, int, int]] = []
        for det_index, detection in unmatched:
            for track_id, (class_id, bbox) in candidates:
                if int(detection["class_id"]) != class_id:
                    continue
                score = box_iou(detection["bbox"], bbox)
                if score >= self.iou_threshold:
                    pairs.append((score, det_index, track_id))
        pairs.sort(reverse=True)

        taken_dets: set[int] = set()
        for _score, det_index, track_id in pairs:
            if det_index in taken_dets or track_id in used_tracks:
                continue
            taken_dets.add(det_index)
            used_tracks.add(track_id)
            assigned.append((track_id, detections[det_index]))

        for det_index, detection in unmatched:
            if det_index in taken_dets:
                continue
            track_id = self._next_id
            self._next_id += 1
            assigned.append((track_id, detection))

        self._last_boxes = {
            track_id: (int(detection["class_id"]), list(detection["bbox"]))
            for track_id, detection in assigned
        }
        return assigned
