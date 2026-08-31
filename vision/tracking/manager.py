from __future__ import annotations

from collections import Counter

from vision.recognition.identity import IdentifiedDetection
from vision.tracking.iou_tracker import IoUTracker
from vision.tracking.tracks import Track, TrackedProduct, TrackState


class TrackManager:
    """Temporal state, confidence stabilization, and duplicate-safe cart counts."""

    def __init__(
        self,
        stable_frames: int = 5,
        max_missing: int = 20,
        iou_threshold: float = 0.30,
        confidence_ema: float = 0.4,
    ) -> None:
        if stable_frames < 1:
            raise ValueError("stable_frames must be >= 1")
        if max_missing < 1:
            raise ValueError("max_missing must be >= 1")
        self.stable_frames = stable_frames
        self.max_missing = max_missing
        self.confidence_ema = confidence_ema
        self.associator = IoUTracker(iou_threshold=iou_threshold)
        self.tracks: dict[int, Track] = {}
        self.frame_index = 0
        self.cart_track_ids: set[int] = set()

    def reset(self) -> None:
        self.associator.reset()
        self.tracks.clear()
        self.frame_index = 0
        self.cart_track_ids.clear()

    def update(
        self,
        detections: list[IdentifiedDetection],
        assigned: list[tuple[int, IdentifiedDetection]] | None = None,
    ) -> list[TrackedProduct]:
        self.frame_index += 1
        active = [track.track_id for track in self.tracks.values() if track.state != TrackState.EXITED]
        pairs = assigned if assigned is not None else self.associator.update(detections, active_track_ids=active)
        matched_ids = {track_id for track_id, _detection in pairs}

        for track_id, detection in pairs:
            self._upsert(track_id, detection)

        for track in list(self.tracks.values()):
            if track.state == TrackState.EXITED:
                continue
            if track.track_id in matched_ids:
                continue
            track.misses += 1
            track.consecutive_hits = 0
            if track.confirmed:
                track.state = TrackState.LEAVING
            if track.misses >= self.max_missing:
                track.state = TrackState.EXITED

        return [track.to_output() for track in self.active_tracks()]

    def _upsert(self, track_id: int, detection: IdentifiedDetection) -> Track:
        existing = self.tracks.get(track_id)
        confidence = float(detection["confidence"])
        if existing is None:
            confirmed = self.stable_frames <= 1
            track = Track(
                track_id=track_id,
                class_id=int(detection["class_id"]),
                product_id=int(detection["product_id"]),
                sku=str(detection["sku"]),
                product_name=str(detection["product_name"]),
                bbox=list(detection["bbox"]),
                confidence=confidence,
                hits=1,
                consecutive_hits=1,
                misses=0,
                confirmed=confirmed,
                state=TrackState.VISIBLE if confirmed else TrackState.ENTERING,
                first_frame=self.frame_index,
                last_frame=self.frame_index,
                history=[confidence],
                price=detection.get("price"),
                tax_rate=detection.get("tax_rate"),
                is_unknown=bool(detection.get("is_unknown")),
            )
            self.tracks[track_id] = track
            if confirmed and not track.is_unknown:
                self.cart_track_ids.add(track_id)
            return track

        existing.bbox = list(detection["bbox"])
        existing.confidence = (
            self.confidence_ema * confidence + (1.0 - self.confidence_ema) * existing.confidence
        )
        existing.hits += 1
        existing.consecutive_hits += 1
        existing.misses = 0
        existing.last_frame = self.frame_index
        existing.history.append(confidence)
        if len(existing.history) > 30:
            existing.history = existing.history[-30:]
        if not existing.locked:
            existing.class_id = int(detection["class_id"])
            existing.product_id = int(detection["product_id"])
            existing.sku = str(detection["sku"])
            existing.product_name = str(detection["product_name"])
            existing.price = detection.get("price")
            existing.tax_rate = detection.get("tax_rate")
            existing.is_unknown = bool(detection.get("is_unknown", existing.is_unknown))
        if not existing.confirmed and existing.consecutive_hits >= self.stable_frames:
            existing.confirmed = True
            if not existing.is_unknown:
                self.cart_track_ids.add(track_id)
        existing.state = TrackState.VISIBLE if existing.confirmed else TrackState.ENTERING
        return existing

    def bind_product(self, track_id: int, product: dict) -> Track | None:
        track = self.tracks.get(int(track_id))
        if track is None:
            return None
        track.product_id = int(product["id"])
        track.sku = str(product["sku"])
        track.product_name = str(product["name"])
        track.price = product.get("price")
        track.tax_rate = product.get("tax_rate")
        track.is_unknown = False
        track.locked = True
        track.confirmed = True
        self.cart_track_ids.add(track.track_id)
        return track

    def active_tracks(self) -> list[Track]:
        return [
            track
            for track in self.tracks.values()
            if track.state != TrackState.EXITED
        ]

    def confirmed_tracks(self) -> list[Track]:
        return [track for track in self.active_tracks() if track.confirmed]

    def visible_outputs(self) -> list[dict]:
        return [track.public_json() for track in self.confirmed_tracks() if track.state == TrackState.VISIBLE]

    def price_outputs(self) -> list[dict]:
        payloads = []
        for track in self.confirmed_tracks():
            mapping = track.price_mapping()
            if mapping is not None:
                payloads.append(mapping)
        return payloads

    def cart_quantities(self) -> dict[str, int]:
        """Unique confirmed track_ids per SKU. One object over 100 frames counts as 1."""
        counts: Counter[str] = Counter()
        for track_id in self.cart_track_ids:
            track = self.tracks.get(track_id)
            if track is None:
                continue
            counts[track.sku] += 1
        return dict(sorted(counts.items()))

    def cart_by_name(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for track_id in self.cart_track_ids:
            track = self.tracks.get(track_id)
            if track is None:
                continue
            counts[track.product_name] += 1
        return dict(sorted(counts.items()))
