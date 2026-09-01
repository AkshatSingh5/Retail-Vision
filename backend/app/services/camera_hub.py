from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from backend.app.config import (
    AUTO_DETECT_ON_CAMERA,
    DETECT_CONFIDENCE,
    ENABLE_EMBEDDING_REFINEMENT,
    FRAME_SKIP,
    STABLE_FRAMES,
    TRACK_IOU_THRESHOLD,
    TRACK_MAX_MISSING,
)
from backend.app.database import get_session_factory
from backend.app.services.acceptance import MESSAGES
from backend.app.services.cart_service import get_cart

CAMERA_START_FAILED = "Camera could not be started.\nPlease check your webcam connection."
MODEL_LOAD_FAILED = "Detection model could not be loaded."
CAMERA_REQUIRED = "Please turn on the camera first."


class CameraControlError(RuntimeError):
    """Raised for invalid camera/detection control requests."""


class CameraHub:
    """Background camera loop; turning the camera on starts automatic YOLO detection."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._jpeg = _placeholder_jpeg("Camera Offline")
        self.status = "idle"
        self.error: str | None = None
        self.fps = 0.0
        self.latency_ms = 0.0
        self.tracks: list[dict] = []
        self.pipeline = None
        self.alerts: list[dict] = []
        self.camera_active = False
        self.detection_active = False
        self.detection_loading = False
        self._frame_log = 0
        self._crops: dict[int, bytes] = {}
        self._bindings: dict[int, dict] = {}

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        self.start_camera()

    def start_camera(self) -> dict[str, Any]:
        if self.running:
            return self.info()
        self._stop.clear()
        self.camera_active = True
        self.detection_active = False
        # Preview-only by default; continuous YOLO cart updates are opt-in.
        self.detection_loading = bool(AUTO_DETECT_ON_CAMERA)
        self.error = None
        self.alerts = []
        self.tracks = []
        self.status = "starting"
        self._set_jpeg(_placeholder_frame("Opening camera..."))
        self._thread = threading.Thread(target=self._run, name="pos-camera", daemon=True)
        self._thread.start()
        if AUTO_DETECT_ON_CAMERA:
            print("Camera ON — automatic YOLO26m detection will start after the camera opens")
        else:
            print("Camera ON — preview only; use Scan Product for recognition")
        return self.info()

    def stop(self) -> None:
        self.stop_camera()

    def stop_camera(self) -> dict[str, Any]:
        self.detection_active = False
        self.detection_loading = False
        self.camera_active = False
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=6)
        self._thread = None
        if self.pipeline is not None:
            self.pipeline.manager.reset()
        self._crops.clear()
        self._bindings.clear()
        self.tracks = []
        self.fps = 0.0
        self.latency_ms = 0.0
        self.status = "idle"
        self.error = None
        self._set_jpeg(_placeholder_frame("Camera Offline"))
        return self.info()

    def start_detection(self) -> dict[str, Any]:
        if not self.camera_active or not self.running:
            raise CameraControlError(CAMERA_REQUIRED)
        if self.detection_active:
            return self.info()
        if self.pipeline is not None:
            self.pipeline.manager.reset()
            self.detection_active = True
            self.detection_loading = False
            self.error = None
            print("YOLO26m detection ON (model already loaded)")
            return self.info()
        self.detection_loading = True
        self.error = None
        print("YOLO26m load requested")
        return self.info()

    def stop_detection(self) -> dict[str, Any]:
        self.detection_active = False
        self.detection_loading = False
        if self.pipeline is not None:
            self.pipeline.manager.reset()
        self._crops.clear()
        self._bindings.clear()
        self.tracks = []
        self.latency_ms = 0.0
        if self.camera_active:
            self.status = "live"
        return self.info()

    def reset_tracking(self) -> None:
        if self.pipeline is not None:
            self.pipeline.manager.reset()
        self._crops.clear()
        self._bindings.clear()

    def jpeg(self) -> bytes:
        with self._lock:
            return self._jpeg

    def capture_frame(self) -> bytes:
        """Return the latest camera JPEG for one-shot Scan Product."""
        if not self.camera_active or not self.running:
            raise CameraControlError(CAMERA_REQUIRED)
        payload = self.jpeg()
        if not payload:
            raise CameraControlError("Camera initialization failed.")
        return payload

    def info(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "running": self.running,
            "camera_active": bool(self.camera_active and self.running),
            "detection_active": bool(self.detection_active),
            "detection_loading": bool(self.detection_loading),
            "error": self.error,
            "fps": round(self.fps, 1),
            "latency_ms": round(self.latency_ms, 1),
            "tracks": self.tracks if self.detection_active else [],
            "unknowns": self.unknowns() if self.detection_active else [],
            "alerts": list(self.alerts[-8:]),
        }

    def unknowns(self) -> list[dict]:
        if self.pipeline is None:
            return []
        items = []
        for track in self.pipeline.manager.active_tracks():
            if not track.is_unknown or track.locked:
                continue
            if track.hits < 2:
                continue
            items.append(
                {
                    "track_id": track.track_id,
                    "confidence": round(float(track.confidence), 4),
                    "bbox": [round(float(v), 2) for v in track.bbox],
                    "product_name": "New Product Detected",
                    "crop_url": f"/pos/crops/{track.track_id}.jpg",
                }
            )
        return items

    def crop_bytes(self, track_id: int) -> bytes | None:
        return self._crops.get(int(track_id))

    def bind_registered_track(self, track_id: int, product) -> None:
        from backend.app.schemas.product import serialize_money

        payload = {
            "id": int(product.id),
            "sku": str(product.sku),
            "name": str(product.name),
            "price": serialize_money(product.price),
            "tax_rate": serialize_money(product.tax_rate),
        }
        self._bindings[int(track_id)] = payload
        if self.pipeline is not None:
            self.pipeline.manager.bind_product(int(track_id), payload)

    def _capture_crops(self, frame) -> None:
        from vision.detection.crop import crop_box

        if self.pipeline is None:
            return
        for track in self.pipeline.manager.active_tracks():
            crop = crop_box(frame, track.bbox)
            if crop is None:
                continue
            try:
                from vision.image_io import encode_jpeg

                self._crops[int(track.track_id)] = encode_jpeg(crop, quality=90)
            except Exception:
                continue

    def _set_jpeg(self, frame: np.ndarray) -> None:
        try:
            from vision.image_io import encode_jpeg

            payload = encode_jpeg(frame, quality=80)
        except Exception:
            return
        with self._lock:
            self._jpeg = payload

    def _ensure_pipeline(self) -> None:
        if self.pipeline is not None:
            return
        from vision.detection.yolo_detector import YOLODetector
        from vision.tracking.pipeline import TrackingPipeline

        # Live POS uses detect() + IoU tracking. Ultralytics ByteTrack's
        # new_track_thresh (~0.60) was dropping every retail_yolo26m_v2 box.
        self.pipeline = TrackingPipeline(
            detector=YOLODetector(confidence_threshold=DETECT_CONFIDENCE),
            use_ultralytics_track=False,
            stable_frames=STABLE_FRAMES,
            max_missing=TRACK_MAX_MISSING,
            track_iou_threshold=TRACK_IOU_THRESHOLD,
            enable_embedding=ENABLE_EMBEDDING_REFINEMENT,
        )

    def _run(self) -> None:
        from vision.inference.camera import Camera, CameraError

        camera: Camera | None = None
        skip_counter = 0
        try:
            self.status = "opening_camera"
            camera = Camera(window_name="Retail Vision POS")
            try:
                camera.open()
            except CameraError:
                self.camera_active = False
                self.detection_active = False
                self.status = "camera_error"
                self.error = CAMERA_START_FAILED.replace("\n", " ")
                self.alerts = [{"reason": "camera_error", "message": CAMERA_START_FAILED}]
                self._set_jpeg(_placeholder_frame(CAMERA_START_FAILED))
                return

            self.status = "live"
            self.camera_active = True
            if AUTO_DETECT_ON_CAMERA and not self.detection_active and self.pipeline is None and not self.detection_loading:
                self.detection_loading = True
            previous = time.perf_counter()
            while not self._stop.is_set():
                try:
                    frame = camera.read()
                except CameraError:
                    self.status = "camera_error"
                    self.error = "Camera disconnected. Please check your webcam connection."
                    self.alerts = [
                        {
                            "reason": "camera_disconnected",
                            "message": MESSAGES["camera_disconnected"],
                        }
                    ]
                    self._set_jpeg(_placeholder_frame("Camera disconnected\nPlease check your webcam connection."))
                    time.sleep(1.0)
                    try:
                        camera.open()
                        self.status = "live"
                        self.error = None
                    except CameraError:
                        continue
                    continue

                if self.detection_loading and self.pipeline is None:
                    self._set_jpeg(_placeholder_frame("Loading YOLO26m...\nPlease wait"))
                    try:
                        print("Loading YOLO26m on camera thread...")
                        self._ensure_pipeline()
                        if self.detection_loading and self.camera_active:
                            self.pipeline.manager.reset()
                            self.detection_active = True
                            self.error = None
                            print(f"YOLO26m ready (conf={DETECT_CONFIDENCE})")
                        else:
                            self.detection_active = False
                    except FileNotFoundError:
                        self.detection_active = False
                        self.error = MODEL_LOAD_FAILED
                        self.alerts = [{"reason": "model_unavailable", "message": MODEL_LOAD_FAILED}]
                        print(MODEL_LOAD_FAILED)
                    except Exception as extra:
                        self.detection_active = False
                        self.error = MODEL_LOAD_FAILED
                        self.alerts = [
                            {"reason": "model_unavailable", "message": f"{MODEL_LOAD_FAILED}\n{extra}"}
                        ]
                        print(f"{MODEL_LOAD_FAILED} {extra}")
                    finally:
                        self.detection_loading = False

                now = time.perf_counter()
                elapsed = now - previous
                self.fps = 1.0 / elapsed if elapsed > 0 else 0.0
                previous = now

                if self.detection_active and self.pipeline is not None:
                    skip_counter += 1
                    if FRAME_SKIP > 0 and skip_counter % (FRAME_SKIP + 1) != 1:
                        self._set_jpeg(frame)
                        continue
                    try:
                        result = self.pipeline.process(frame)
                        for track_id, payload in self._bindings.items():
                            self.pipeline.manager.bind_product(track_id, payload)
                        annotated = self.pipeline.annotate(frame, result, fps=self.fps)
                        self.latency_ms = result["latency_ms"]
                        self.tracks = [
                            track.to_output() for track in self.pipeline.manager.active_tracks()
                        ]
                        self._capture_crops(frame)
                        self._frame_log += 1
                        if self._frame_log == 1 or self._frame_log % 20 == 0:
                            print(
                                f"YOLO26m frame {self._frame_log}: {result['detections']} boxes, "
                                f"{len(result['tracks'])} tracks, {self.latency_ms:.0f} ms"
                            )
                        self._apply_cart()
                        self._set_jpeg(annotated)
                    except Exception as extra:
                        self.detection_active = False
                        if self.pipeline is not None:
                            self.pipeline.manager.reset()
                        self.tracks = []
                        self.alerts = [
                            {
                                "reason": "detection_error",
                                "message": f"Detection stopped.\n{extra}",
                            }
                        ]
                        self.error = f"Detection stopped. {extra}"
                        print(f"YOLO26m error: {extra}")
                        self._set_jpeg(_placeholder_frame(f"Detection stopped\n{extra}"))
                else:
                    self.latency_ms = 0.0
                    self.tracks = []
                    self._set_jpeg(frame)
        except Exception as extra:
            self.camera_active = False
            self.detection_active = False
            self.status = "error"
            self.error = str(extra)
            self._set_jpeg(_placeholder_frame("Vision error\nUse manual add"))
        finally:
            if camera is not None:
                try:
                    camera.release()
                except Exception:
                    pass
            self.camera_active = False
            self.detection_active = False
            self.detection_loading = False
            if self.status not in {"camera_error", "error", "model_unavailable"}:
                self.status = "idle"
                self._set_jpeg(_placeholder_frame("Camera Offline"))

    def _apply_cart(self) -> None:
        if self.pipeline is None or not self.detection_active:
            return
        session = None
        try:
            session = get_session_factory()()
            tracks = [
                track.to_output()
                for track in self.pipeline.manager.tracks.values()
                if track.track_id in self.pipeline.manager.cart_track_ids
            ]
            snapshot = get_cart().apply_tracks(tracks, session=session)
            session.commit()
            self.alerts = list(snapshot.get("alerts") or [])
        except Exception as extra:
            if session is not None:
                session.rollback()
            self.alerts = [
                {
                    "reason": "database_unavailable",
                    "message": MESSAGES["database_unavailable"],
                    "detail": str(extra),
                }
            ]
        finally:
            if session is not None:
                session.close()


def _placeholder_frame(message: str) -> np.ndarray:
    from vision.image_io import draw_labeled_canvas

    return draw_labeled_canvas(message)


def _placeholder_jpeg(message: str) -> bytes:
    try:
        from vision.image_io import encode_jpeg

        return encode_jpeg(_placeholder_frame(message), quality=80)
    except Exception:
        return b""


_hub: CameraHub | None = None


def get_camera_hub() -> CameraHub:
    global _hub
    if _hub is None:
        _hub = CameraHub()
    return _hub
