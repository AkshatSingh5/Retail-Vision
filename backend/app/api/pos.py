from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse

from backend.app.config import FRONTEND_DIR, ROOT_DIR
from backend.app.services.camera_hub import CameraControlError, get_camera_hub

router = APIRouter(tags=["pos"])
_HTML_HEADERS = {"Cache-Control": "no-store"}


def _pos_index():
    frontend_index = FRONTEND_DIR / "index.html"
    if frontend_index.is_file():
        return frontend_index
    return ROOT_DIR / "backend" / "app" / "static" / "pos" / "index.html"


def _mjpeg():
    hub = get_camera_hub()
    while True:
        frame = hub.jpeg()
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.05)


@router.get("/pos")
def pos_page() -> FileResponse:
    return FileResponse(_pos_index(), media_type="text/html", headers=_HTML_HEADERS)


@router.get("/pos/stream")
def pos_stream() -> StreamingResponse:
    return StreamingResponse(_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")


@router.get("/pos/status")
def pos_status() -> dict:
    return get_camera_hub().info()


@router.post("/pos/camera/start")
def pos_camera_start() -> dict:
    return get_camera_hub().start_camera()


@router.post("/pos/camera/stop")
def pos_camera_stop() -> dict:
    return get_camera_hub().stop_camera()


@router.post("/pos/detection/start")
def pos_detection_start() -> dict:
    try:
        return get_camera_hub().start_detection()
    except CameraControlError as extra:
        raise HTTPException(status_code=400, detail=str(extra)) from extra


@router.post("/pos/detection/stop")
def pos_detection_stop() -> dict:
    return get_camera_hub().stop_detection()


@router.post("/pos/camera/capture")
def pos_camera_capture() -> Response:
    try:
        payload = get_camera_hub().capture_frame()
    except CameraControlError as extra:
        raise HTTPException(status_code=400, detail=str(extra)) from extra
    return Response(content=payload, media_type="image/jpeg")


@router.get("/pos/crops/{track_id}.jpg")
def pos_crop(track_id: int) -> Response:
    payload = get_camera_hub().crop_bytes(int(track_id))
    if not payload:
        raise HTTPException(status_code=404, detail="No crop is available for that detection.")
    return Response(content=payload, media_type="image/jpeg")
