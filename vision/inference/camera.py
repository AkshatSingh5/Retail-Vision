from __future__ import annotations

import sys

import cv2
import numpy as np

from backend.app.config import CAMERA_INDEX, WINDOW_NAME


class CameraError(RuntimeError):
    """Raised when the webcam cannot be opened or read."""


class Camera:
    """OpenCV webcam capture with safe open, display, and shutdown."""

    def __init__(self, index: int | None = None, window_name: str | None = None) -> None:
        self.index = CAMERA_INDEX if index is None else index
        self.window_name = WINDOW_NAME if window_name is None else window_name
        self.capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        backends = []
        if sys.platform.startswith("win"):
            backends.append(cv2.CAP_DSHOW)
        backends.append(cv2.CAP_ANY)

        last_error = "unknown error"
        for backend in backends:
            capture = cv2.VideoCapture(self.index, backend)
            if capture.isOpened():
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                ok, _frame = capture.read()
                if ok:
                    self.capture = capture
                    return
                capture.release()
                last_error = "opened but failed to read a frame"
            else:
                capture.release()
                last_error = "could not open device"

        raise CameraError(
            f"Unable to initialize webcam at index {self.index} ({last_error}). "
            "Check that a camera is connected and CAMERA_INDEX in .env is correct."
        )

    def read(self) -> np.ndarray:
        if self.capture is None:
            raise CameraError("Camera is not open. Call open() first.")
        ok, frame = self.capture.read()
        if not ok or frame is None:
            raise CameraError("Failed to read a frame from the webcam.")
        return frame

    def show(self, frame: np.ndarray) -> None:
        cv2.imshow(self.window_name, frame)

    def wait_key(self, delay_ms: int = 1) -> int:
        return cv2.waitKey(delay_ms) & 0xFF

    def should_quit(self, key: int) -> bool:
        return key in {ord("q"), ord("Q"), 27}

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        cv2.destroyAllWindows()
        cv2.waitKey(1)
