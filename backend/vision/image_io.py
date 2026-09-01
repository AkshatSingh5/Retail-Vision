"""Image decode/encode helpers that do not import OpenCV on Vercel.

Vercel serverless images lack libGL/libgthread. Importing cv2 there crashes the
function (FUNCTION_INVOCATION_FAILED) before any route can run. Pillow + NumPy
cover scan, storage, and color embeddings on that runtime.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from backend.app.config import ON_VERCEL

_cv2_module = None
_cv2_failed = False


def cv2_available() -> bool:
    """False on Vercel. Elsewhere True only if OpenCV actually imports."""
    global _cv2_module, _cv2_failed
    if ON_VERCEL or _cv2_failed:
        return False
    if _cv2_module is not None:
        return True
    try:
        import cv2 as module
    except Exception:
        _cv2_failed = True
        return False
    _cv2_module = module
    return True


def _opencv():
    if not cv2_available():
        raise RuntimeError("OpenCV is not available in this runtime.")
    return _cv2_module


def decode_bgr(image_bytes: bytes) -> np.ndarray | None:
    if not image_bytes:
        return None
    if cv2_available():
        array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = _opencv().imdecode(array, _opencv().IMREAD_COLOR)
        return None if frame is None or getattr(frame, "size", 0) == 0 else frame
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return None
    return np.asarray(image, dtype=np.uint8)[:, :, ::-1].copy()


def encode_jpeg(frame: np.ndarray, quality: int = 90) -> bytes:
    if frame is None or getattr(frame, "size", 0) == 0:
        raise ValueError("Empty image.")
    if cv2_available():
        ok, encoded = _opencv().imencode(".jpg", frame, [int(_opencv().IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            raise ValueError("JPEG encode failed.")
        return encoded.tobytes()
    rgb = np.ascontiguousarray(frame[:, :, ::-1])
    image = Image.fromarray(rgb)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality))
    return buffer.getvalue()


def read_bgr(path: str | Path) -> np.ndarray | None:
    payload = Path(path).read_bytes() if Path(path).is_file() else b""
    return decode_bgr(payload)


def write_jpeg(path: str | Path, frame: np.ndarray, quality: int = 92) -> None:
    Path(path).write_bytes(encode_jpeg(frame, quality=quality))


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if cv2_available():
        return _opencv().cvtColor(image, _opencv().COLOR_BGR2GRAY)
    return np.round(
        0.114 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.299 * image[:, :, 2]
    ).astype(np.uint8)


def resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if cv2_available():
        return _opencv().resize(image, (width, height), interpolation=_opencv().INTER_AREA)
    rgb = image if image.ndim == 2 else image[:, :, ::-1]
    resampled = Image.fromarray(rgb).resize((width, height), Image.Resampling.BILINEAR)
    array = np.asarray(resampled)
    if image.ndim == 2:
        return array
    return array[:, :, ::-1].copy()


def letterbox_bgr(image: np.ndarray, size: int) -> np.ndarray:
    frame = image
    if frame.ndim == 2:
        frame = np.stack([frame, frame, frame], axis=-1)
    elif frame.shape[2] == 4:
        frame = frame[:, :, :3]
    elif frame.shape[2] != 3:
        raise ValueError("Unsupported product image channels.")
    height, width = frame.shape[:2]
    if height < 8 or width < 8:
        raise ValueError("Product crop is too small.")
    scale = size / max(height, width)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = resize(frame, new_w, new_h)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y0 = (size - new_h) // 2
    x0 = (size - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def bgr_to_hsv(image: np.ndarray) -> np.ndarray:
    if cv2_available():
        return _opencv().cvtColor(image, _opencv().COLOR_BGR2HSV)
    b = image[:, :, 0].astype(np.float32) / 255.0
    g = image[:, :, 1].astype(np.float32) / 255.0
    r = image[:, :, 2].astype(np.float32) / 255.0
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc
    h = np.zeros_like(maxc)
    mask = delta > 1e-6
    rc = np.equal(maxc, r) & mask
    gc = np.equal(maxc, g) & mask
    bc = np.equal(maxc, b) & mask
    h[rc] = np.mod((g[rc] - b[rc]) / delta[rc], 6.0)
    h[gc] = (b[gc] - r[gc]) / delta[gc] + 2.0
    h[bc] = (r[bc] - g[bc]) / delta[bc] + 4.0
    hsv = np.empty_like(image)
    hsv[:, :, 0] = np.clip(h * 30.0, 0, 179)
    hsv[:, :, 1] = np.clip(np.where(maxc > 1e-6, delta / np.maximum(maxc, 1e-6) * 255.0, 0), 0, 255)
    hsv[:, :, 2] = np.clip(maxc * 255.0, 0, 255)
    return hsv.astype(np.uint8)


def bgr_to_lab(image: np.ndarray) -> np.ndarray:
    if cv2_available():
        return _opencv().cvtColor(image, _opencv().COLOR_BGR2LAB)
    rgb = image[:, :, ::-1].astype(np.float32) / 255.0
    mask = rgb > 0.04045
    rgb = np.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    x /= 0.95047
    z /= 1.08883
    eps = 0.008856
    kappa = 903.3

    def _f(channel: np.ndarray) -> np.ndarray:
        return np.where(channel > eps, np.cbrt(channel), (kappa * channel + 16.0) / 116.0)

    fx, fy, fz = _f(x), _f(y), _f(z)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_ch = 200.0 * (fy - fz)
    lab = np.empty_like(image)
    lab[:, :, 0] = np.clip(L * 255.0 / 100.0, 0, 255)
    lab[:, :, 1] = np.clip(a + 128.0, 0, 255)
    lab[:, :, 2] = np.clip(b_ch + 128.0, 0, 255)
    return lab.astype(np.uint8)


def calc_hist(image: np.ndarray, channels: list[int], bins: list[int], ranges: list[float]) -> np.ndarray:
    if cv2_available():
        hist = _opencv().calcHist([image], channels, None, bins, ranges)
        return _opencv().normalize(hist, hist).flatten().astype(np.float32)
    samples = image.reshape(-1, image.shape[-1])[:, channels].astype(np.float32)
    pairs = list(zip(ranges[0::2], ranges[1::2]))
    hist, _ = np.histogramdd(samples, bins=bins, range=pairs)
    flat = hist.astype(np.float32).ravel()
    norm = float(np.linalg.norm(flat))
    if norm > 0:
        flat = flat / norm
    return flat


def edge_map(gray: np.ndarray, size: tuple[int, int] = (16, 16)) -> np.ndarray:
    if cv2_available():
        edges = _opencv().Canny(gray, 50, 140).astype(np.float32) / 255.0
        return _opencv().resize(edges, size, interpolation=_opencv().INTER_AREA).flatten()
    gray_f = gray.astype(np.float32)
    gx = np.abs(np.diff(gray_f, axis=1, prepend=gray_f[:, :1]))
    gy = np.abs(np.diff(gray_f, axis=0, prepend=gray_f[:1, :]))
    edges = np.clip((gx + gy) / 255.0, 0.0, 1.0)
    return resize((edges * 255.0).astype(np.uint8), size[0], size[1]).astype(np.float32).flatten() / 255.0


def replicate_border(image: np.ndarray, pad: int) -> np.ndarray:
    if cv2_available():
        return _opencv().copyMakeBorder(image, pad, pad, pad, pad, _opencv().BORDER_REPLICATE)
    return np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode="edge")


def scale_abs(image: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    if cv2_available():
        return _opencv().convertScaleAbs(image, alpha=alpha, beta=beta)
    return np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def laplacian_var(gray: np.ndarray) -> float:
    if cv2_available():
        return float(_opencv().Laplacian(gray, _opencv().CV_64F).var())
    gray_f = gray.astype(np.float32)
    center = gray_f[1:-1, 1:-1]
    lap = (
        gray_f[:-2, 1:-1]
        + gray_f[2:, 1:-1]
        + gray_f[1:-1, :-2]
        + gray_f[1:-1, 2:]
        - 4.0 * center
    )
    return float(lap.var()) if lap.size else 0.0


def draw_labeled_canvas(message: str, width: int = 1280, height: int = 720) -> np.ndarray:
    image = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except Exception:
        font = ImageFont.load_default()
    y = max(40, height // 2 - 60)
    for line in message.split("\n"):
        bbox = draw.textbbox((0, 0), line, font=font)
        text_w = bbox[2] - bbox[0]
        draw.text(((width - text_w) // 2, y), line, fill=(232, 197, 71), font=font)
        y += 44
    return np.asarray(image, dtype=np.uint8)[:, :, ::-1].copy()
