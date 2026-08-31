"""Shared inference device selection for YOLO26m and DINOv2."""

from __future__ import annotations

import torch


def cuda_available() -> bool:
    return bool(torch.cuda.is_available())


def gpu_name() -> str | None:
    if not cuda_available():
        return None
    return torch.cuda.get_device_name(0)


def resolve_device(preference: str | None = None, *, warn: bool = True) -> str:
    """Honor YOLO_DEVICE / DINO_DEVICE. Prefer CUDA; never hide a missing GPU."""
    pref = (preference or "").strip().lower()
    if pref in {"cpu"}:
        if warn and cuda_available():
            print("WARNING: Device preference is CPU while CUDA is available.")
        return "cpu"
    if pref in {"cuda", "gpu", ""}:
        if cuda_available():
            return "cuda"
        if warn:
            print("WARNING: CUDA unavailable. Running on CPU.")
        return "cpu"
    if pref.startswith("cuda:"):
        if cuda_available():
            return pref
        if warn:
            print("WARNING: CUDA unavailable. Running on CPU.")
        return "cpu"
    if cuda_available():
        return "cuda"
    if warn:
        print("WARNING: CUDA unavailable. Running on CPU.")
    return "cpu"


def device_banner_name(device: str) -> str:
    return "CUDA" if str(device).startswith("cuda") else "CPU"
