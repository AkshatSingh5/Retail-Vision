"""Canonical settings live in `backend.app.config`.

This module re-exports them so callers can use `backend.app.core.config`.
"""

from backend.app.config import *  # noqa: F403
from backend.app.config import cors_allow_origins

__all__ = ["cors_allow_origins"]
