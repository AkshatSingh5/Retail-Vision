from __future__ import annotations

from pydantic import BaseModel, Field


class CaptionOut(BaseModel):
    prompt: str = Field(min_length=1)
    task: str = "<MORE_DETAILED_CAPTION>"
    model: str
    device: str
