from __future__ import annotations

from pydantic import BaseModel


class DetectionResult(BaseModel):
    is_spoof: bool
    spoof_confidence: float
