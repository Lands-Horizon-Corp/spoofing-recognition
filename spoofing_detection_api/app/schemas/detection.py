from pydantic import BaseModel


class DetectionResult(BaseModel):
    is_spoof: bool
    live_confidence: float
    spoof_confidence: float
