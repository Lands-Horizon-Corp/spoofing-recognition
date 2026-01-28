from pydantic import BaseModel


class DetectionResult(BaseModel):
    is_spoof: bool
    live_confidence: float
    spoof_confidence: float
    model_version: str = "1.0"
