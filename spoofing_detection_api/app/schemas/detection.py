from pydantic import BaseModel


class DetectionResult(BaseModel):
    is_spoof: bool
    confidence: float
    model_version: str = "1.0"
