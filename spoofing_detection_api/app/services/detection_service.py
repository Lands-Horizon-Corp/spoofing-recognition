from PIL import Image
import io
from fastapi import UploadFile
import numpy as np
from app.models.prediction_model import SpoofDetector


detector = SpoofDetector()


async def predict_spoof(upload_file: UploadFile) -> dict:
    """Orchestrates the prediction pipeline"""

    contents = await upload_file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)
    except Exception as e:
        raise ValueError(
            f"Invalid image file, file type detected: {upload_file.content_type}"
        ) from e
    prediction, confidence = detector.predict(image_np)

    return {
        "is_spoof": bool(prediction),
        "confidence": float(confidence),
        "model_version": detector.version,
    }
