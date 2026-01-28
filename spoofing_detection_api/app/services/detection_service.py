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
    prediction, live_confidence, spoof_confidence = detector.predict(image_np)

    return {
        "is_spoof": bool(prediction),
        "live_confidence": float(live_confidence),
        "spoof_confidence": float(spoof_confidence),
        "model_version": detector.version,
    }


if __name__ == "__main__":
    import asyncio
    from pathlib import Path

    BASEDIR = Path(__file__).resolve().parent

    class MockUploadFile:
        def __init__(self, file_path):
            self.filename = file_path
            self.file_path = file_path

        async def read(self):
            with open(self.file_path, "rb") as f:
                return f.read()

    TEST_IMG_PATH = BASEDIR / "test_img.png"
    img = MockUploadFile(TEST_IMG_PATH)
    try:
        pred = asyncio.run(predict_spoof(img))
        print(pred)
    except Exception as e:
        print(f"Error during prediction: {e}")
