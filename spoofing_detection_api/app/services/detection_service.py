from __future__ import annotations

import asyncio
import io
from typing import cast

import numpy as np
from app.core.config import model_config
from app.models.spoof_detector_model import spoof_detector
from PIL import Image


async def predict_spoof(upload_file: bytes) -> dict:
    """Orchestrates the prediction pipeline
    Args:
        upload_file (bytes): The uploaded image file to be analyzed for spoofing.
    Returns:
        dict: A dictionary containing the prediction results, including:
            - 'is_spoof' (bool): Indicates whether the image is classified as a
                                spoof based on model threshold and confidence scores
                                or live_confidence is not above 0.90.
            - 'live_confidence' (float): The confidence score for the image being live.
            - 'spoof_confidence' (float): The confidence score for the image being a spoof.
    """

    try:
        image = Image.open(io.BytesIO(upload_file)).convert('RGB')
        image = image.resize(
            (model_config.TARGET_SIZE, model_config.TARGET_SIZE))
        image_np = np.array(image)
    except Exception as e:
        raise ValueError(
            f"Invalid image file, file type detected:"
            f" {type(upload_file)}") from e
    session = spoof_detector.load_model()
    prediction, spoof_confidence = await asyncio.to_thread(
        spoof_detector.predict, image_np, session)

    return {
        'is_spoof': bool(prediction),
        'spoof_confidence': float(spoof_confidence),
    }


if __name__ == '__main__':
    from pathlib import Path

    BASEDIR = Path(__file__).resolve().parent

    class MockUploadFile:
        def __init__(self, file_path):
            self.filename = file_path
            self.file_path = file_path

        async def read(self):
            with open(self.file_path, 'rb') as f:
                return f.read()

    TEST_IMG_PATH = BASEDIR / 'test_img.png'
    img = MockUploadFile(TEST_IMG_PATH)
    try:
        pred = asyncio.run(predict_spoof(cast(bytes, img.read())))
        print(pred)
    except Exception as e:
        print(f"Error during prediction: {e}")
