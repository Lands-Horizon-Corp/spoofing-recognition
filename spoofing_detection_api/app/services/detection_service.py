from __future__ import annotations

import asyncio
import io
from typing import cast

import numpy as np
from app.core.config import model_config
from app.models.face_detector_model import face_detector
from app.models.frontal_face_detector_model import frontal_classifier
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
    except Exception as e:
        raise ValueError(
            f"Invalid image file, file type detected:"
            f" {type(upload_file)}") from e
    faces = await asyncio.to_thread(
        face_detector.find_faces, image)
    if not faces:
        raise ValueError('No faces detected in the image.')
    if len(faces) > 1:
        raise ValueError('Multiple faces detected.')
    face = faces[0]
    left, top, right, bottom = face['bbox']
    print(f"bbox: {face['bbox']}, image size: {image.size}")
    extracted_data = frontal_classifier.detect_frontal_face(image)
    face_image = image.crop((left, top, right, bottom))
    print(f"Cropped face image size: {face_image.size}")

    if not extracted_data:
        raise ValueError(
            'face forward properly')

    if not extracted_data[0]['is_mouth_detected']:
        raise ValueError(
            'mouth not detected, please ensure the face is fully visible and properly aligned')
    if extracted_data[0]['is_wearing_glasses']:
        raise ValueError(
            'glasses detected, please remove glasses and try again')
    face_image = face_image.resize(
        (model_config.TARGET_SIZE, model_config.TARGET_SIZE))
    face_image = np.array(face_image)
    prediction, spoof_confidence = await asyncio.to_thread(
        spoof_detector.predict, face_image)

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
