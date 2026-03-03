from __future__ import annotations

import asyncio
import io
import logging
from typing import cast

import numpy as np
from app.core.config import model_config
from app.models.emotion_detection import emotion_detector
from app.models.face_detector_model import face_detector
from app.models.frontal_face_detector_model import frontal_classifier
from app.models.glass_detector_model import glass_detector
from app.models.spoof_detector_model import spoof_detector
from PIL import Image

logger = logging.getLogger(__name__)


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
    extracted_data = frontal_classifier.detect_proper_face_pipeline(image)
    face_image = image.crop((left, top, right, bottom))
    print(f"Cropped face image size: {face_image.size}")

    if not extracted_data:
        raise ValueError(
            'ERR_NO_FACE')
    if not extracted_data[0]['is_frontal']:
        print(f"Frontal face detection data: {extracted_data}")
        raise ValueError(
            'ERR_FACE_NOT_FRONTAL')

    if not extracted_data[0]['is_mouth_detected']:
        raise ValueError(
            'ERR_MOUTH_NOT_DETECTED')

    have_glass = glass_detector.predict(face_image)
    if have_glass:
        raise ValueError(
            'ERR_GLASSES_DETECTED')

    if not extracted_data[0]['is_eyes_open']:
        raise ValueError(
            'ERR_EYES_CLOSED')

    emotion = emotion_detector.detect(face_image)
    print(f"Emotion analysis result: {emotion}")
    if emotion != 'neutral':
        raise ValueError(
            'ERR_EMOTION_NOT_NEUTRAL')

    face_image = face_image.resize(
        (model_config.TARGET_SIZE, model_config.TARGET_SIZE))
    face_image = np.array(face_image)
    prediction, spoof_confidence = await asyncio.to_thread(
        spoof_detector.predict, face_image)
    logger.info(f"Prediction: {prediction},"
                f"Spoof Confidence: {spoof_confidence}")
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
