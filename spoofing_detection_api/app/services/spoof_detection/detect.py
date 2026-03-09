from __future__ import annotations

import io

import numpy as np
from app.core.config import model_config
from app.schemas.detection import DetectionResult
from app.services.spoof_detection.checkers.covered import covered_checker
from app.services.spoof_detection.checkers.emotion import emotion_checker
from app.services.spoof_detection.checkers.face_direction import \
    face_direction_checker
from app.services.spoof_detection.checkers.glass import glass_checker
from app.services.spoof_detection.checkers.single_face import face_detector
from app.services.spoof_detection.spoof_model import spoof_detector
from PIL import Image


def detect_spoof_service(uploaded_file: bytes) -> DetectionResult:
    image = open_image(uploaded_file)
    faces = face_detector.find_faces(image)

    if not faces:
        raise ValueError(
            'ERR_NO_FACE')
    if len(faces) > 1:
        raise ValueError(
            'ERR_MULTIPLE_FACES')

    face = faces[0]
    left, top, right, bottom = face['bbox']
    face_image = image.crop((left, top, right, bottom))

    is_facing_forward = face_direction_checker.is_facing_forward(face_image)
    if not is_facing_forward:
        raise ValueError(
            'ERR_FACE_NOT_FRONTAL')

    have_glasses = glass_checker.detect_glasses(image)
    if have_glasses:
        raise ValueError(
            'ERR_GLASSES_DETECTED')

    _,  blendshapes, face_landmarks,  = covered_checker.extract_landmarks(
        face_image)
    is_eye_covered = covered_checker.is_eyes_covered(blendshapes)
    if is_eye_covered:
        raise ValueError(
            'ERR_EYES_CLOSED')

    is_mouth_covered = covered_checker.is_mouth_covered(face_landmarks)
    if is_mouth_covered:
        raise ValueError(
            'ERR_MOUTH_NOT_DETECTED')

    emotion = emotion_checker.detect(face_image)
    if emotion is None:
        raise ValueError(
            'ERR_EMOTION_DETECTION_FAILED')

    face_image = face_image.resize(
        (model_config.TARGET_SIZE, model_config.TARGET_SIZE))
    face_image = np.array(face_image)
    prediction, spoof_confidence = spoof_detector.predict(face_image)

    print(
        f"Emotion: {emotion}\n",
        f"Face direction frontal: {is_facing_forward}\n",
        f"Glasses detected: {have_glasses}\n",
        f"Eyes covered: {is_eye_covered}\n",
        f"Mouth covered: {is_mouth_covered}\n",
        f"Spoof prediction: {prediction}\n",
        f"Spoof confidence: {spoof_confidence}"
    )

    return DetectionResult(
        is_spoof=prediction,
        spoof_confidence=spoof_confidence
    )


def open_image(upload_file: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(upload_file)).convert('RGB')
        # Save the image for debugging purposes
        image.save('./spoofing_detection_api/debug_image.jpg')
    except Exception as e:
        raise ValueError(
            f"Invalid image file, file type detected:"
            f" {type(upload_file)}") from e
    return image
