from __future__ import annotations

from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from app.core.config import settings
from cv2 import data
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image


class CoveredChecker:
    def __init__(self):

        self.face_mesh_detector = None

        self.frontal_classifier = cv2.CascadeClassifier(
            data.haarcascades + 'haarcascade_frontalface_default.xml')

    def load_model(self):
        if self.face_mesh_detector is None:
            base_options = python.BaseOptions(
                model_asset_path=settings.FACE_LANDMARKS_MODEL_PATH
            )
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                output_face_blendshapes=True
            )
            self.face_mesh_detector = vision.FaceLandmarker.create_from_options(
                options)

    def is_eyes_covered(self, blendshapes: dict[str, float]) -> bool:
        left_blink_score = blendshapes.get('eyeBlinkLeft', 0.0)
        right_blink_score = blendshapes.get('eyeBlinkRight', 0.0)
        is_eyes_covered = (
            left_blink_score > 0.5
            and right_blink_score > 0.5
        )
        return is_eyes_covered

    def is_mouth_covered(self, face_landmarks) -> bool:
        mouth_indices = [13, 14, 78, 308]
        is_mouth_covered = False
        for idx in mouth_indices:
            lm = face_landmarks[idx]
            if lm.x < 0.0 or lm.x > 1.0 or lm.y < 0.0 or lm.y > 1.0:
                is_mouth_covered = True
                break
        return is_mouth_covered

    def detect_covered(self, image: Image.Image) -> bool:

        _, blendshapes, face_landmarks = self.extract_landmarks(image)
        if not blendshapes or not face_landmarks:
            return False

        eyes_covered = self.is_eyes_covered(blendshapes)
        mouth_covered = self.is_mouth_covered(face_landmarks)

        return eyes_covered or mouth_covered

    def extract_landmarks(self, image: Image.Image) -> tuple[np.ndarray, dict[str, float],  list[Any]]:  # noqa: E501
        img_rgb = np.array(image.convert('RGB'))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        self.load_model()
        assert self.face_mesh_detector is not None, 'Face mesh detector model is not loaded'
        detection_result = self.face_mesh_detector.detect(mp_image)
        if not detection_result.face_landmarks:
            return np.array([]), {}, detection_result

        h, w, _ = img_rgb.shape
        face_landmarks = detection_result.face_landmarks[0]

        pixel_points = []
        for landmark in face_landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            pixel_points.append([x, y])

        blendshapes: dict[str, float] = {}
        if detection_result.face_blendshapes:
            for category in detection_result.face_blendshapes[0]:
                blendshapes[category.category_name] = category.score

        return np.array(pixel_points), blendshapes, face_landmarks


covered_checker = CoveredChecker()
