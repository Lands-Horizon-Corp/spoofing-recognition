from __future__ import annotations

import logging
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from app.core.config import settings
from cv2 import data
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image

logger = logging.getLogger(__name__)


class ProperFaceDetector:
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

    def is_frontal_face(self, image_matrix) -> bool:
        frontal_faces = self.frontal_classifier.detectMultiScale(
            image_matrix,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        if len(frontal_faces) == 0:
            return False
        return True

    def detect_proper_face_pipeline(self, image: Image.Image) -> list:
        image_matrix, img_bgr = self._preprocess(image)
        landmarks, blendshapes, face_landmarks = self.extract_landmarks(image)

        if landmarks.size == 0:
            return []

        extracted_data = []

        x_val = int(np.min(landmarks[:, 0]))
        y_val = int(np.min(landmarks[:, 1]))
        x_max = int(np.max(landmarks[:, 0]))
        y_max = int(np.max(landmarks[:, 1]))
        w_val = max(0, x_max - x_val)
        h_val = max(0, y_max - y_val)

        is_mouth_detected = self.is_mouth_detected(face_landmarks)
        left_blink_score = blendshapes.get('eyeBlinkLeft', 0.0)
        right_blink_score = blendshapes.get('eyeBlinkRight', 0.0)
        is_eyes_open = (
            left_blink_score < 0.5
            and right_blink_score < 0.5
        )

        extracted_data.append({
            'x_coordinate': x_val,
            'y_coordinate': y_val,
            'box_width': w_val,
            'box_height': h_val,
            'is_frontal': self.is_frontal_face(image_matrix),
            'is_mouth_detected': is_mouth_detected,
            'is_eyes_open': is_eyes_open
        })
        return extracted_data

    def is_mouth_detected(self, face_landmarks) -> bool:
        mouth_indices = [13, 14, 78, 308]
        is_mouth_detected = True
        for idx in mouth_indices:
            lm = face_landmarks[idx]
            if lm.x < 0.0 or lm.x > 1.0 or lm.y < 0.0 or lm.y > 1.0:
                is_mouth_detected = False
                break
        return is_mouth_detected

    def _preprocess(self, image):
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image_matrix = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return image_matrix, img_bgr

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


proper_face_detector = ProperFaceDetector()
