from __future__ import annotations

from typing import Any

import mediapipe as mp
import numpy as np
import requests  # type: ignore
from app.core.config import settings
from app.core.constants.spoof_errors import DetectionError
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image


async def download_file(file_url: str, file_path: str):
    print(f'Downloading file {file_url} to {file_path}...')
    try:
        response = requests.get(file_url, stream=True)
        response.raise_for_status()

        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f'Model downloaded successfully to {file_path}')
    except requests.exceptions.RequestException as e:
        print(f'Error downloading model: {e}')


def calculate_sigmoid(x):
    return 1 / (1 + np.exp(-x))


class MediaPipeUtils:
    def __init__(self):
        self.face_mesh_detector = None

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

    def extract_landmarks(self, image: Image.Image) -> tuple[np.ndarray, dict[str, float],  list[Any]]:  # noqa: E501
        img_rgb = np.array(image.convert('RGB'))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        self.load_model()
        assert self.face_mesh_detector is not None, 'Face mesh detector model is not loaded'
        detection_result = self.face_mesh_detector.detect(mp_image)
        if not detection_result.face_landmarks:
            raise ValueError(DetectionError.NO_FACE.value)

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

    def close(self):
        if self.face_mesh_detector is not None:
            self.face_mesh_detector.close()
            self.face_mesh_detector = None
