from __future__ import annotations

import math
from typing import Any
from typing import TypeAlias

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


TBlendshapes: TypeAlias = dict[str, float]

TFaceLandmarks: TypeAlias = list[Any]

TPose: TypeAlias = dict[str, float]


class MediaPipeUtils:
    """
    should pass the whole image not the cropped face image.
    """

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
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
            self.face_mesh_detector = vision.FaceLandmarker.create_from_options(
                options)

    def get_head_pose(self, transformation_matrix: np.ndarray) -> tuple[float, float, float]:
        rmat = transformation_matrix[:3, :3]
        sy = math.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
        singular = sy < 1e-6

        if not singular:
            pitch = math.atan2(rmat[2, 1], rmat[2, 2])
            yaw = math.atan2(-rmat[2, 0], sy)
            roll = math.atan2(rmat[1, 0], rmat[0, 0])
        else:
            pitch = math.atan2(-rmat[1, 2], rmat[1, 1])
            yaw = math.atan2(-rmat[2, 0], sy)
            roll = 0

        return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)

    def extract_landmarks(self, image: Image.Image) -> tuple[np.ndarray, TBlendshapes, TFaceLandmarks, TPose]:  # noqa: E501
        img_rgb = np.array(image.convert('RGB'))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        self.load_model()
        assert self.face_mesh_detector is not None, 'Face mesh detector model is not loaded'

        detection_result = self.face_mesh_detector.detect(mp_image)
        if not detection_result.face_landmarks:
            raise ValueError(DetectionError.NO_FACE.value)

        h, w, _ = img_rgb.shape
        face_landmarks = detection_result.face_landmarks[0]
        pixel_points = [[int(landmark.x * w), int(landmark.y * h)]
                        for landmark in face_landmarks]

        blendshapes: TBlendshapes = {}
        if detection_result.face_blendshapes:
            for category in detection_result.face_blendshapes[0]:
                blendshapes[category.category_name] = category.score
        pose: TPose = {}
        if detection_result.facial_transformation_matrixes:
            matrix = detection_result.facial_transformation_matrixes[0]
            pitch, yaw, roll = self.get_head_pose(matrix)
            pose = {'pitch': pitch, 'yaw': yaw, 'roll': roll}

        return np.array(pixel_points), blendshapes, face_landmarks, pose

    def close(self):
        if self.face_mesh_detector is not None:
            self.face_mesh_detector.close()
            self.face_mesh_detector = None


mp_utils = MediaPipeUtils()
