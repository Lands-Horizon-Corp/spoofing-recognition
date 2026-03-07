from __future__ import annotations

import logging

import cv2
import numpy as np
from cv2 import data
from PIL import Image

logger = logging.getLogger(__name__)

# might change to mediapipe later


class FaceDirectionChecker:
    def __init__(self):

        self.frontal_classifier = cv2.CascadeClassifier(
            data.haarcascades + 'haarcascade_frontalface_default.xml')

    def is_facing_forward(self, image: Image.Image) -> bool:
        """
        Check if the face is facing forward.

        :param image: The input image.
        :return: True if the face is facing forward, False otherwise.
        """
        image_matrix = self._preprocess(image)
        frontal_faces = self.frontal_classifier.detectMultiScale(
            image_matrix,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        if len(frontal_faces) == 0:
            return False
        return True

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image_matrix = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return image_matrix


face_direction_checker = FaceDirectionChecker()
