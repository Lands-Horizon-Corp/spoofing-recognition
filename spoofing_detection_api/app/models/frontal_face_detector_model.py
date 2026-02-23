from __future__ import annotations

import cv2
import numpy as np
from app.core.config import model_config
from cv2 import data
from PIL import Image


class FrontalFaceDetectorModel:
    def __init__(self):
        self.frontal_classifier = cv2.CascadeClassifier(
            data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_frontal_face(self, image: Image.Image) -> list:
        image_matrix = self._preprocess(image)
        straight_faces = self.frontal_classifier.detectMultiScale(
            image_matrix, 1.1, 4)
        extracted_data = []
        for (x_val, y_val, w_val, h_val) in straight_faces:
            extracted_data.append({
                'x_coordinate': int(x_val),
                'y_coordinate': int(y_val),
                'box_width': int(w_val),
                'box_height': int(h_val),
                'is_frontal': True
            })
        return extracted_data

    def _preprocess(self, image):
        image_matrix = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        print(f"face image size: {image.size}, matrix shape: {
              image_matrix.shape}")
        image_matrix = cv2.cvtColor(image_matrix, cv2.COLOR_BGR2GRAY)
        image_matrix = cv2.resize(
            image_matrix, (model_config.TARGET_SIZE, model_config.TARGET_SIZE))
        return image_matrix


frontal_classifier = FrontalFaceDetectorModel()
