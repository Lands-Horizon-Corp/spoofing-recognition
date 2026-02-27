from __future__ import annotations

import cv2
import numpy as np
from app.core.config import model_config
from cv2 import data
from cv2 import face
from PIL import Image


class FrontalFaceDetectorModel:
    def __init__(self):
        self.frontal_classifier = cv2.CascadeClassifier(
            data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.geometry_mapper = face.createFacemarkLBF()
        self.spectacle_tracker = cv2.CascadeClassifier(
            'haarcascade_eye_tree_eyeglasses.xml')

    def detect_frontal_face(self, image: Image.Image) -> list:
        image_matrix = self._preprocess(image)
        straight_faces = self.frontal_classifier.detectMultiScale(
            image_matrix, 1.1, 4)
        extracted_data = []
        for (x_val, y_val, w_val, h_val) in straight_faces:

            _, face_points = self.geometry_mapper.fit(
                image_matrix, np.array([[x_val, y_val, w_val, h_val]]))

            mouth_keypoints = face_points[0][0][48:68]
            contour_area = cv2.contourArea(mouth_keypoints)
            expected_minimum_area = (w_val * h_val) * 0.02

            is_mouth_detected = contour_area > expected_minimum_area

            head_region_gray = image_matrix[y_val:y_val +
                                            h_val, x_val:x_val+w_val]

            is_wearing_glasses = len(
                self.spectacle_tracker.detectMultiScale(head_region_gray)) > 0

            left_eye_points = face_points[0][0][36:42]
            right_eye_points = face_points[0][0][42:48]

            aperture_left = self.calculate_eye_aperture(left_eye_points)
            aperture_right = self.calculate_eye_aperture(right_eye_points)

            average_aperture = (aperture_left + aperture_right) / 2.0

            is_eyes_open = average_aperture > 0.25
            extracted_data.append({
                'x_coordinate': int(x_val),
                'y_coordinate': int(y_val),
                'box_width': int(w_val),
                'box_height': int(h_val),
                'is_frontal': True,
                'is_mouth_detected': is_mouth_detected,
                'is_wearing_glasses': is_wearing_glasses,
                'is_eyes_open': is_eyes_open

            })
        return extracted_data

    def _preprocess(self, image):
        image_matrix = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        print(f"face image size: {image.size}, "
              f"matrix shape: {image_matrix.shape}")
        image_matrix = cv2.cvtColor(image_matrix, cv2.COLOR_BGR2GRAY)
        image_matrix = cv2.resize(
            image_matrix, (model_config.TARGET_SIZE, model_config.TARGET_SIZE))
        return image_matrix

    def calculate_eye_aperture(self, ocular_landmarks):
        height_left = np.linalg.norm(ocular_landmarks[1] - ocular_landmarks[5])
        height_right = np.linalg.norm(
            ocular_landmarks[2] - ocular_landmarks[4])

        width = np.linalg.norm(ocular_landmarks[0] - ocular_landmarks[3])

        aperture_score = (height_left + height_right) / (2.0 * width)

        return aperture_score


frontal_classifier = FrontalFaceDetectorModel()
