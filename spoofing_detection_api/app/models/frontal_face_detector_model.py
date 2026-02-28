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
        image_matrix, img_brg = self._preprocess(image)
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
            left_visible, right_visible = self.is_eyes_visible(
                img_brg, left_eye_points, right_eye_points)
            average_aperture = (aperture_left + aperture_right) / 2.0

            is_eyes_open = average_aperture > 0.25 and left_visible and right_visible
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
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        print(f"face image size: {image.size}, "
              f"matrix shape: {img_bgr.shape}")
        image_matrix = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        image_matrix = cv2.resize(
            image_matrix, (model_config.TARGET_SIZE, model_config.TARGET_SIZE))
        return image_matrix, img_bgr

    def calculate_eye_aperture(self, ocular_landmarks):
        height_left = np.linalg.norm(ocular_landmarks[1] - ocular_landmarks[5])
        height_right = np.linalg.norm(
            ocular_landmarks[2] - ocular_landmarks[4])

        width = np.linalg.norm(ocular_landmarks[0] - ocular_landmarks[3])

        aperture_score = (height_left + height_right) / (2.0 * width)

        return aperture_score

    def check_eye_region_visibility(self, frame, eye_points):
        """
        Checks if the eye region has enough edge detail and skin-like texture.
        A covered eye region will lack edge structure (hair/cloth is uniform).
        """
        # Get bounding box around eye points with padding
        x_coords = eye_points[:, 0]
        y_coords = eye_points[:, 1]

        x_min = max(int(np.min(x_coords)) - 10, 0)
        x_max = min(int(np.max(x_coords)) + 10, frame.shape[1])
        y_min = max(int(np.min(y_coords)) - 15, 0)
        y_max = min(int(np.max(y_coords)) + 15, frame.shape[0])

        eye_region = frame[y_min:y_max, x_min:x_max]

        if eye_region.size == 0:
            return False

        gray_region = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)

        # --- Check 1: Laplacian variance (edge sharpness) ---
        laplacian_var = cv2.Laplacian(gray_region, cv2.CV_64F).var()
        # A covered region (by cloth, hair mass, hand) has low edge variance
        if laplacian_var < 20:  # tune this threshold
            return False

        # --- Check 2: Sclera/iris color check (white or colored region expected) ---
        hsv_region = cv2.cvtColor(eye_region, cv2.COLOR_BGR2HSV)

        # Check for white-ish sclera presence
        white_mask = cv2.inRange(hsv_region, np.array(
            [0, 0, 180]), np.array([180, 40, 255]))
        white_ratio = np.sum(white_mask > 0) / white_mask.size

        if white_ratio < 0.05:  # at least 5% should be sclera-like
            return False

        return True

    def is_eyes_visible(self, img_brg, left_eye_points, right_eye_points):
        orig_h, orig_w = img_brg.shape[:2]

        scale_x = orig_w / model_config.TARGET_SIZE
        scale_y = orig_h / model_config.TARGET_SIZE

        # Scale eye points
        left_eye_points_orig = left_eye_points * [scale_x, scale_y]
        right_eye_points_orig = right_eye_points * [scale_x, scale_y]

        # Now pass the scaled points to visibility check
        left_visible = self.check_eye_region_visibility(
            img_brg, left_eye_points_orig)
        right_visible = self.check_eye_region_visibility(
            img_brg, right_eye_points_orig)

        return left_visible, right_visible


frontal_classifier = FrontalFaceDetectorModel()
