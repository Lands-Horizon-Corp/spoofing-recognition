from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np
from app.core.config import settings
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image


class FrontalFaceDetectorModel:
    def __init__(self):
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

    def detect_frontal_face(self, image: Image.Image) -> list:
        image_matrix, img_bgr = self._preprocess(image)
        landmarks, blendshapes = self.extract_landmarks(image)

        if landmarks.size == 0:
            return []

        extracted_data = []

        x_val = int(np.min(landmarks[:, 0]))
        y_val = int(np.min(landmarks[:, 1]))
        x_max = int(np.max(landmarks[:, 0]))
        y_max = int(np.max(landmarks[:, 1]))
        w_val = max(0, x_max - x_val)
        h_val = max(0, y_max - y_val)

        right_eye_points, left_eye_points = self.get_eye_points(landmarks)

        jaw_open_score = blendshapes.get('jawOpen', 0.0)
        is_mouth_detected = jaw_open_score > 0.15

        is_wearing_glasses = self.check_for_glasses(image_matrix, landmarks)

        # left_visible, right_visible = self.is_eyes_visible(
        #     img_bgr, left_eye_points, right_eye_points)
        left_blink_score = blendshapes.get('eyeBlinkLeft', 0.0)
        right_blink_score = blendshapes.get('eyeBlinkRight', 0.0)
        is_eyes_open = (
            left_blink_score < 0.5
            and right_blink_score < 0.5
            # and left_visible
            # and right_visible
        )

        extracted_data.append({
            'x_coordinate': x_val,
            'y_coordinate': y_val,
            'box_width': w_val,
            'box_height': h_val,
            'is_frontal': True,
            'is_mouth_detected': is_mouth_detected,
            'is_wearing_glasses': is_wearing_glasses,
            'is_eyes_open': is_eyes_open
        })
        return extracted_data

    def get_eye_points(self, landmarks):
        right_eye_idx = [33, 160, 158, 133, 153, 144]
        left_eye_idx = [362, 385, 387, 263, 373, 380]

        right_eye = np.array([landmarks[i] for i in right_eye_idx])
        left_eye = np.array([landmarks[i] for i in left_eye_idx])

        return right_eye, left_eye

    def _preprocess(self, image):
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image_matrix = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return image_matrix, img_bgr

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
        if laplacian_var < 10:  # tune this threshold
            return False

        # --- Check 2: Sclera/iris color check (white or colored region expected) ---
        hsv_region = cv2.cvtColor(eye_region, cv2.COLOR_BGR2HSV)

        # Check for white-ish sclera presence
        white_mask = cv2.inRange(hsv_region, np.array(
            [0, 0, 180]), np.array([180, 40, 255]))
        white_ratio = np.sum(white_mask > 0) / white_mask.size

        if white_ratio < 0.02:  # at least 5% should be sclera-like
            return False

        return True

    def is_eyes_visible(self, img_bgr, left_eye_points, right_eye_points):
        left_visible = self.check_eye_region_visibility(
            img_bgr, left_eye_points)
        right_visible = self.check_eye_region_visibility(
            img_bgr, right_eye_points)

        return left_visible, right_visible

    def check_for_glasses(self, image_matrix, face_points):
        roi_indices = [127, 356, 105, 334, 116, 345]

        pts = face_points[roi_indices]

        eye_top = max(0, int(np.min(pts[:, 1])))
        eye_bottom = min(image_matrix.shape[0], int(np.max(pts[:, 1])))
        eye_left = max(0, int(np.min(pts[:, 0])))
        eye_right = min(image_matrix.shape[1], int(np.max(pts[:, 0])))

        roi = image_matrix[eye_top:eye_bottom, eye_left:eye_right]

        if roi.size == 0:
            return False

        edges = cv2.Canny(roi, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size

        return edge_density > 0.1

    def extract_landmarks(self, image: Image.Image) -> tuple[np.ndarray, dict[str, float]]:
        img_rgb = np.array(image.convert('RGB'))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        detection_result = self.face_mesh_detector.detect(mp_image)
        if not detection_result.face_landmarks:
            return np.array([]), {}

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

        return np.array(pixel_points), blendshapes


frontal_classifier = FrontalFaceDetectorModel()
