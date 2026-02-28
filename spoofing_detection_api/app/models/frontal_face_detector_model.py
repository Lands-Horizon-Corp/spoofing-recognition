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
        self.frontal_classifier = cv2.CascadeClassifier(
            data.haarcascades + 'haarcascade_frontalface_default.xml')

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

    # def detect_frontal_face(self, image: Image.Image) -> list:
    #     image_matrix, _ = self._preprocess(image)
    #     frontal_faces = self.frontal_classifier.detectMultiScale(
    #         image_matrix,
    #         scaleFactor=1.1,
    #         minNeighbors=5,
    #         minSize=(30, 30)
    #     )
    #     if len(frontal_faces) == 0:
    #         return []
    #     return self.detect_proper_face_pipeline(image)

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
            'is_frontal': self.is_frontal_face(image_matrix),
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
        """
        Detect glasses using multiple methods.
        Robust to different face angles.
        """
        h, w = image_matrix.shape[:2]

        left_eye_indices = [33, 133, 157, 158, 159, 160,
                            161, 246, 7, 163, 144, 145, 153, 154, 155]
        right_eye_indices = [362, 263, 384, 385, 386, 387,
                             388, 466, 249, 390, 373, 374, 380, 381, 382]
        nose_bridge_indices = [168, 6, 197, 195, 5]

        # Wider ROI covering eyebrow-to-cheek area for angled faces
        left_wide_indices = [70, 63, 105, 66, 107, 55, 65,
                             52, 53, 46, 124, 35, 111, 117, 118, 119, 120, 121, 128]
        right_wide_indices = [300, 293, 334, 296, 336, 285, 295,
                              282, 283, 276, 353, 265, 340, 346, 347, 348, 349, 350, 357]

        def get_roi(indices, padding=20):
            pts = face_points[indices]
            x_min = max(0, int(np.min(pts[:, 0])) - padding)
            x_max = min(w, int(np.max(pts[:, 0])) + padding)
            y_min = max(0, int(np.min(pts[:, 1])) - padding)
            y_max = min(h, int(np.max(pts[:, 1])) + padding)
            return image_matrix[y_min:y_max, x_min:x_max]

        left_roi = get_roi(left_eye_indices)
        right_roi = get_roi(right_eye_indices)
        left_wide_roi = get_roi(left_wide_indices, padding=10)
        right_wide_roi = get_roi(right_wide_indices, padding=10)
        nose_roi = get_roi(nose_bridge_indices, padding=10)

        if left_roi.size == 0 or right_roi.size == 0:
            return False

        left_indicators = 0
        right_indicators = 0
        nose_bridge_indicator = False

        for roi, wide_roi, side in [(left_roi, left_wide_roi, 'left'),
                                    (right_roi, right_wide_roi, 'right')]:
            count = 0

            blurred = cv2.GaussianBlur(roi, (3, 3), 0)

            # --- Method 1: Multi-angle edge detection ---
            edges = cv2.Canny(blurred, 50, 150)  # Raised from (30, 100)
            edge_density = np.sum(edges > 0) / max(edges.size, 1)

            if edge_density > 0.18:  # Raised from 0.10
                count += 1
                logger.debug(
                    f"{side} eye - edge density: {edge_density:.3f} (PASS)")
            else:
                logger.debug(f"{side} eye - edge density: {edge_density:.3f}")

            # --- Method 2: Line detection at ALL angles using HoughLinesP ---
            lines = cv2.HoughLinesP(
                edges, rho=1, theta=np.pi / 180,
                # Stricter: threshold 10->15, minLen 8->12, maxGap 5->3
                threshold=15, minLineLength=12, maxLineGap=3
            )
            long_lines = 0
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    if length > 15:  # Raised from 10
                        long_lines += 1

            if long_lines >= 5:  # Raised from 3
                count += 1
                logger.debug(f"{side} eye - long lines: {long_lines} (PASS)")
            else:
                logger.debug(f"{side} eye - long lines: {long_lines}")

            # --- Method 3: Contour-based frame detection ---
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            large_contours = [c for c in contours if cv2.arcLength(
                c, True) > 50]  # Raised from 30

            if len(large_contours) >= 3:  # Raised from 2
                count += 1
                logger.debug(
                    f"{side} eye - large contours: {len(large_contours)} (PASS)")
            else:
                logger.debug(
                    f"{side} eye - large contours: {len(large_contours)}")

            # --- Method 4: Reflection/glare detection ---
            _, bright = cv2.threshold(
                roi, 220, 255, cv2.THRESH_BINARY)  # Raised from 200
            bright_ratio = np.sum(bright > 0) / max(bright.size, 1)

            if 0.01 < bright_ratio < 0.15:  # Narrowed from (0.005, 0.2)
                count += 1
                logger.debug(
                    f"{side} eye - bright_ratio: {bright_ratio:.4f} (PASS)")
            else:
                logger.debug(f"{side} eye - bright_ratio: {bright_ratio:.4f}")

            # --- Method 5: Gradient magnitude (detects frame edges at any angle) ---
            grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
            avg_gradient = np.mean(gradient_mag)

            if avg_gradient > 40:  # Raised from 25
                count += 1
                logger.debug(
                    f"{side} eye - avg gradient: {avg_gradient:.2f} (PASS)")
            else:
                logger.debug(f"{side} eye - avg gradient: {avg_gradient:.2f}")

            # --- Method 6: Temple frame detection (wider ROI) ---
            if wide_roi.size > 0:
                wide_blurred = cv2.GaussianBlur(wide_roi, (3, 3), 0)
                # Raised from (40, 120)
                wide_edges = cv2.Canny(wide_blurred, 50, 150)
                # Check for diagonal/angled lines (temple arms)
                wide_lines = cv2.HoughLinesP(
                    wide_edges, rho=1, theta=np.pi / 180,
                    # Stricter: threshold 8->12, minLen 12->18, maxGap 5->3
                    threshold=12, minLineLength=18, maxLineGap=3
                )
                angled_lines = 0
                if wide_lines is not None:
                    for line in wide_lines:
                        x1, y1, x2, y2 = line[0]
                        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                        if length > 20:  # Raised from 15
                            dx = abs(x2 - x1)
                            dy = abs(y2 - y1)
                            # Angled line (not purely H or V)
                            if dx > 5 and dy > 5:  # Raised from 3
                                angled_lines += 1

                if angled_lines >= 3:  # Raised from 2
                    count += 1
                    logger.debug(
                        f"{side} eye - angled temple lines: {angled_lines} (PASS)")
                else:
                    logger.debug(
                        f"{side} eye - angled temple lines: {angled_lines}")

            if side == 'left':
                left_indicators = count
            else:
                right_indicators = count

        # --- Nose bridge check ---
        if nose_roi.size > 0:
            nose_blurred = cv2.GaussianBlur(nose_roi, (3, 3), 0)
            # Raised from (30, 100)
            nose_edges = cv2.Canny(nose_blurred, 50, 150)
            nose_edge_density = np.sum(
                nose_edges > 0) / max(nose_edges.size, 1)

            if nose_edge_density > 0.15:  # Raised from 0.08
                nose_bridge_indicator = True
                logger.debug(
                    f"Nose bridge - edge density: {nose_edge_density:.4f} (PASS)")
            else:
                logger.debug(
                    f"Nose bridge - edge density: {nose_edge_density:.4f}")

        logger.info(
            f"Glasses indicators - Left: {left_indicators}/6,"
            f" Right: {right_indicators}/6, Nose: {nose_bridge_indicator}")

        # Detection logic - STRICTER to reduce false positives:
        # 1. Both eyes need 3+ indicators (was 2)
        # 2. One eye has 4+ and nose bridge detected (was 3)
        # 3. Combined indicators >= 7 (was 5)
        has_glasses = (
            (left_indicators >= 3 and right_indicators >= 3)
            or ((left_indicators >= 4 or right_indicators >= 4) and nose_bridge_indicator)
            or (left_indicators + right_indicators >= 7)
        )

        return has_glasses
    # def check_for_glasses(self, face_landmarks):
    #     eye_inds = [33, 133, 157, 158, 159, 160, 161, 246, 7, 163, 144, 145, 153, 154, 155,
    #                 362, 263, 384, 385, 386, 387, 388, 466, 249, 390, 373, 374, 380, 381, 382]
    #     nose_inds = [168, 6, 197, 195, 5]

    #     avg_eye_vis = np.mean([face_landmarks[i].visibility for i in eye_inds])
    #     avg_nose_vis = np.mean([face_landmarks[i].visibility for i in nose_inds])

    #     # Thresholds – tune on your data
    #     return avg_eye_vis < 0.7 or avg_nose_vis < 0.6

    def extract_landmarks(self, image: Image.Image) -> tuple[np.ndarray, dict[str, float],  list[Any]]:  # noqa: E501
        img_rgb = np.array(image.convert('RGB'))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

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


frontal_classifier = ProperFaceDetector()
