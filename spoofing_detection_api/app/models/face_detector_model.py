from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort
from app.core.config import settings
from PIL import Image


class FaceDetectorModel:
    def __init__(self, model_path, confidence_cutoff=0.7):
        self.threshold = confidence_cutoff
        self.model = self._load_model(model_path)
        self.target_size = (320, 240)          # (width, height)
        self.priors = self._generate_priors()  # correct shape (4420, 4)

    def _load_model(self, model_path):
        options = ort.SessionOptions()
        options.log_severity_level = 3
        return ort.InferenceSession(model_path, sess_options=options)

    def _generate_priors(self):
        """
        Generate 4420 anchor boxes for the RFB-320 model.
        Returns array of shape (4420, 4) in [xmin, ymin, xmax, ymax] normalized (0..1).
        """
        img_w, img_h = self.target_size
        # Feature map sizes (width, height) for strides [8, 16, 32, 64]
        fmap_sizes = [(40, 30), (20, 15), (10, 8), (5, 4)]
        # Anchor scales (in pixels) – adjust if your model uses different values
        scales = [
            [16],           # map0: one scale + aspect ratios → 3 anchors per cell
            [64, 128],      # map1: two scales, no aspect ratios → 2 anchors per cell
            [128, 256],     # map2: two scales, no aspect ratios → 2 anchors per cell
            [256]           # map3: one scale + aspect ratios → 3 anchors per cell
        ]
        # Aspect ratios for maps that use them (both r and 1/r are generated)
        aspect_ratios = [
            [2],   # for map0
            [],    # no aspect ratios for map1
            [],    # no aspect ratios for map2
            [2]    # for map3
        ]

        priors = []
        for i, (f_w, f_h) in enumerate(fmap_sizes):
            scale_list = scales[i]
            ar_list = aspect_ratios[i]

            for y in range(f_h):
                for x in range(f_w):
                    cx = (x + 0.5) / f_w
                    cy = (y + 0.5) / f_h

                    # Square boxes for each scale
                    for s in scale_list:
                        w = s / img_w
                        h = s / img_h
                        priors.append([cx - w/2, cy - h/2, cx + w/2, cy + h/2])

                    # Aspect ratio boxes for each scale (only if ar_list is non-empty)
                    for s in scale_list:
                        for ar in ar_list:
                            # Box with aspect ratio ar (width larger)
                            w = s * np.sqrt(ar) / img_w
                            h = s / np.sqrt(ar) / img_h
                            priors.append(
                                [cx - w/2, cy - h/2, cx + w/2, cy + h/2])
                            # Box with aspect ratio 1/ar (height larger)
                            w = s / np.sqrt(ar) / img_w
                            h = s * np.sqrt(ar) / img_h
                            priors.append(
                                [cx - w/2, cy - h/2, cx + w/2, cy + h/2])

        priors = np.array(priors, dtype=np.float32)
        # Sanity check: should be 4420
        assertExpect = f"Expected 4420 priors, got {priors.shape[0]}"
        assert priors.shape[0] == 4420, assertExpect

        return priors

    def _decode_boxes(self, deltas, priors):
        """
        Convert anchor deltas to absolute [x1, y1, x2, y2] in normalized coordinates.
        deltas: (num_priors, 4) – raw model output (dx, dy, dw, dh)
        priors: (num_priors, 4) – anchor boxes in [x1, y1, x2, y2] normalized
        """
        # Convert priors to [cx, cy, w, h]
        prior_cx = (priors[:, 0] + priors[:, 2]) / 2
        prior_cy = (priors[:, 1] + priors[:, 3]) / 2
        prior_w = priors[:, 2] - priors[:, 0]
        prior_h = priors[:, 3] - priors[:, 1]

        # Deltas
        dx = deltas[:, 0]
        dy = deltas[:, 1]
        dw = deltas[:, 2]
        dh = deltas[:, 3]

        # Typical variances used in the original repo
        center_variance = 0.1
        size_variance = 0.2
        cx = dx * center_variance * prior_w + prior_cx
        cy = dy * center_variance * prior_h + prior_cy
        w = np.exp(dw * size_variance) * prior_w
        h = np.exp(dh * size_variance) * prior_h

        # Convert back to [x1, y1, x2, y2] and clip
        x1 = np.clip(cx - w / 2, 0, 1)
        y1 = np.clip(cy - h / 2, 0, 1)
        x2 = np.clip(cx + w / 2, 0, 1)
        y2 = np.clip(cy + h / 2, 0, 1)
        return np.stack([x1, y1, x2, y2], axis=1)

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        image = image.resize(self.target_size)
        img = np.array(image, dtype=np.float32)
        img = (img - 127.0) / 128.0          # normalization used by the model
        img = np.transpose(img, [2, 0, 1])    # HWC -> CHW
        img = np.expand_dims(img, axis=0)     # add batch
        return img

    def _detect_faces(self, session: ort.InferenceSession, image_np: np.ndarray) -> tuple:
        confidences, boxes = session.run(None, {'input': image_np})
        # confidences: (1, 4420, 2) -> take face class (index 1)
        assert isinstance(confidences, np.ndarray)
        face_scores = confidences[0, :, 1]
        # boxes: (1, 4420, 4) -> raw deltas
        assert isinstance(boxes, np.ndarray)
        raw_deltas = boxes[0]
        return face_scores, raw_deltas

    def find_faces(self, image: Image.Image) -> list:
        orig_w, orig_h = image.size
        input_tensor = self._preprocess(image)
        scores, deltas = self._detect_faces(self.model, input_tensor)

        # Decode deltas using anchors to get normalized [x1,y1,x2,y2]
        boxes_norm = self._decode_boxes(deltas, self.priors)

        # Filter by confidence threshold
        mask = scores > self.threshold
        scores = scores[mask]
        boxes_norm = boxes_norm[mask]
        if len(scores) == 0:
            return []

        # Convert normalized boxes to absolute pixel coordinates in 320x240 space
        boxes_abs = boxes_norm.copy()
        boxes_abs[:, 0] *= self.target_size[0]   # x1
        boxes_abs[:, 1] *= self.target_size[1]   # y1
        boxes_abs[:, 2] *= self.target_size[0]   # x2
        boxes_abs[:, 3] *= self.target_size[1]   # y2

        # Convert to [x, y, width, height] for OpenCV NMS
        x = boxes_abs[:, 0]
        y = boxes_abs[:, 1]
        w = boxes_abs[:, 2] - boxes_abs[:, 0]
        h = boxes_abs[:, 3] - boxes_abs[:, 1]
        boxes_xywh = np.column_stack([x, y, w, h])

        # Apply OpenCV NMS
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(),
            scores.tolist(),
            self.threshold,
            0.3
        )

        if len(indices) == 0:
            return []
        indices = np.array(indices)
        indices = indices.flatten()
        keep_boxes = boxes_abs[indices]
        keep_scores = scores[indices]

        # Scale to original image size and build result list
        faces = []
        for score, box in zip(keep_scores, keep_boxes):
            x1 = int(box[0] * orig_w / self.target_size[0])
            y1 = int(box[1] * orig_h / self.target_size[1])
            x2 = int(box[2] * orig_w / self.target_size[0])
            y2 = int(box[3] * orig_h / self.target_size[1])
            faces.append({
                'confidence': float(score),
                'bbox': [x1, y1, x2, y2]
            })
        return faces


# Singleton instance
face_detector = FaceDetectorModel(model_path=settings.FACE_DETECTOR_MODEL_PATH)
