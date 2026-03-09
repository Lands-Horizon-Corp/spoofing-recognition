from __future__ import annotations

import numpy as np
import onnxruntime as ort
from app.core.config import model_config
from app.core.config import settings
from app.core.utils import calculate_sigmoid
from spoofdet.verify_memory import print_memory_usage


class SpoofDetector:
    def __init__(self):
        self.model = None
        print_memory_usage('SpoofDetector Initialized')

    def load_model(self):
        if self.model is not None:
            return self.model
        ort_session = ort.InferenceSession(settings.MODEL_PATH)
        self.model = ort_session
        print_memory_usage('Model Loaded into SpoofDetector')

    def predict(self, image: np.ndarray) -> tuple:
        self.load_model()
        assert self.model is not None, 'Model session is not loaded'
        processed = self._preprocess_img(image)
        model_input_name = self.model.get_inputs()[0].name

        ort_inputs = {model_input_name: processed}

        outputs = self.model.run(None, ort_inputs)
        spoof_confidence = np.array(outputs[0]).item()
        spoof_confidence = calculate_sigmoid(spoof_confidence)
        prediction = bool(spoof_confidence >
                          model_config.THRESHOLD)
        return prediction, spoof_confidence

    def _preprocess_img(self, img_np: np.ndarray) -> np.ndarray:
        assert img_np.dtype == np.uint8, f'Image dtype must be uint8 {img_np.dtype}'  # noqa: E501
        img_np = img_np.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np - mean) / std
        img_np = img_np.transpose((2, 0, 1))

        # (C, H, W) -> (1, C, H, W)
        img_np = np.expand_dims(img_np, axis=0)

        assert (
            img_np.ndim == 4
        ), 'image must have 4 dimensions: \n'
        f"{img_np.shape} \n"
        f"{img_np.ndim}"
        return img_np


spoof_detector = SpoofDetector()

if __name__ == '__main__':
    detector = SpoofDetector()

    print('Model loaded successfully.')
    print(
        f"threshold: {model_config.THRESHOLD}"
        f" target_size: {model_config.TARGET_SIZE}")
