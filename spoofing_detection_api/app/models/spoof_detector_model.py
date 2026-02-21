from __future__ import annotations

import numpy as np
import onnxruntime
import onnxruntime as ort
from app.core.config import model_config
from app.core.config import settings
from spoofdet.verify_memory import print_memory_usage


def calculate_sigmoid(x):
    return 1 / (1 + np.exp(-x))


class SpoofDetector:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.version = '1.0'
        self.config = model_config
        self._initialized = True
        print_memory_usage('SpoofDetector Initialized')

    def load_model(self) -> onnxruntime.InferenceSession:
        ort_session = ort.InferenceSession(settings.MODEL_PATH)
        print_memory_usage('Model Loaded into SpoofDetector')
        return ort_session

    def preprocess(self, input_image: np.ndarray) -> np.ndarray:
        assert input_image.dtype == np.uint8, 'Image dtype must be uint8'

        processed_image = self._preprocess_img(input_image)
        processed_image = processed_image.astype(np.float32)
        assert (
            processed_image.ndim == 4
        ), 'image must have 4 dimensions: \n'
        f"{processed_image.shape} \n"
        f"{processed_image.ndim}"
        return processed_image

    def predict(self, image: np.ndarray, session: onnxruntime.InferenceSession) -> tuple:
        processed = self.preprocess(image)
        model_input_name = session.get_inputs()[0].name
        ort_inputs = {model_input_name: processed}
        outputs = session.run(None, ort_inputs)
        spoof_confidence = np.array(outputs[0]).item()
        spoof_confidence = calculate_sigmoid(spoof_confidence)
        prediction = (spoof_confidence >
                      self.config.THRESHOLD).astype(np.int32)
        return prediction, spoof_confidence

    def _preprocess_img(self, img_np: np.ndarray) -> np.ndarray:

        img_np = img_np.astype(np.float32) / 255.0
        # 1. Load Image
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Apply normalization: (pixel - mean) / std
        # Broadcasting works automatically here (H, W, 3) - (3,)
        img_np = (img_np - mean) / std

        # 5. Transpose Layout (HWC -> CHW)
        # PyTorch/ONNX expects [Channels, Height, Width], but PIL gives [Height, Width, Channels]
        img_np = img_np.transpose((2, 0, 1))

        # 6. Add Batch Dimension (C, H, W) -> (1, C, H, W)
        img_np = np.expand_dims(img_np, axis=0)

        return img_np


spoof_detector = SpoofDetector()

if __name__ == '__main__':
    detector = SpoofDetector()

    print('Model loaded successfully.')
    print(
        f"threshold: {detector.config.THRESHOLD}"
        f" target_size: {detector.config.TARGET_SIZE}")
