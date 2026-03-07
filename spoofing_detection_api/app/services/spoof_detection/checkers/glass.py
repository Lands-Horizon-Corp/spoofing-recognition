from __future__ import annotations

import numpy as np
import onnxruntime
from app.core.config import settings
from app.core.utils import calculate_sigmoid
from PIL import Image


class GlassChecker:
    def __init__(self,):
        self.model = None

    def load_model(self):
        if self.model is None:
            self.model = onnxruntime.InferenceSession(
                settings.GLASS_DETECTOR_MODEL_PATH)

    def detect_glasses(self, img: Image.Image) -> bool:
        self.load_model()
        assert self.model is not None, 'Model is not loaded.'
        img_array = self.preprocess(img)
        outputs = self.model.run(None, {'input': img_array})
        print(f"Raw model output: {outputs}")
        logit = np.array(outputs[0]).flatten()[
            0]  # Extract single scalar value
        confidence = calculate_sigmoid(logit)
        result = confidence > 0.4
        print(
            f"Glass detection result: {result}",
            f"confidence: {confidence}")
        return result

    def preprocess(self, image: Image.Image) -> np.ndarray:
        img = image.resize((256, 256))
        img = np.array(img).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img


glass_checker = GlassChecker()
