from __future__ import annotations

import numpy as np
import onnxruntime
from app.core.config import settings
from app.core.utils import calculate_sigmoid
from PIL import Image


class GlassDetectorModel:
    def __init__(self,):
        self.is_model_loaded = False
        self.model = None

    def load_model(self):
        if not self.is_model_loaded:
            # Load the ONNX model here
            self.model = onnxruntime.InferenceSession(
                settings.GLASS_DETECTOR_MODEL_PATH)
            self.is_model_loaded = True

    def predict(self, img: Image.Image) -> bool:
        self.load_model()
        assert self.model is not None, 'Model is not loaded.'
        img_array = self.preprocess(img)
        outputs = self.model.run(None, {'input': img_array})
        print(f"Raw model output: {outputs}")
        logit = np.array(outputs[0]).flatten()[
            0]  # Extract single scalar value
        confidence = calculate_sigmoid(logit)
        result = confidence > 0.5
        print(
            f"Glass detection result: {result}",
            f"confidence: {confidence}")
        return result

    def preprocess(self, image: Image.Image) -> np.ndarray:
        img = image.resize((256, 256))
        img = np.array(img).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img


glass_detector = GlassDetectorModel()
