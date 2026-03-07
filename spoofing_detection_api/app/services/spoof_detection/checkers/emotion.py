from __future__ import annotations

import numpy as np
from app.models.deep_face.utils import FaceAnalysis
from PIL.Image import Image


class EmotionChecker:
    def __init__(self):
        self.model = None

    def load_model(self):
        if self.model is None:
            self.model = FaceAnalysis(model_name='emotion')

    def detect(self, image: Image) -> str | int | None:
        self.load_model()
        try:
            img_array = np.array(image)
            assert self.model is not None, 'Emotion detection model is not loaded'
            analysis = self.model.predict(img_array)
            return analysis
        except Exception as e:
            print(f"Error in emotion detection: {e}")
            return None


emotion_checker = EmotionChecker()
