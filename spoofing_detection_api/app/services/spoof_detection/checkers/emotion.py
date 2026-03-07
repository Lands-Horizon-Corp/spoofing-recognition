from __future__ import annotations

import numpy as np
from app.models.deep_face.utils import FaceAnalysis
from PIL.Image import Image


class EmotionChecker:
    def __init__(self):
        pass

    def load_model(self):
        try:
            self.model = FaceAnalysis(model_name='emotion')
        except Exception as e:
            print(f"Error loading emotion detection model: {e}")
            self.model = None

    def detect(self, image: Image) -> str | int | None:
        try:
            img_array = np.array(image)
            assert self.model is not None, 'Emotion detection model is not loaded'
            analysis = self.model.predict(img_array)
            return analysis
        except Exception as e:
            print(f"Error in emotion detection: {e}")
            return None


emotion_checker = EmotionChecker()
