from __future__ import annotations

from app.models.deep_face.utils import FaceAnalysis


class EmotionDetection:
    def __init__(self):
        pass

    def detect(self, image_path):
        try:
            analysis = FaceAnalysis(model_name='emotion').predict(image_path)
            return analysis
        except Exception as e:
            print(f"Error in emotion detection: {e}")
            return None


emotion_detector = EmotionDetection()
