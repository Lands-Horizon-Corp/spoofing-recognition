from spoofdet.efficient_net.model_utils import get_model


import torch
import numpy as np
from app.core.config import settings

from spoofdet.efficient_net.model_utils import get_model
from spoofdet.data_processing import get_transform_pipeline


class SpoofDetector:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            print("Creating the object for the first time...")
            cls._instance = super(SpoofDetector, cls).__new__(cls)
            # Initialize your heavy setup here (e.g. loading weights)
            cls._instance._load_model()
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.model = self._load_model()
        self.version = "1.0"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self._initialized = True

    def _load_model(self):
        # Load your trained model here
        # Example for PyTorch:
        model = get_model()
        model.load_state_dict(torch.load(settings.MODEL_PATH))
        model.eval()
        return model

    def predict(self, image: np.ndarray) -> tuple:
        processed = self.preprocess(image)
        with torch.no_grad():
            outputs = self.model(processed)
            probs = torch.sigmoid(outputs)
            prediction = (probs[:, 1] > settings.MODEL_THRESHOLD).long()
            confidence = probs[:, 1].item()
        return prediction, confidence

    def preprocess(self, image):
        _, gpu_transform_val = get_transform_pipeline(device=self.device)
        image = gpu_transform_val(image)

        return image
