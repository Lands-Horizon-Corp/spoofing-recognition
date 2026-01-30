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
            cls._instance = super(SpoofDetector, cls).__new__(cls)
        cls._instance.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.model = self._load_model()
        self.version = "1.0"
        self._initialized = True

    def _load_model(self):
        model = get_model()
        model.to(device=self.device)
        model.load_state_dict(torch.load(settings.MODEL_PATH, map_location=self.device))
        model.eval()
        return model

    def predict(self, image: np.ndarray) -> tuple:
        processed = self.preprocess(image)
        with torch.no_grad():
            outputs = self.model(processed)
            probs = torch.sigmoid(outputs)
            prediction = (probs[:, 1] > settings.MODEL_THRESHOLD).long()
            spoof_confidence = probs[:, 1].item()
            live_confidence = probs[:, 0].item()
        return prediction, live_confidence, spoof_confidence

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        assert image.dtype == np.uint8, "Image dtype must be uint8"
        _, gpu_transform_val = get_transform_pipeline(
            device=self.device, target_size=settings.MODEL_TARGET_SIZE
        )
        if isinstance(image, np.ndarray):
            # Convert NumPy (H, W, C) -> Tensor (C, H, W)
            image = torch.from_numpy(image).permute(2, 0, 1)
        processed_image = gpu_transform_val(image).unsqueeze(0).to(self.device)
        assert (
            processed_image.ndim == 4
        ), f"Preprocessed image must have 4 dimensions: {processed_image.shape} {processed_image.ndim}"
        print(f"Processed image shape: {processed_image.shape}")
        return processed_image


if __name__ == "__main__":
    detector = SpoofDetector()

    print("Model loaded successfully.")
    # params needed
    print(
        f"threshold: {settings.MODEL_THRESHOLD} target_size: {settings.MODEL_TARGET_SIZE}"
    )
