from __future__ import annotations

import numpy as np
import torch
from app.core.config import settings

from spoofdet.data_processing import get_transform_pipeline
from spoofdet.efficient_net.model_utils import get_model


class SpoofDetector:
    _instance = None
    _initialized = False
    device: torch.device  # Add type annotation for device

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu',
            )
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.model = self._load_model()
        self.version = '1.0'
        self._initialized = True

    def _load_model(self):
        model = get_model()
        model.to(device=self.device)
        model.load_state_dict(
            torch.load(
                settings.MODEL_PATH,
                map_location=self.device,
            ),
        )
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

    def preprocess(self, input_image: np.ndarray) -> torch.Tensor:
        assert input_image.dtype == np.uint8, 'Image dtype must be uint8'
        _, gpu_transform_val = get_transform_pipeline(
            device=self.device,
            target_size=settings.MODEL_TARGET_SIZE,
        )
        if isinstance(input_image, np.ndarray):
            # Convert NumPy (H, W, C) -> Tensor (C, H, W)
            image_tensor: torch.Tensor = torch.from_numpy(
                input_image,
            ).permute(2, 0, 1)
        processed_image = (
            gpu_transform_val(
                image_tensor,
            )
            .unsqueeze(0)
            .to(self.device)
        )
        assert (
            processed_image.ndim == 4
        ), 'image must have 4 dimensions: \n' \
            f"{processed_image.shape} \n" \
            f"{processed_image.ndim}"
        return processed_image


if __name__ == '__main__':
    detector = SpoofDetector()

    print('Model loaded successfully.')
    print( f"threshold: {settings.MODEL_THRESHOLD} target_size: {settings.MODEL_TARGET_SIZE}")
