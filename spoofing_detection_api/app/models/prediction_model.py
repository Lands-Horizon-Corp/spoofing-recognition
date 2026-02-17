from __future__ import annotations

import numpy as np
import torch
from app.core.config import ModelConfig
from app.core.config import settings
from spoofdet.verify_memory import print_memory_usage


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
        self.model = None
        self.version = '1.0'
        self.config = ModelConfig()
        self._initialized = True
        print_memory_usage('SpoofDetector Initialized')

    def _load_model(self):
        if self.model is None:
            model = torch.jit.load(
                settings.MODEL_PATH, map_location=self.device)
            model.to(device=self.device)
            model = torch.compile(model, backend='inductor')
            model.eval()
            self.model = model
            print_memory_usage('[model] Model Loaded')
            return self.model

    def predict(self, image: np.ndarray) -> tuple:
        self._load_model()
        processed = self.preprocess(image)
        with torch.no_grad():
            assert self.model is not None, 'Model must be loaded before prediction'
            outputs = self.model(processed)
            probs = torch.sigmoid(outputs)
            prediction = (probs > self.config.THRESHOLD).long()
            spoof_confidence = probs
        return prediction, spoof_confidence

    def preprocess(self, input_image: np.ndarray) -> torch.Tensor:
        assert input_image.dtype == np.uint8, 'Image dtype must be uint8'
        # _, gpu_transform_val = get_transform_pipeline(
        #     device=self.device,
        #     target_size=self.config.TARGET_SIZE,
        # )
        # if isinstance(input_image, np.ndarray):
        # Convert NumPy (H, W, C) -> Tensor (C, H, W)
        # image_tensor: torch.Tensor = torch.from_numpy(
        #     input_image,
        # ).permute(2, 0, 1)
        processed_image = torch.from_numpy(
            self._preprocess_img(input_image)
                .astype(np.float32)
        )

        assert (
            processed_image.ndim == 4
        ), 'image must have 4 dimensions: \n'
        f"{processed_image.shape} \n"
        f"{processed_image.ndim}"
        return processed_image

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


if __name__ == '__main__':
    detector = SpoofDetector()

    print('Model loaded successfully.')
    print(
        f"threshold: {detector.config.THRESHOLD}"
        f" target_size: {detector.config.TARGET_SIZE}")
