from __future__ import annotations

import torch
from spoofdet.config import MEAN
from spoofdet.config import STD
from torchvision.transforms import v2


def get_transform_pipeline(
    target_size: int,
) -> tuple[v2.Compose, v2.Compose]:
    """
    Returns training and validation transform pipelines moved to the specified device.
    """

    gpu_transforms_train = v2.Compose(
        [
            # v2.ToImage(),
            # v2.RandomResizedCrop(
            #     size=(target_size, target_size),
            #     scale=(0.7, 1.0),  # Zoom range
            #     ratio=(0.75, 1.33),
            #     antialias=True,
            # ),
            v2.RandomHorizontalFlip(p=0.5),
            # v2.RandomRotation(degrees=30),
            # v2.RandomPerspective(distortion_scale=0.3, p=0.2),
            # v2.RandomAffine(
            #     degrees=0,
            #     translate=(0.1, 0.1),  # Shift left/right/up/down
            #     scale=(0.8, 1.2),  # Zoom In AND Zoom Out (crucial!)
            # ),
            v2.ToDtype(torch.float32, scale=True),
            v2.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0,
            ),
            v2.RandomGrayscale(p=0.1),
            # v2.Grayscale(num_output_channels=3),
            # v2.RandomGrayscale(p=0.1),
            # v2.GaussianBlur(kernel_size=3, sigma=(0.2, 2.0)),
            # v2.GaussianNoise(sigma=0.02),
            v2.RandomErasing(p=0.2),
            v2.Normalize(mean=MEAN, std=STD),
            # v2.RandomChoice(
            #     [
            #         v2.MixUp(num_classes=2, alpha=0.2),
            #         v2.CutMix(num_classes=2, alpha=1.0),
            #     ]
            # ),
        ],
    )

    gpu_transforms_val = v2.Compose(
        [
            # v2.Resize((target_size, target_size), antialias=True),
            # v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=MEAN, std=STD),
        ],
    )
    return gpu_transforms_train, gpu_transforms_val
