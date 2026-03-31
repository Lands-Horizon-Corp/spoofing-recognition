from __future__ import annotations

from typing import cast

import torch
from torch import nn


def freeze_stages(model: nn.Module, frozen_stages: int):
    """
    Freezes the initial layers of the model.
    Robust to structure (Torchvision vs Timm) and Pylance type checking.
    """

    # 1. Handle Timm Models (MobileNetV4, etc.)
    if hasattr(model, 'blocks'):
        print('Detected Timm model structure.')

        # Freeze Stem (if present)
        if hasattr(model, 'conv_stem'):
            for param in cast(nn.Module, model.conv_stem).parameters():
                param.requires_grad = False

        if hasattr(model, 'bn1'):
            for param in cast(nn.Module, model.bn1).parameters():
                param.requires_grad = False

        # Freeze Blocks
        # We explicitly iterate over children to ensure they are Modules
        blocks = list(cast(nn.ModuleList, model.blocks).children())
        limit = min(frozen_stages, len(blocks))

        for i in range(limit):
            block = blocks[i]
            # SAFETY CHECK: Ensure it is actually a Module before accessing parameters
            if isinstance(block, nn.Module):
                for param in block.parameters():
                    param.requires_grad = False
            else:

                print(f"Warning: Block {i} is not an nn.Module.")

        print(f"Frozen Timm backbone: Stem + first {limit} blocks")

    # 2. Handle Torchvision Models (EfficientNet, ResNet, etc.)
    elif hasattr(model, 'features'):
        print('Detected Torchvision model structure.')
        features = list(cast(nn.ModuleList, model.features).children())
        limit = min(frozen_stages, len(features))

        for i in range(limit):
            feature = features[i]
            if isinstance(feature, nn.Module):
                for param in feature.parameters():
                    param.requires_grad = False

        print(f"Frozen Torchvision features: first {limit} layers")

    else:
        print('Warning: Model structure unknown. Skipping freeze.')


def adaptive_batch_norm(model, val_transforms, data_loader, device, num_batches=100, momentum=0.1):
    """Adapts the batch normalization layers of the model using a subset of the training data"""

    model.train()
    # reset running mean and variance for all batch normalization layers
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            module.reset_running_stats()
            module.momentum = momentum

    with torch.no_grad():

        for i, (imgs, _) in enumerate(data_loader):
            if i >= num_batches:
                break
            imgs = imgs.to(device)
            model(imgs)
    print('Adaptive BatchNorm completed')
