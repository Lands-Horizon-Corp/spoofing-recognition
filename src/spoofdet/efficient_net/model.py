from __future__ import annotations

from typing import cast

import lightning as L
import torch
from torch import nn
from torchvision import models
from torchvision.transforms import v2


class EfficientNet(L.LightningModule):
    def __init__(self, model, val_transforms, train_transforms, criterion):
        super().__init__()
        self.model: nn.Module = model
        self.train_transforms: v2.Compose = train_transforms
        self.val_transforms: v2.Compose = val_transforms
        self.criterion: nn.Module = criterion

    def training_step(self, batch):
        self.model.train()
        for img, label in batch:
            img = self.train_transforms(img)
            output = self.model(img)
            loss = self.criterion(output, label)
            self.log('train_loss', loss)

    def validation_step(self, batch):
        self.model.eval()
        for img, label in batch:
            img = self.val_transforms(img)
            output = self.model(img)
            loss = self.criterion(output, label)
            self.log('val_loss', loss)


def get_model(with_weights: bool = False, device: torch.device = torch.device('cpu')) -> nn.Module:
    """Getting the EfficientNet v2 small model for either training or inference"""

    if with_weights:
        model = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.DEFAULT,
        )
    else:
        model = models.efficientnet_v2_s(weights=None)

    in_features = cast(nn.Linear, model.classifier[1]).in_features
    model.classifier[1] = nn.Linear(in_features, 2)

    return model.to(device)


def freeze_stages(model: nn.Module, num_unfrozen_stages: int):
    """Freezes the initial layers of the model based on num_unfrozen_stages"""
    total_stages_unfreeze = 7 - num_unfrozen_stages
    features = list(cast(nn.Module, model.features).children())
    for i in range(total_stages_unfreeze):
        for param in features[i].parameters():
            param.requires_grad = False

    print(
        f"Unfrozen layers: {total_stages_unfreeze}",
    )


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
            imgs = val_transforms(imgs)
            model(imgs)
