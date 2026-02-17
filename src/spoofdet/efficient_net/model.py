from __future__ import annotations

from typing import cast

import lightning as L
import matplotlib.pyplot as plt
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from spoofdet.efficient_net.model_utils import adaptive_batch_norm
from spoofdet.efficient_net.model_utils import freeze_stages
from torch import nn
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.classification import BinaryConfusionMatrix
from torchmetrics.classification import BinaryF1Score
from torchmetrics.classification import BinaryPrecision
from torchmetrics.classification import BinaryRecall
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.classification import MulticlassPrecision
from torchmetrics.classification import MulticlassRecall
from torchvision import models
from torchvision.transforms import v2


class EfficientNetSpoofingDetection(L.LightningModule):
    def __init__(self,
                 backbone_lr=1e-5,
                 head_lr=3e-4,
                 val_transforms=None,
                 train_transforms=None,
                 train_data_loader=None,
                 criterion=torch.nn.BCEWithLogitsLoss()):
        super().__init__()
        self.save_hyperparameters(
            ignore=['val_transforms',
                    'train_transforms',
                    'train_data_loader',
                    'criterion',
                    'test_acc',
                    'test_precision',
                    'test_recall',
                    'test_f1',
                    'test_precision_perclass',
                    'test_recall_perclass',
                    'test_f1_perclass',
                    'test_confmat'])
        self.train_data_loader = train_data_loader
        self.train_transforms: v2.Compose | None = train_transforms
        self.val_transforms: v2.Compose | None = val_transforms
        self.criterion: nn.Module = criterion
        self.backbone_lr: float = backbone_lr
        self.head_lr: float = head_lr
        self.backbone: nn.Module = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.DEFAULT,
        )
        self.in_features: int = cast(
            nn.Linear, self.backbone.classifier[1]).in_features
        self.backbone.classifier[1] = nn.Identity()
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

        self.test_acc = BinaryAccuracy()
        self.test_precision = BinaryPrecision()
        self.test_recall = BinaryRecall()
        self.test_f1 = BinaryF1Score()

        self.test_precision_perclass = MulticlassPrecision(
            num_classes=2, average=None)
        self.test_recall_perclass = MulticlassRecall(
            num_classes=2, average=None)
        self.test_f1_perclass = MulticlassF1Score(num_classes=2, average=None)
        self.test_confmat = BinaryConfusionMatrix()
        self._test_total_samples = 0

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

    def on_fit_start(self) -> None:
        freeze_stages(self.backbone, frozen_stages=3)
        adaptive_batch_norm(self.backbone, self.val_transforms,
                            self.train_data_loader, self.device)

    def training_step(self, batch, batch_idx):
        img, label = batch
        label = label.view(-1, 1).float()
        if self.train_transforms is not None:
            img = self.train_transforms(img)
        output = self.forward(img)
        loss = self.criterion(output, label)
        self.log('train_loss', loss,  prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, label = batch
        label = label.view(-1, 1).float()
        if self.val_transforms is not None:
            img = self.val_transforms(img)
        output = self.forward(img)
        loss = self.criterion(output, label)
        self.log('val_loss', loss, on_epoch=True,
                 prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([
            {'params': self.backbone.parameters(), 'lr': self.backbone_lr},
            {'params': self.head.parameters(), 'lr': self.head_lr},
        ], weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'  # This must exactly match your self.log() name
        }

    def test_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.view(-1, 1).float()   # [B, 1]
        if self.val_transforms is not None:
            images = self.val_transforms(images)  # [B, C, H, W]
        logits = self(images)                 # [B, 1]
        loss = self.criterion(logits, labels)

        # Convert logits → binary predictions (0 or 1)
        # liveniness should be 95% confident
        preds = (torch.sigmoid(logits) > 0.05).int()

        # ----- UPDATE ALL METRICS -----
        self.test_acc.update(preds, labels.int())
        self.test_precision.update(preds, labels.int())
        self.test_recall.update(preds, labels.int())
        self.test_f1.update(preds, labels.int())

        # Per‑class metrics (average=None)
        self.test_precision_perclass.update(preds, labels.int())
        self.test_recall_perclass.update(preds, labels.int())
        self.test_f1_perclass.update(preds, labels.int())
        self.test_confmat.update(preds, labels.int())

        self._test_total_samples += labels.size(0)

        # Log test loss (optional)
        self.log('test_loss', loss, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        # ----- 1. Overall scalar metrics -----
        if self._test_total_samples == 0:
            print('No test samples processed – skipping test metrics.')
            return
        acc = self.test_acc.compute()
        prec = self.test_precision.compute()
        rec = self.test_recall.compute()
        f1 = self.test_f1.compute()

        self.log('test/acc', acc)
        self.log('test/precision', prec)
        self.log('test/recall', rec)
        self.log('test/f1', f1)

        # ----- 2. Per‑class metrics (LIVE = class 0, SPOOF = class 1) -----
        # [prec_live, prec_spoof]
        prec_perclass = self.test_precision_perclass.compute()
        # [rec_live, rec_spoof]
        rec_perclass = self.test_recall_perclass.compute()
        # [f1_live, f1_spoof]
        f1_perclass = self.test_f1_perclass.compute()
        confmat = self.test_confmat.compute()                   # 2x2 tensor
        if rec_perclass.numel() < 2:
            print(f" Unexpected per‑class tensor shape: {
                  rec_perclass.shape}")
            return

        # ----- 3. Anti‑spoofing specific metrics -----
        # APCER = Spoof misclassified as Live = 1 - Recall_Spoof
        # BPCER = Live misclassified as Spoof = 1 - Recall_Live
        apcer = 1.0 - rec_perclass[1]   # class 1 = spoof
        bpcer = 1.0 - rec_perclass[0]   # class 0 = live
        hter = (apcer + bpcer) / 2.0    # Half Total Error Rate

        self.log('test/apcer', apcer)
        self.log('test/bpcer', bpcer)
        self.log('test/hter', hter)

        # Log per‑class metrics for inspection
        self.log('test/precision_live', prec_perclass[0])
        self.log('test/precision_spoof', prec_perclass[1])
        self.log('test/recall_live', rec_perclass[0])      # = 1 - BPCER
        self.log('test/recall_spoof', rec_perclass[1])     # = 1 - APCER
        self.log('test/f1_live', f1_perclass[0])
        self.log('test/f1_spoof', f1_perclass[1])

        # ---- Create figure ----
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        confmat_np = confmat.cpu().numpy().astype(int)
        im = ax.imshow(confmat_np, cmap='Blues', interpolation='nearest')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Live', 'Spoof'])
        ax.set_yticklabels(['Live', 'Spoof'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(confmat_np[i, j]),
                        ha='center', va='center', color='black')
        plt.colorbar(im)
        cast(TensorBoardLogger, self.logger).experiment.add_figure(
            'test/confusion_matrix', fig, global_step=self.current_epoch)
        plt.close(fig)

        # ----- 5. IMPORTANT: Reset metrics for next test run -----
        self.test_acc.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_f1.reset()
        self.test_precision_perclass.reset()
        self.test_recall_perclass.reset()
        self.test_f1_perclass.reset()
        self.test_confmat.reset()
        self._test_total_samples = 0


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
