from __future__ import annotations

from typing import cast
from typing import Literal

import lightning as L
import matplotlib.pyplot as plt
import timm
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
                 backbone_model,
                 backbone_lr=1e-5,
                 head_lr=3e-4,
                 val_transforms=None,
                 train_transforms=None,
                 train_data_loader=None,
                 criterion=torch.nn.BCEWithLogitsLoss(),
                 target_size=224,
                 frozen_stages=3):
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
        self.backbone_model: Literal['efficientnet_v2_s',
                                     'mobile_net_v4'] = backbone_model
        self.train_data_loader = train_data_loader
        self.train_transforms: v2.Compose | None = train_transforms
        self.val_transforms: v2.Compose | None = val_transforms
        self.criterion: nn.Module = criterion
        self.backbone_lr: float = backbone_lr
        self.head_lr: float = head_lr
        self.target_size: int = target_size
        self.frozen_stages: int = frozen_stages
        self.backbone: nn.Module = get_model(
            model_name=backbone_model, with_weights=True, device=self.device)
        self.backbone.eval()
        with torch.no_grad():
            dummy_input = torch.zeros(
                1, 3, target_size, target_size).to(self.device)
            # We move backbone to device temporarily if needed, though usually CPU is fine for init
            output = self.backbone(dummy_input)
            self.in_features = output.shape[1]
        self.backbone.train()  # Set backbone back to train mode after feature extraction
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

        # Validation metrics to monitor performance with 0.05 threshold
        self.val_acc = BinaryAccuracy()
        self.val_f1 = BinaryF1Score()
        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

    def on_fit_start(self) -> None:
        adaptive_batch_norm(self.backbone, self.val_transforms,
                            self.train_data_loader, self.device)
        freeze_stages(self.backbone, frozen_stages=self.frozen_stages)

    def _step(self, batch):
        img, label = batch
        original_label = label.clone()  # Keep original labels for metrics
        label = label.view(-1, 1).float()
        smoothing_value = 0.08
        label = label * (1.0 - smoothing_value) + 0.5 * smoothing_value
        logits = self.forward(img)
        loss = self.criterion(logits, label)
        return loss, logits, original_label

    def training_step(self, batch, batch_idx):
        loss, logits, label = self._step(batch)
        self.log('train_loss', loss,  prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, label = self._step(batch)

        # Track accuracy with 0.05 threshold to monitor actual performance
        preds = (torch.sigmoid(logits) > 0.2).int().squeeze(1)
        self.val_acc.update(preds, label.int())
        self.val_precision.update(preds, label.int())
        self.val_recall.update(preds, label.int())
        self.val_f1.update(preds, label.int())

        self.log('val_loss', loss, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        self.log('val_acc', self.val_acc, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        self.log('val_precision', self.val_precision, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        self.log('val_recall', self.val_recall, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        self.log('val_f1', self.val_f1, on_epoch=True,
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
            'monitor': 'val_loss'
        }

    def test_step(self, batch, batch_idx):
        loss, logits, labels = self._step(batch)
        # liveniness should be 95% confident
        preds = (torch.sigmoid(logits) > 0.2).int().squeeze(1)

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
            print(f" Unexpected per‑class tensor shape: "
                  f"{rec_perclass.shape}")
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


def get_model(model_name: str,
              with_weights: bool = False,
              device: torch.device = torch.device('cpu')) -> nn.Module:

    model = None

    if model_name == 'efficientnet_v2_s':
        # 1. Load Torchvision model
        weights = models.EfficientNet_V2_S_Weights.DEFAULT if with_weights else None
        model = models.efficientnet_v2_s(weights=weights)

        # 2. Extract input features BEFORE replacing the head
        # EfficientNet classifier is Sequential: [0]=Dropout, [1]=Linear
        n_features = model.classifier[1].in_features

        # 3. Standardize: Remove head and attach num_features
        cast(nn.Module, model).classifier = nn.Identity()
        model.num_features = n_features

    elif model_name == 'mobile_net_v4':
        # 1. Load Timm model
        # num_classes=0 automatically removes the head and pools the features
        model = timm.create_model(
            'mobilenetv4_conv_medium.e500_r224_in1k',  # Use r224 for 224x224 input
            pretrained=with_weights,
            num_classes=0
        )
        # Timm models already have a .num_features attribute, so no extra work needed

    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    return model.to(device)
