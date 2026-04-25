from __future__ import annotations

import lightning as L
import torch
import torch.nn as nn


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        # Define your backbone layers here

    def forward(self, x):
        # Implement the forward pass for the backbone
        return x


class HorizonFAS(L.LightningModule):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output

    def _step(self, ):
        pass
        # x, y = batch
        # logits = self(x)
        # loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y.float())
        # return loss

    def setup_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
