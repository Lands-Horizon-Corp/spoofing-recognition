from __future__ import annotations

import lightning as L
from spoofdet.models.horizon_fas_net.blocks import CDCBlock
from spoofdet.models.horizon_fas_net.blocks import ConvBlock
from torch import nn


class Backbone(nn.Module):
    def __init__(self, cfg, num_classes=2, num_channels=3):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.num_channels = num_channels

        self.conv1 = ConvBlock(num_channels, cfg[0],
                               kernel_size=3, stride=2, padding=3)
        self.conv2_dw_cdc = CDCBlock(cfg[0], cfg[1], kernel_size=3,
                                     padding=1, stride=1, theta=0.7)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_dw_cdc(x)
        return x


class Head(nn.Module):
    def __init__(self, cfg, num_classes=2):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.fc = nn.Linear(cfg[-1], num_classes)

    def forward(self, x):
        pass


class HorizonFASNet(L.LightningModule):
    def __init__(self, cfg, num_classes=2, num_channels=3):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.num_channels = num_channels

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass


cfg_dict = {
    '1.8M_': [
        32,
        32,
        103,
        103,
        64,
        13,
        13,
        64,
        13,
        13,
        64,
        13,
        13,
        64,
        13,
        13,
        64,
        231,
        231,
        128,
        231,
        231,
        128,
        52,
        52,
        128,
        26,
        26,
        128,
        77,
        77,
        128,
        26,
        26,
        128,
        26,
        26,
        128,
        308,
        308,
        128,
        26,
        26,
        128,
        26,
        26,
        128,
        512,
        512,
    ]
}
