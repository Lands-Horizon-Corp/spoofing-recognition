from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DepthwiseCDC(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, theta=0.7):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size,
                              padding=padding, groups=channels, bias=False)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        kernel_size = self.conv.kernel_size[0]
        center_idx = (kernel_size * kernel_size) // 2

        weight_diff = self.conv.weight.view(self.conv.out_channels, 1, -1)
        center_weights = weight_diff[:, :, center_idx].view(
            self.conv.out_channels, 1, 1, 1)

        out_diff = out_normal - \
            F.conv2d(x, center_weights, padding=0, groups=self.conv.groups)

        return self.theta * out_diff + (1 - self.theta) * out_normal


class DWCDC_SE_Block(nn.Module):
    def __init__(self, in_channels, out_channels, use_se=True, theta=0.7):
        super().__init__()

        self.dw_cdc = DepthwiseCDC(
            in_channels, kernel_size=3, padding=1, theta=theta)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False)

        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(out_channels)

    def forward(self, x):
        x = self.dw_cdc(x)
        x = self.pointwise(x)

        if self.use_se:
            x = self.se(x)

        return x


class FourierBranch(nn.Module):
    def __init__(self, in_channels=128):
        super().__init__()

        self.ft_predictor = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        predicted_spectrum = self.ft_predictor(x)
        return predicted_spectrum
