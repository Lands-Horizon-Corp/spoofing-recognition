from __future__ import annotations

import torch.nn as nn


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
