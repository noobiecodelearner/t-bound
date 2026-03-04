"""[cnn). — width-scalable CNN for vision tasks in [t-bound)."""

import math
import torch
import torch.nn as nn
from typing import Tuple


class ScalableCNN(nn.Module):
    """
    Three-stage CNN scaled by width_multiplier on base channels [64, 128, 256].

    Scaling dimension: width.
    Architecture family: cnn.
    Use count_parameters() to get N for logging.
    Use estimate_flops() to get compute for logging.
    """

    BASE_CHANNELS = [64, 128, 256]

    def __init__(self, num_classes: int, width_multiplier: float = 1.0,
                 in_channels: int = 3, dropout: float = 0.3):
        super().__init__()
        self.width_multiplier = width_multiplier
        self.num_classes = num_classes

        c = [max(1, int(b * width_multiplier)) for b in self.BASE_CHANNELS]

        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, c[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(c[0], c[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout / 2),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(c[0], c[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(c[1], c[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout / 2),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(c[1], c[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(c[2], c[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[2]),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout2d(dropout),
        )
        self.classifier = nn.Linear(c[2], num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def estimate_flops(self, input_size: Tuple[int, int] = (32, 32)) -> int:
        """Rough FLOPs estimate for one forward pass."""
        h, w = input_size
        c = [max(1, int(b * self.width_multiplier)) for b in self.BASE_CHANNELS]
        flops = 0
        # stage 1: two conv layers, input halved by pooling
        flops += 2 * 3 * 3 * 3 * c[0] * h * w * 2
        h, w = h // 2, w // 2
        # stage 2
        flops += 2 * 3 * 3 * c[0] * c[1] * h * w * 2
        h, w = h // 2, w // 2
        # stage 3
        flops += 2 * 3 * 3 * c[1] * c[2] * h * w * 2
        # classifier
        flops += c[2] * self.num_classes
        return int(flops)
