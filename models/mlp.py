"""[mlp). — width-scalable MLP for tabular tasks in [t-bound)."""

import torch
import torch.nn as nn
from typing import Tuple


class ScalableMLP(nn.Module):
    """
    Three hidden layer MLP scaled by hidden_size (width).

    Scaling dimension: width.
    Architecture family: mlp.
    Use count_parameters() to get N for logging.
    """

    def __init__(self, input_dim: int, num_classes: int,
                 hidden_size: int = 256, dropout: float = 0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.network = nn.Sequential(
            # layer 1
            nn.Linear(input_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            # layer 2
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            # layer 3
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            # output
            nn.Linear(hidden_size // 2, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def estimate_flops(self, input_dim: int = None) -> int:
        """Rough FLOPs estimate for one forward pass."""
        flops = 0
        prev = input_dim or self.network[0].in_features
        for m in self.modules():
            if isinstance(m, nn.Linear):
                flops += prev * m.out_features * 2
                prev = m.out_features
        return int(flops)
