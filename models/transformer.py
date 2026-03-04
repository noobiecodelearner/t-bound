"""[transformer). — d_model-scalable Transformer for NLP tasks in [t-bound)."""

import math
import torch
import torch.nn as nn
from typing import Dict, Tuple


class ScalableTransformer(nn.Module):
    """
    Transformer encoder scaled by (num_layers, d_model) pairs.

    Scaling dimension: d_model (width).
    Architecture family: transformer.
    Forward expects dict with input_ids and attention_mask.
    Use count_parameters() to get N for logging.
    """

    def __init__(self, num_classes: int, vocab_size: int = 30522,
                 num_layers: int = 2, d_model: int = 128,
                 max_seq_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_classes = num_classes

        # derive nhead: largest power-of-2 divisor of d_model, capped at 8
        nhead = 1
        for h in [8, 4, 2, 1]:
            if d_model % h == 0:
                nhead = h
                break

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = self._sinusoidal_encoding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )
        self._init_weights()

    @staticmethod
    def _sinusoidal_encoding(max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        x = self.embedding(input_ids)
        pe = self.pos_encoding[:, :x.size(1), :].to(x.device)
        x = self.dropout(x + pe)

        # TransformerEncoder expects src_key_padding_mask: True = ignore
        pad_mask = (attention_mask == 0)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = self.norm(x)

        # mean pool over non-padding tokens
        mask_f = attention_mask.unsqueeze(-1).float()
        x = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-9)
        return self.classifier(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def estimate_flops(self, seq_len: int = 128) -> int:
        """Rough FLOPs estimate for one forward pass."""
        flops = 0
        # embedding lookup: free (index)
        # each transformer layer
        for _ in range(self.num_layers):
            d = self.d_model
            # self-attention: Q,K,V projections + attention + output
            flops += 4 * seq_len * d * d * 2
            # attention scores
            flops += seq_len * seq_len * d * 2
            # ffn
            flops += 2 * seq_len * d * d * 4 * 2
        # classifier
        flops += self.d_model * (self.d_model // 2) * 2
        flops += (self.d_model // 2) * self.num_classes * 2
        return int(flops)
