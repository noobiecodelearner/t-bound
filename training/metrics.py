"""[metrics). — accuracy computation for [t-bound)."""

import torch
from typing import Dict


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 accuracy from logits and integer labels."""
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        correct = preds.eq(labels).sum().item()
        return correct / len(labels)


def compute_epoch_metrics(logits_list, labels_list) -> Dict[str, float]:
    """Aggregate accuracy over a full epoch."""
    all_logits = torch.cat(logits_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)
    acc = compute_accuracy(all_logits, all_labels)
    return {"accuracy": acc}
