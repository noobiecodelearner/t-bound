"""[trainer). — domain-agnostic trainer with num_steps budget for [t-bound)."""

import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union

from training.metrics import compute_accuracy
from training.energy import estimate_energy_kwh, estimate_carbon_grams


# ── optimizer builder ─────────────────────────────────────────────────────────

def build_optimizer(model: nn.Module, optimizer_name: str,
                    learning_rate: float, weight_decay: float):
    name = optimizer_name.lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate,
                                weight_decay=weight_decay)
    elif name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
    elif name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate,
                               weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


# ── Trainer ───────────────────────────────────────────────────────────────────

class Trainer:
    """
    Domain-agnostic trainer. Handles vision, NLP, and tabular inputs.

    Training budget:
        num_steps mode (recommended):
            Set num_steps. Epochs are derived automatically.
            epochs = ceil(num_steps / steps_per_epoch)
            Stops mid-epoch when num_steps is reached.
            Every model gets the same optimization budget regardless
            of model size or dataset fraction.

        epochs mode (legacy):
            Set epochs directly. Use only for backward compatibility.
            Not recommended for scaling experiments — different model
            sizes and fractions get different effective budgets.

    Input handling:
        vision:   batch = (tensor_images, tensor_labels)
        nlp:      batch = (dict{input_ids, attention_mask}, tensor_labels)
        tabular:  batch = (tensor_features, tensor_labels)
    """

    def __init__(self, model: nn.Module, device: str = "cuda",
                 gpu_wattage: float = 250.0):
        self.model = model.to(device)
        self.device = device
        self.gpu_wattage = gpu_wattage
        self.criterion = nn.CrossEntropyLoss()

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer_name: str = "adam",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        num_steps: Optional[int] = None,
        epochs: Optional[int] = None,
        verbose: bool = False,
    ) -> Dict:
        """
        Train the model and return results dict.

        Either num_steps or epochs must be provided.
        num_steps is strongly preferred for scaling experiments.

        Returns:
            train_time_seconds: float
            best_val_accuracy:  float  — best val acc across all steps
            final_train_accuracy: float
            best_step:          int
            epoch_logs:         list of dicts per epoch
        """
        if num_steps is None and epochs is None:
            raise ValueError("Provide either num_steps or epochs.")

        # derive epochs from num_steps
        steps_per_epoch = len(train_loader)
        if num_steps is not None:
            epochs = math.ceil(num_steps / steps_per_epoch)
            target_steps = num_steps
        else:
            target_steps = epochs * steps_per_epoch

        optimizer = build_optimizer(self.model, optimizer_name,
                                    learning_rate, weight_decay)

        best_val_acc = 0.0
        best_step = 0
        epoch_logs = []
        global_step = 0
        start_time = time.time()

        for epoch in range(epochs):
            if global_step >= target_steps:
                break

            # ── train epoch ──────────────────────────────────────────────────
            self.model.train()
            train_correct = 0
            train_total = 0

            for batch in train_loader:
                if global_step >= target_steps:
                    break

                inputs, labels = self._unpack_batch(batch)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                logits = self.model(inputs)
                loss = self.criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                with torch.no_grad():
                    preds = logits.argmax(dim=1)
                    train_correct += preds.eq(labels).sum().item()
                    train_total += labels.size(0)

                global_step += 1

            train_acc = train_correct / max(train_total, 1)

            # ── val epoch ────────────────────────────────────────────────────
            val_acc = self._evaluate(val_loader)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_step = global_step

            log = {
                "epoch": epoch + 1,
                "step": global_step,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
            }
            epoch_logs.append(log)

            if verbose:
                print(f"  epoch {epoch+1:3d} | step {global_step:6d} "
                      f"| train_acc {train_acc:.4f} | val_acc {val_acc:.4f}")

        train_time = time.time() - start_time
        final_train_acc = epoch_logs[-1]["train_accuracy"] if epoch_logs else 0.0

        return {
            "train_time_seconds": train_time,
            "best_val_accuracy": best_val_acc,
            "final_train_accuracy": final_train_acc,
            "best_step": best_step,
            "epoch_logs": epoch_logs,
        }

    def _evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                inputs, labels = self._unpack_batch(batch)
                labels = labels.to(self.device)
                logits = self.model(inputs)
                preds = logits.argmax(dim=1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
        return correct / max(total, 1)

    def _unpack_batch(self, batch):
        """Handle vision, NLP, and tabular batch formats."""
        inputs, labels = batch
        if isinstance(inputs, dict):
            # NLP: dict of tensors
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        else:
            inputs = inputs.to(self.device)
        return inputs, labels
