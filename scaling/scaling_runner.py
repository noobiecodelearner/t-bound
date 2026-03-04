"""[scaling_runner). — experiment orchestration for [t-bound).

Two sweep types:
    n_d_lr_grid:  joint N × D × lr grid (primary sweep)
    batch_grid:   batch sweep at fixed N* and lr*

Training budget: num_steps (not epochs). Epochs derived per run.
"""

import math
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from data.vision_loader   import load_vision
from data.nlp_loader      import load_nlp
from data.tabular_loader  import load_tabular
from models.cnn           import ScalableCNN
from models.transformer   import ScalableTransformer
from models.mlp           import ScalableMLP
from training.trainer     import Trainer
from training.energy      import estimate_energy_kwh, estimate_carbon_grams
from scaling.generalization_warning import GeneralizationWarningDetector
from utils.logger         import ExperimentLogger
from utils.seed           import set_seed


class ScalingRunner:
    """
    Orchestrates all scaling experiments for one dataset config.

    Usage:
        runner = ScalingRunner(config, device="cuda", verbose=True)
        runner.run()
    """

    def __init__(self, config: Dict, device: str = "cpu",
                 verbose: bool = False,
                 results_dir: str = "results"):
        self.cfg     = config
        self.device  = device
        self.verbose = verbose
        self.logger  = ExperimentLogger(
            results_dir=results_dir,
            source="internal",
            project_id=f"{config['domain']}_{config['dataset']}",
        )
        self.gen_detector = GeneralizationWarningDetector()

    # ── public entry point ────────────────────────────────────────────────────

    def run(self):
        sweep_type = self.cfg["sweep_type"]
        if sweep_type == "n_d_lr_grid":
            self._run_n_d_lr_grid()
        elif sweep_type == "batch_grid":
            self._run_batch_grid()
        else:
            raise ValueError(f"Unknown sweep_type: {sweep_type}. "
                             f"Use n_d_lr_grid or batch_grid.")

    # ── n_d_lr_grid ───────────────────────────────────────────────────────────

    def _run_n_d_lr_grid(self):
        """
        Primary sweep: 6 N × 6 D × 6 lr = 216 runs per dataset.
        Fixed batch = fixed_batch from config.
        Training budget = num_steps from config.
        """
        dataset_sizes  = self.cfg["dataset_sizes"]
        lr_values      = self.cfg["lr_sweep_values"]
        model_scales   = self.cfg["model_scales"]
        fixed_batch    = self.cfg["fixed_batch"]
        num_steps      = self.cfg["num_steps"]

        total = len(dataset_sizes) * len(model_scales) * len(lr_values)
        done  = 0

        print(f"\n[t-bound] n_d_lr_grid: {self.cfg['dataset']} "
              f"— {total} runs")

        for d_frac in dataset_sizes:
            for scale in model_scales:
                for lr in lr_values:
                    done += 1
                    if self.verbose:
                        print(f"  [{done}/{total}] D={d_frac:.2f} "
                              f"scale={scale} lr={lr:.5f}")
                    set_seed(42)
                    self._run_single(
                        scale=scale,
                        dataset_fraction=d_frac,
                        learning_rate=lr,
                        batch_size=fixed_batch,
                        num_steps=num_steps,
                        sweep_type="n_d_lr_grid",
                    )

    # ── batch_grid ────────────────────────────────────────────────────────────

    def _run_batch_grid(self):
        """
        Batch sweep: 6 batch × 3 D fractions = 18 runs per dataset.
        Fixed N* and lr* from config (filled in after n_d_lr_grid).
        """
        n_star     = self.cfg.get("fixed_n_star")
        lr_star    = self.cfg.get("fixed_lr_star")
        if n_star is None or lr_star is None:
            raise ValueError(
                "batch_grid requires fixed_n_star and fixed_lr_star in config. "
                "Run n_d_lr_grid first, then fill in these values."
            )

        batch_values  = self.cfg["batch_sweep_values"]
        dataset_sizes = self.cfg["dataset_sizes"]
        num_steps     = self.cfg["num_steps"]

        # find the model scale closest to n_star
        scale_for_n_star = self._find_scale_for_n_star(float(n_star))

        total = len(batch_values) * len(dataset_sizes)
        done  = 0

        print(f"\n[t-bound] batch_grid: {self.cfg['dataset']} "
              f"— {total} runs at N*≈{n_star}")

        for batch in batch_values:
            for d_frac in dataset_sizes:
                done += 1
                if self.verbose:
                    print(f"  [{done}/{total}] batch={batch} D={d_frac:.2f}")
                set_seed(42)
                self._run_single(
                    scale=scale_for_n_star,
                    dataset_fraction=d_frac,
                    learning_rate=float(lr_star),
                    batch_size=int(batch),
                    num_steps=num_steps,
                    sweep_type="batch_grid",
                )

    # ── single training run ───────────────────────────────────────────────────

    def _run_single(self, scale, dataset_fraction: float,
                    learning_rate: float, batch_size: int,
                    num_steps: int, sweep_type: str):
        """Build model, load data, train, log one row to runs.csv."""

        domain  = self.cfg["domain"]
        dataset = self.cfg["dataset"]

        # ── load data ────────────────────────────────────────────────────────
        train_loader, val_loader, _, n_train = self._get_loaders(
            dataset_fraction, batch_size
        )

        # ── build model ──────────────────────────────────────────────────────
        model, params, flops = self._build_model(scale, n_train)

        # ── train ─────────────────────────────────────────────────────────────
        trainer = Trainer(model, device=self.device,
                          gpu_wattage=self.cfg.get("gpu_wattage", 250))
        result  = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer_name=self.cfg["optimizer"],
            learning_rate=learning_rate,
            weight_decay=self.cfg.get("weight_decay", 1e-4),
            num_steps=num_steps,
            verbose=False,
        )

        # ── compute derived metrics ──────────────────────────────────────────
        energy  = estimate_energy_kwh(result["train_time_seconds"],
                                      self.cfg.get("gpu_wattage", 250))
        gap     = self.gen_detector.gap(result["final_train_accuracy"],
                                        result["best_val_accuracy"])
        warning = self.gen_detector.classify(result["final_train_accuracy"],
                                             result["best_val_accuracy"])

        # ── log ───────────────────────────────────────────────────────────────
        self.logger.log(
            domain=domain,
            dataset=dataset,
            architecture=self.cfg["architecture"],
            num_classes=self.cfg["num_classes"],
            dataset_size=n_train,
            dataset_fraction=dataset_fraction,
            sweep_type=sweep_type,
            params=params,
            learning_rate=learning_rate,
            batch_size=batch_size,
            weight_decay=self.cfg.get("weight_decay", 1e-4),
            optimizer=self.cfg["optimizer"],
            num_steps=num_steps,
            val_accuracy=round(result["best_val_accuracy"], 6),
            train_accuracy=round(result["final_train_accuracy"], 6),
            best_step=result["best_step"],
            train_time_seconds=round(result["train_time_seconds"], 2),
            energy_kwh=round(energy, 6),
            compute_flops=flops,
            generalization_gap=round(gap, 6),
            gen_warning=warning,
        )

        if self.verbose:
            print(f"    → params={params:,} val_acc={result['best_val_accuracy']:.4f} "
                  f"gap={warning}")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _get_loaders(self, dataset_fraction: float, batch_size: int):
        """Dispatch to correct domain loader."""
        domain  = self.cfg["domain"]
        dataset = self.cfg["dataset"]
        path    = self.cfg["data_path"]

        if domain == "vision":
            train, val, test, n = load_vision(
                dataset, path, dataset_fraction, batch_size
            )
            return train, val, test, n

        elif domain == "nlp":
            train, val, test, n = load_nlp(
                dataset, path,
                tokenizer_path=self.cfg.get("tokenizer_path",
                                            "data/raw/bert-base-uncased"),
                dataset_fraction=dataset_fraction,
                batch_size=batch_size,
            )
            return train, val, test, n

        elif domain == "tabular":
            train, val, test, n, input_dim = load_tabular(
                dataset, path, dataset_fraction, batch_size
            )
            self._last_input_dim = input_dim
            return train, val, test, n

        else:
            raise ValueError(f"Unknown domain: {domain}")

    def _build_model(self, scale, n_train: int):
        """Build model from scale descriptor. Returns (model, params, flops)."""
        domain = self.cfg["domain"]
        nc     = self.cfg["num_classes"]

        if domain == "vision":
            # scale is a float: width_multiplier
            model = ScalableCNN(num_classes=nc, width_multiplier=float(scale))
            params = model.count_parameters()
            flops  = model.estimate_flops()

        elif domain == "nlp":
            # scale is a list [num_layers, d_model]
            num_layers, d_model = int(scale[0]), int(scale[1])
            model = ScalableTransformer(
                num_classes=nc, num_layers=num_layers, d_model=d_model
            )
            params = model.count_parameters()
            flops  = model.estimate_flops()

        elif domain == "tabular":
            # scale is an int: hidden_size
            input_dim = getattr(self, "_last_input_dim",
                                self.cfg.get("input_dim", 54))
            model = ScalableMLP(
                input_dim=input_dim, num_classes=nc,
                hidden_size=int(scale)
            )
            params = model.count_parameters()
            flops  = model.estimate_flops(input_dim)

        else:
            raise ValueError(f"Unknown domain: {domain}")

        return model, params, flops

    def _find_scale_for_n_star(self, n_star: float):
        """Find the model scale whose parameter count is closest to n_star."""
        domain       = self.cfg["domain"]
        model_scales = self.cfg["model_scales"]
        nc           = self.cfg["num_classes"]
        best_scale   = model_scales[0]
        best_diff    = float("inf")

        for scale in model_scales:
            if domain == "vision":
                m = ScalableCNN(num_classes=nc, width_multiplier=float(scale))
            elif domain == "nlp":
                m = ScalableTransformer(
                    num_classes=nc, num_layers=int(scale[0]), d_model=int(scale[1])
                )
            elif domain == "tabular":
                input_dim = self.cfg.get("input_dim", 54)
                m = ScalableMLP(
                    input_dim=input_dim, num_classes=nc, hidden_size=int(scale)
                )
            else:
                raise ValueError(f"Unknown domain: {domain}")

            diff = abs(m.count_parameters() - n_star)
            if diff < best_diff:
                best_diff  = diff
                best_scale = scale

        return best_scale
