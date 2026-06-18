"""[optimizer). — scaling law optimization for [t-bound).

Two optimization paths:
    optimize_accuracy(target_accuracy)     → N*, lr*, batch*, CI bands
    optimize_compute_budget(budget_hours)  → N*, D*, lr*, batch*, Chinchilla
"""

import numpy as np
from typing import Dict, Optional

from scaling.surface_fit import (
    fit_model_size, fit_lr_scaling, fit_batch_scaling,
    fit_nd_surface, solve_chinchilla_frontier,
)
from scaling.bootstrap import BootstrapUncertainty


class ScaleOptimizer:
    """
    Finds optimal training configuration from fitted scaling laws.

    Typical usage after n_d_lr_grid:
        optimizer = ScaleOptimizer(runs_df)
        rec = optimizer.optimize_accuracy(target_accuracy=0.85)

    Typical usage after batch_grid + n_d_lr_grid:
        rec = optimizer.optimize_accuracy(target_accuracy=0.85, include_batch=True)

    Chinchilla path (requires n_d_lr_grid with multiple D values):
        rec = optimizer.optimize_compute_budget(compute_budget_hours=10)
    """

    def __init__(self, runs_df, dataset: str = None,
                 n_bootstrap: int = 200, seed: int = 42):
        """
        Args:
            runs_df: pandas DataFrame loaded from results/runs.csv
            dataset: filter to one dataset if multiple in df
        """
        self.df        = runs_df
        self.dataset   = dataset
        self.bootstrap = BootstrapUncertainty(n_bootstrap=n_bootstrap, seed=seed)
        self._df        = self._filter(runs_df, dataset)

    def _filter(self, df, dataset):
        if dataset:
            return df[df["dataset"] == dataset].copy()
        return df.copy()

    # ── path A — accuracy target ──────────────────────────────────────────────

    def optimize_accuracy(
        self,
        target_accuracy: float,
        dataset_fraction: float = 1.0,
        include_batch: bool = False,
    ) -> Dict:
        """
        Given target accuracy τ, find minimum N* and optimal lr*, batch*.

        Uses model size scaling law: Accuracy*(N) = a - b·N^(-α)
        where Accuracy*(N) = max over lr of val_accuracy at each N.

        Args:
            target_accuracy:  τ, the accuracy you want to achieve
            dataset_fraction: which D fraction to use for fitting (default 1.0)
            include_batch:    whether to include batch_grid results for batch*

        Returns dict:
            n_star, lr_star, batch_star, expected_accuracy,
            ci_lower, ci_upper, confidence,
            compute_saved_fraction, carbon_saved_g,
            alpha, beta, gamma, fit_r2
        """
        grid_df = self._df[
            (self._df["sweep_type"] == "n_d_lr_grid") &
            (self._df["dataset_fraction"] == dataset_fraction)
        ]

        if len(grid_df) < 3:
            raise ValueError(
                f"Not enough runs for dataset_fraction={dataset_fraction}. "
                f"Found {len(grid_df)}, need at least 3."
            )

        # best accuracy per N (optimized over lr)
        best_per_n = (
            grid_df.groupby("params")["val_accuracy"]
            .max()
            .reset_index()
        )
        params_arr = best_per_n["params"].values.astype(float)
        acc_arr    = best_per_n["val_accuracy"].values.astype(float)

        # fit model size scaling
        fit = fit_model_size(params_arr, acc_arr)

        # find N*
        n_star = fit["optimal_n_fn"](target_accuracy)
        if n_star is None:
            return {
                "error": f"Target accuracy {target_accuracy} is unreachable. "
                         f"Max achievable: {fit['a']:.4f}",
                "max_achievable_accuracy": fit["a"],
            }

        # fit lr scaling: lr*(N) = c·N^(-β)
        best_lr_per_n = (
            grid_df.groupby("params")
            .apply(lambda g: g.loc[g["val_accuracy"].idxmax(), "learning_rate"])
            .reset_index()
        )
        best_lr_per_n.columns = ["params", "lr_star"]
        lr_fit = fit_lr_scaling(
            best_lr_per_n["params"].values.astype(float),
            best_lr_per_n["lr_star"].values.astype(float),
        )
        lr_star = lr_fit["optimal_lr_fn"](n_star)

        # batch* from batch_grid if available
        batch_star = 128  # default
        gamma_val  = None
        if include_batch:
            batch_df = self._df[self._df["sweep_type"] == "batch_grid"]
            if len(batch_df) >= 2:
                # use best batch at full D
                best_batch_df = batch_df[
                    batch_df["dataset_fraction"] == dataset_fraction
                ]
                if len(best_batch_df) >= 2:
                    best_b = best_batch_df.loc[
                        best_batch_df["val_accuracy"].idxmax(), "batch_size"
                    ]
                    batch_star = int(best_b)

        # bootstrap CI on N*
        def _fit_fn(x, y):
            return fit_model_size(x, y)

        def _opt_fn(result):
            return result["optimal_n_fn"](target_accuracy)

        ci = self.bootstrap.compute_ci(params_arr, acc_arr, _fit_fn, _opt_fn)

        # compute savings vs domain-appropriate baseline
        # NLP:     Chinchilla D/20  (derived for transformers on text)
        # Vision:  D/10             (empirically, vision needs more params per sample)
        # Tabular: max params in grid (Chinchilla ratio doesn't apply to MLPs)
        full_d  = self._df["dataset_size"].max()
        domain  = self._df["domain"].iloc[0] if "domain" in self._df.columns else "unknown"

        if domain == "nlp":
            baseline_n    = full_d / 20.0
            baseline_type = "chinchilla_20"
        elif domain == "vision":
            baseline_n    = full_d / 10.0
            baseline_type = "chinchilla_10"
        else:
            # tabular: use largest model in grid as baseline
            baseline_n    = float(self._df["params"].max())
            baseline_type = "largest_in_grid"

        compute_saved = max(0.0, 1.0 - (n_star / baseline_n)) if baseline_n > 0 else 0.0
        # rough carbon: proportional to compute
        carbon_per_step = 0.01  # grams, rough estimate
        carbon_saved = compute_saved * n_star * carbon_per_step

        return {
            "n_star":                   int(round(n_star)),
            "lr_star":                  round(lr_star, 6),
            "batch_star":               batch_star,
            "expected_accuracy":        round(float(fit["predict_fn"](n_star)), 4),
            "target_accuracy":          target_accuracy,
            "ci_lower":                 ci.get("ci_lower"),
            "ci_upper":                 ci.get("ci_upper"),
            "ci_success":               ci.get("success", False),
            "confidence":               self._confidence_level(len(grid_df)),
            "compute_saved_fraction":   round(compute_saved, 4),
            "baseline_n":               int(round(baseline_n)),
            "baseline_type":            baseline_type,
            "carbon_saved_g":           round(carbon_saved, 2),
            "alpha":                    round(fit["alpha"], 4),
            "a":                        round(fit["a"], 4),
            "b":                        round(fit["b"], 4),
            "beta":                     round(lr_fit["beta"], 4),
            "fit_r2":                   round(fit["r2"], 4),
            "n_runs_used":              len(grid_df),
        }

    # ── path B — compute budget (Chinchilla) ──────────────────────────────────

    def optimize_compute_budget(
        self,
        compute_budget_hours: float,
        gpu_wattage: float = 250.0,
        flops_per_param_per_sample: float = 6.0,
    ) -> Dict:
        """
        Given compute budget C (GPU hours), find optimal (N*, D*) via
        the Chinchilla frontier on the fitted (N,D) surface.

        Requires n_d_lr_grid results with multiple D values.

        Args:
            compute_budget_hours:       total GPU hours available
            gpu_wattage:                GPU power in watts (for energy estimate)
            flops_per_param_per_sample: FLOPs per param per training sample

        Returns dict:
            n_star, d_star, lr_star, batch_star, expected_accuracy,
            compute_budget_flops, energy_kwh, carbon_g
        """
        grid_df = self._df[self._df["sweep_type"] == "n_d_lr_grid"]

        if len(grid_df) < 4:
            raise ValueError(
                f"Need at least 4 runs for Chinchilla surface fit. "
                f"Found {len(grid_df)}."
            )

        # best accuracy per (N, D) optimized over lr
        best_per_nd = (
            grid_df.groupby(["params", "dataset_size"])["val_accuracy"]
            .max()
            .reset_index()
        )

        params_arr  = best_per_nd["params"].values.astype(float)
        dsize_arr   = best_per_nd["dataset_size"].values.astype(float)
        acc_arr     = best_per_nd["val_accuracy"].values.astype(float)

        # fit (N,D) surface
        surface = fit_nd_surface(params_arr, dsize_arr, acc_arr)

        # convert compute budget to FLOPs
        # 1 GPU hour = 3600 seconds
        # FLOPs/sec for A100-class GPU ≈ 312e12
        # Adjust gpu_tflops to match your hardware
        gpu_tflops = 312.0  # A100 FP32 TFLOPs/s
        compute_budget_flops = (
            compute_budget_hours * 3600 * gpu_tflops * 1e12
        )

        # solve frontier
        frontier = solve_chinchilla_frontier(
            surface=surface,
            compute_budget_flops=compute_budget_flops,
            flops_per_param_per_sample=flops_per_param_per_sample,
        )

        n_star = frontier["n_star"]
        d_star = frontier["d_star"]

        # get lr* from lr scaling fit
        best_lr_per_n = (
            grid_df.groupby("params")
            .apply(lambda g: g.loc[g["val_accuracy"].idxmax(), "learning_rate"])
            .reset_index()
        )
        best_lr_per_n.columns = ["params", "lr_star"]
        lr_fit  = fit_lr_scaling(
            best_lr_per_n["params"].values.astype(float),
            best_lr_per_n["lr_star"].values.astype(float),
        )
        lr_star = lr_fit["optimal_lr_fn"](n_star)

        # energy
        energy_kwh = (compute_budget_hours * gpu_wattage) / 1000.0
        carbon_g   = energy_kwh * 475.0

        return {
            "n_star":               int(round(n_star)),
            "d_star":               int(round(d_star)),
            "lr_star":              round(lr_star, 6),
            "batch_star":           128,  # default, override with batch_grid result
            "expected_accuracy":    round(frontier["expected_accuracy"], 4),
            "compute_budget_hours": compute_budget_hours,
            "compute_budget_flops": compute_budget_flops,
            "energy_kwh":           round(energy_kwh, 4),
            "carbon_g":             round(carbon_g, 2),
            "alpha":                round(surface["alpha"], 4),
            "delta":                round(surface["delta"], 4),
            "surface_r2":           round(surface["r2"], 4),
            "n_runs_used":          len(grid_df),
        }

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _confidence_level(n_runs: int) -> str:
        if n_runs == 0:
            return "very_low"
        elif n_runs < 4:
            return "low"
        elif n_runs < 7:
            return "medium"
        else:
            return "high"