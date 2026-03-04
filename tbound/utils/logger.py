"""[logger). — experiment, fits, and prior logging for [t-bound)."""

import csv
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


# ── shared helpers ────────────────────────────────────────────────────────────

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _ensure_csv(path: Path, fieldnames: list) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()


def _append_row(path: Path, fieldnames: list, row: dict) -> None:
    _ensure_csv(path, fieldnames)
    # fill missing keys with empty string
    full_row = {k: row.get(k, "") for k in fieldnames}
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(full_row)


# ── ExperimentLogger — one row per training run → runs.csv ───────────────────

RUNS_FIELDS = [
    # identity
    "run_id", "source", "project_id", "timestamp",
    # task
    "domain", "dataset", "architecture", "num_classes",
    "dataset_size", "dataset_fraction", "full_dataset_size",
    # sweep
    "sweep_type",
    # variables
    "params", "learning_rate", "batch_size", "weight_decay",
    "optimizer", "num_steps",
    # results
    "val_accuracy", "train_accuracy", "best_step",
    "train_time_seconds",
    # efficiency
    "energy_kwh", "compute_flops",
    # quality
    "generalization_gap", "gen_warning",
]


class ExperimentLogger:
    """Logs one row per training run to runs.csv."""

    def __init__(self, results_dir: str = "results", source: str = "internal",
                 project_id: str = "default"):
        self.path = Path(results_dir) / "runs.csv"
        self.source = source
        self.project_id = project_id

    def log(self, **kwargs) -> str:
        run_id = _make_id("run")
        row = {
            "run_id": run_id,
            "source": self.source,
            "project_id": self.project_id,
            "timestamp": _utc_now(),
            **kwargs,
        }
        _append_row(self.path, RUNS_FIELDS, row)
        return run_id


# ── FitsLogger — one row per fitted exponent per dataset → fits.csv ──────────

FITS_FIELDS = [
    "fit_id", "dataset", "architecture_family", "domain",
    "dataset_size_regime", "sweep_type", "exponent_type",
    # model size fit: α, a, b
    # lr fit: β, c
    # batch fit: γ, d
    # dataset fit: δ
    "exponent_value", "param_a", "param_b", "param_c", "param_d",
    "r2", "mae", "aic",
    "n_runs_used",
    "optimal_value",      # N* | lr* | batch* depending on sweep_type
    "ci_lower_95", "ci_upper_95",
    "bootstrap_success",
    "timestamp",
]


class FitsLogger:
    """Logs one row per fitted exponent to fits.csv."""

    def __init__(self, results_dir: str = "results"):
        self.path = Path(results_dir) / "fits.csv"

    def log(self, **kwargs) -> str:
        fit_id = _make_id("fit")
        row = {
            "fit_id": fit_id,
            "timestamp": _utc_now(),
            **kwargs,
        }
        _append_row(self.path, FITS_FIELDS, row)
        return fit_id


# ── PriorLogger — one row per (family, domain, regime, sweep_type) → prior.csv

PRIOR_FIELDS = [
    "prior_id",
    "architecture_family",      # cnn | transformer | mlp
    "domain",                   # vision | nlp | tabular
    "dataset_size_regime",      # small | medium | large
    "sweep_type",               # model_size | lr | batch | dataset
    "n_source_curves",          # how many scaling curves contributed
    "n_source_implementations", # how many distinct architectures contributed
    # exponents — scale-invariant, transfer across customers
    "alpha_mean", "alpha_std", "alpha_min", "alpha_max",  # model size
    "beta_mean",  "beta_std",                              # lr
    "gamma_mean", "gamma_std",                             # batch
    "delta_mean", "delta_std",                             # dataset size
    # NOTE: a (ceiling) and b (coefficient) are NOT stored here
    # they are customer-specific and measurement-scale-dependent
    # additive decomposition — alpha(arch, domain) = mu + arch_offset + domain_offset
    "decomposition_alpha",    # predicted alpha from additive model
    "decomposition_residual", # observed - predicted (small = transfer reliable)
    "transfer",               # True = decomposition reliable, False = use own alpha only
    "last_updated",
]


class PriorLogger:
    """Writes aggregated prior entries to prior.csv."""

    def __init__(self, prior_dir: str = "prior"):
        self.path = Path(prior_dir) / "prior.csv"

    def log(self, **kwargs) -> str:
        prior_id = _make_id("prior")
        row = {
            "prior_id": prior_id,
            "last_updated": _utc_now(),
            **kwargs,
        }
        _append_row(self.path, PRIOR_FIELDS, row)
        return prior_id

    def load(self) -> list:
        """Load all prior entries as list of dicts."""
        if not self.path.exists():
            return []
        with open(self.path, newline="") as f:
            return list(csv.DictReader(f))