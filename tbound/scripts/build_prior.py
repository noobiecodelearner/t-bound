"""
[build_prior). — build prior.csv from fits.csv and literature values.

Run after all experiments complete (Week 8).

Usage:
    python scripts/build_prior.py

Reads:
    results/fits.csv         — your internal experiment results
    prior/literature_values.csv — manually curated from papers

Writes:
    prior/prior.csv          — aggregated prior for (family, domain, regime)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
from utils.logger import PriorLogger


FITS_PATH       = Path("results/fits.csv")
LITERATURE_PATH = Path("prior/literature_values.csv")
PRIOR_DIR       = Path("prior")


def dataset_size_regime(n_samples: int) -> str:
    if n_samples < 20_000:
        return "small"
    elif n_samples < 200_000:
        return "medium"
    else:
        return "large"


def aggregate_exponents(values: list, name: str) -> dict:
    """Compute statistics for a list of exponent values."""
    arr = np.array([v for v in values if v is not None and np.isfinite(v)])
    if len(arr) == 0:
        return {f"{name}_mean": None, f"{name}_std": None,
                f"{name}_min": None, f"{name}_max": None}
    return {
        f"{name}_mean": float(np.mean(arr)),
        f"{name}_std":  float(np.std(arr)),
        f"{name}_min":  float(np.min(arr)),
        f"{name}_max":  float(np.max(arr)),
    }


def main():
    PRIOR_DIR.mkdir(parents=True, exist_ok=True)

    # ── load fits ─────────────────────────────────────────────────────────────
    if not FITS_PATH.exists():
        print(f"[t-bound] fits.csv not found at {FITS_PATH}")
        print("  Run experiments first.")
        return

    fits_df = pd.read_csv(FITS_PATH)
    print(f"[t-bound] Loaded {len(fits_df)} fit rows from {FITS_PATH}")

    # ── load literature values ────────────────────────────────────────────────
    lit_df = None
    if LITERATURE_PATH.exists():
        lit_df = pd.read_csv(LITERATURE_PATH)
        print(f"[t-bound] Loaded {len(lit_df)} literature values")
    else:
        print(f"[t-bound] No literature_values.csv found at {LITERATURE_PATH}")
        print("  Proceeding with experimental data only.")

    # ── aggregate by (architecture_family, domain, dataset_size_regime, sweep_type)
    logger = PriorLogger(prior_dir=str(PRIOR_DIR))

    # overwrite existing prior.csv
    prior_path = PRIOR_DIR / "prior.csv"
    if prior_path.exists():
        prior_path.unlink()

    groups = fits_df.groupby(
        ["architecture_family", "domain", "dataset_size_regime", "sweep_type"]
    )

    for (arch, domain, regime, sweep), grp in groups:
        alpha_vals = grp[grp["exponent_type"] == "alpha"]["exponent_value"].tolist()
        beta_vals  = grp[grp["exponent_type"] == "beta"]["exponent_value"].tolist()
        gamma_vals = grp[grp["exponent_type"] == "gamma"]["exponent_value"].tolist()
        delta_vals = grp[grp["exponent_type"] == "delta"]["exponent_value"].tolist()

        # add literature values if available
        if lit_df is not None:
            lit_match = lit_df[
                (lit_df["architecture_family"] == arch) &
                (lit_df["domain"] == domain)
            ]
            if len(lit_match) > 0:
                alpha_vals += lit_match["alpha_mean"].dropna().tolist()

        alpha_stats = aggregate_exponents(alpha_vals, "alpha")
        beta_stats  = aggregate_exponents(beta_vals,  "beta")
        gamma_stats = aggregate_exponents(gamma_vals, "gamma")
        delta_stats = aggregate_exponents(delta_vals, "delta")

        n_datasets = len(grp["dataset"].unique()) if "dataset" in grp.columns else len(grp)

        prior_id = logger.log(
            architecture_family=arch,
            domain=domain,
            dataset_size_regime=regime,
            sweep_type=sweep,
            n_source_curves=len(alpha_vals),
            n_source_implementations=1,  # update when multiple impls added
            **alpha_stats,
            **beta_stats,
            **gamma_stats,
            **delta_stats,
        )

        print(f"  {arch:12s} {domain:8s} {regime:8s} {sweep:15s} "
              f"→ α={alpha_stats.get('alpha_mean', 'N/A')} "
              f"({len(alpha_vals)} curves)")

    print(f"\n[t-bound] prior.csv written to {prior_path}")
    print("  This file is now ready for Dayanch's API (api/services/prior_service.py)")


if __name__ == "__main__":
    main()
