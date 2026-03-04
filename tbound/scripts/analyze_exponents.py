"""
[analyze_exponents). — CV analysis on fitted exponents.

Run this after all n_d_lr_grid experiments complete and fits.csv is populated.
Run it BEFORE deciding whether to run the batch grid.

What it does:
    1. Computes coefficient of variation (std/mean) for each exponent
       across all datasets. Low CV = stable = safe to freeze globally.
    2. Prints a recommendation: freeze or fit per customer.
    3. Saves a summary to results/exponent_analysis.csv.

Usage:
    python3 scripts/analyze_exponents.py

Decision rules:
    CV < 0.05  -> freeze globally. customers never need to sweep this.
    CV < 0.10  -> freeze exponent, fit constant from customer data only.
    CV >= 0.10 -> always fit per customer. prior is just a regularizer.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path

FITS_PATH   = Path("results/fits.csv")
OUTPUT_PATH = Path("results/exponent_analysis.csv")


def cv(values):
    arr = np.array([v for v in values if v is not None and np.isfinite(v)])
    if len(arr) < 2:
        return None, None, None, arr if len(arr) else np.array([])
    return float(np.mean(arr)), float(np.std(arr)), float(np.std(arr) / np.mean(arr)), arr


def decision(coeff_var):
    if coeff_var is None:
        return "insufficient data"
    if coeff_var < 0.05:
        return "FREEZE GLOBALLY — customers never need to sweep this"
    if coeff_var < 0.10:
        return "FREEZE EXPONENT — fit constant c from customer data only"
    return "FIT PER CUSTOMER — prior is a regularizer, not a substitute"


def main():
    if not FITS_PATH.exists():
        print(f"[analyze_exponents] fits.csv not found at {FITS_PATH}")
        print("  Run experiments and fit_surface first.")
        return

    fits = pd.read_csv(FITS_PATH)
    print(f"[analyze_exponents] Loaded {len(fits)} rows from fits.csv")
    print(f"  Datasets:    {fits['dataset'].nunique()}")
    print(f"  Sweep types: {fits['sweep_type'].unique().tolist()}")
    print()

    exponents = ["alpha", "beta", "gamma", "delta"]
    descriptions = {
        "alpha": "model size exponent    Accuracy = a - b*N^(-alpha)",
        "beta":  "lr exponent            lr*(N) = c*N^(-beta)",
        "gamma": "batch exponent         batch*(N) = d*N^(gamma)",
        "delta": "dataset size exponent  Accuracy ~ D^(-delta)",
    }

    rows = []
    print("=" * 72)
    print(f"{'Exponent':<8} {'Mean':>6} {'Std':>6} {'CV':>6}  Decision")
    print("=" * 72)

    for exp in exponents:
        grp = fits[fits["exponent_type"] == exp]["exponent_value"]
        mean, std, coeff_var, arr = cv(grp.tolist())

        if mean is None:
            print(f"{exp:<8}  no data yet")
            continue

        dec = decision(coeff_var)
        print(f"{exp:<8} {mean:>6.3f} {std:>6.3f} {coeff_var:>6.3f}  {dec}")
        print(f"         {descriptions[exp]}")

        per_dataset = fits[fits["exponent_type"] == exp].groupby("dataset")["exponent_value"]
        print(f"         per dataset:")
        for ds, vals in per_dataset:
            v = vals.values
            print(f"           {ds:20s} {np.mean(v):.3f} +/- {np.std(v):.3f}")
        print()

        rows.append({
            "exponent":   exp,
            "mean":       round(mean, 4),
            "std":        round(std, 4),
            "cv":         round(coeff_var, 4),
            "n_datasets": len(arr),
            "min":        round(float(np.min(arr)), 4),
            "max":        round(float(np.max(arr)), 4),
            "decision":   dec,
        })

    print("=" * 72)

    if rows:
        gamma_row = next((r for r in rows if r["exponent"] == "gamma"), None)
        beta_row  = next((r for r in rows if r["exponent"] == "beta"),  None)

        print()
        print("BATCH GRID RECOMMENDATION:")
        if gamma_row and gamma_row["cv"] < 0.05:
            print(f"  gamma CV = {gamma_row['cv']:.3f} < 0.05")
            print(f"  -> SKIP batch grid entirely for customers.")
            print(f"  -> Use global gamma = {gamma_row['mean']:.3f} for all recommendations.")
            print(f"  -> Consider skipping remaining internal batch grids too.")
        elif gamma_row:
            print(f"  gamma CV = {gamma_row['cv']:.3f} >= 0.05")
            print(f"  -> Run batch grid. Customers need to sweep batch size.")

        print()
        print("LR SWEEP RECOMMENDATION:")
        if beta_row and beta_row["cv"] < 0.10:
            print(f"  beta CV = {beta_row['cv']:.3f} < 0.10")
            print(f"  -> Freeze beta = {beta_row['mean']:.3f}.")
            print(f"  -> Customers only need to fit constant c from 1-2 lr runs.")
        elif beta_row:
            print(f"  beta CV = {beta_row['cv']:.3f} >= 0.10")
            print(f"  -> Fit beta per customer. Cannot freeze.")

    if rows:
        df = pd.DataFrame(rows)
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_PATH, index=False)
        print()
        print(f"[analyze_exponents] Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()