"""
[analyze_exponents). - CV analysis on fitted exponents + n_star_error validation.

Run this after each dataset's 216 runs complete and fit_surface has been run.
Run it BEFORE deciding whether to run the batch grid.

What it does:
    1. Computes coefficient of variation (std/mean) for each exponent
       across all datasets. Low CV = stable = safe to freeze globally.
    2. Prints a recommendation: freeze or fit per customer.
    3. Computes n_star_error for every (dataset, architecture_family) cell:
         true_n_star  = smallest model whose val_accuracy is within 1% of best
         n_star_error = abs(predicted - true) / true
       Appends n_star_error to results/n_star_errors.csv.
    4. Saves a summary to results/exponent_analysis.csv.

Usage:
    python3 scripts/analyze_exponents.py

Decision rules (CV):
    CV < 0.05  -> freeze globally. customers never need to sweep this.
    CV < 0.10  -> freeze exponent, fit constant from customer data only.
    CV >= 0.10 -> always fit per customer. prior is just a regularizer.

n_star_error thresholds:
    < 0.15        reliable. good to proceed.
    0.15 - 0.30   moderate. monitor.
    > 0.30        investigate immediately. power law may be misspecified.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path

from scaling.surface_fit import fit_model_size

RUNS_PATH        = Path("results/runs.csv")
FITS_PATH        = Path("results/fits.csv")
OUTPUT_PATH      = Path("results/exponent_analysis.csv")
NSTAR_ERR_PATH   = Path("results/n_star_errors.csv")


def cv(values):
    arr = np.array([v for v in values if v is not None and np.isfinite(v)])
    if len(arr) < 2:
        return None, None, None, arr if len(arr) else np.array([])
    return float(np.mean(arr)), float(np.std(arr)), float(np.std(arr) / np.mean(arr)), arr


def decision(coeff_var):
    if coeff_var is None:
        return "insufficient data"
    if coeff_var < 0.05:
        return "FREEZE GLOBALLY - customers never need to sweep this"
    if coeff_var < 0.10:
        return "FREEZE EXPONENT - fit constant c from customer data only"
    return "FIT PER CUSTOMER - prior is a regularizer, not a substitute"


def find_true_n_star(runs_df, dataset, arch, tolerance=0.01):
    """
    True N* = smallest model whose best val_accuracy (across lr) is within
    tolerance of the overall best val_accuracy for this (dataset, arch) cell.

    Returns (true_n_star, best_accuracy) or (None, None) if insufficient data.
    """
    cell = runs_df[
        (runs_df["dataset"] == dataset) &
        (runs_df["architecture"] == arch) &
        (runs_df["sweep_type"] == "n_d_lr_grid")
    ]
    if len(cell) < 3:
        return None, None

    best_per_n = cell.groupby("params")["val_accuracy"].max().reset_index()
    best_per_n = best_per_n.sort_values("params")

    best_acc = best_per_n["val_accuracy"].max()
    threshold = best_acc * (1 - tolerance)

    # smallest model within tolerance of best
    qualifying = best_per_n[best_per_n["val_accuracy"] >= threshold]
    if len(qualifying) == 0:
        return None, None

    true_n_star = int(qualifying["params"].min())
    return true_n_star, float(best_acc)


def compute_predicted_n_star(runs_df, dataset, arch, target_accuracy):
    """
    Fit the scaling curve on full-dataset runs and predict N* for target_accuracy.
    Returns predicted_n_star or None if fitting fails.
    """
    cell = runs_df[
        (runs_df["dataset"] == dataset) &
        (runs_df["architecture"] == arch) &
        (runs_df["sweep_type"] == "n_d_lr_grid") &
        (runs_df["dataset_fraction"] == 1.0)
    ]

    if len(cell) < 3:
        # fall back to all fractions if full-dataset runs are sparse
        cell = runs_df[
            (runs_df["dataset"] == dataset) &
            (runs_df["architecture"] == arch) &
            (runs_df["sweep_type"] == "n_d_lr_grid")
        ]

    if len(cell) < 3:
        return None

    best_per_n = cell.groupby("params")["val_accuracy"].max().reset_index()
    params_arr = best_per_n["params"].values.astype(float)
    acc_arr    = best_per_n["val_accuracy"].values.astype(float)

    try:
        fit = fit_model_size(params_arr, acc_arr)
        n_star = fit["optimal_n_fn"](target_accuracy)
        return n_star
    except Exception:
        return None


def compute_n_star_errors(runs_df):
    """
    For each (dataset, architecture_family) cell in runs_df, compute n_star_error.
    Returns list of dicts.
    """
    results = []

    # iterate all unique (dataset, architecture) combinations dynamically
    cells = runs_df[["dataset", "architecture", "domain"]].drop_duplicates()

    for _, row in cells.iterrows():
        dataset = row["dataset"]
        arch    = row["architecture"]
        domain  = row["domain"]

        true_n_star, best_acc = find_true_n_star(runs_df, dataset, arch)
        if true_n_star is None:
            continue

        # use best_acc * 0.98 as target (just below best, so curve must extrapolate slightly)
        target = best_acc * 0.98
        predicted_n_star = compute_predicted_n_star(runs_df, dataset, arch, target)

        if predicted_n_star is None:
            continue

        n_star_error = abs(predicted_n_star - true_n_star) / max(true_n_star, 1)

        if n_star_error < 0.15:
            verdict = "RELIABLE"
        elif n_star_error < 0.30:
            verdict = "MODERATE - monitor"
        else:
            verdict = "HIGH ERROR - investigate power law specification"

        results.append({
            "dataset":          dataset,
            "architecture":     arch,
            "domain":           domain,
            "true_n_star":      true_n_star,
            "predicted_n_star": int(round(predicted_n_star)),
            "target_accuracy":  round(target, 4),
            "best_accuracy":    round(best_acc, 4),
            "n_star_error":     round(n_star_error, 4),
            "verdict":          verdict,
        })

    return results


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

        exp_fits = fits[fits["exponent_type"] == exp]
        per_dataset = exp_fits.groupby("dataset")
        print(f"         per dataset:")
        for ds, grp in per_dataset:
            val = float(grp["exponent_value"].mean())
            ci_lo = grp["ci_lower_95"].dropna()
            ci_hi = grp["ci_upper_95"].dropna()
            if len(ci_lo) > 0 and len(ci_hi) > 0:
                lo = float(ci_lo.iloc[0])
                hi = float(ci_hi.iloc[0])
                print(f"           {ds:20s} {val:.3f}  CI=[{lo:.3f}, {hi:.3f}]")
            else:
                print(f"           {ds:20s} {val:.3f}  CI=n/a")
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
        print(f"[analyze_exponents] Exponent summary saved to {OUTPUT_PATH}")

    # -- n_star_error ----------------------------------------------------------
    if not RUNS_PATH.exists():
        print()
        print(f"[analyze_exponents] runs.csv not found - skipping n_star_error computation.")
        return

    runs_df = pd.read_csv(RUNS_PATH)
    print()
    print("=" * 72)
    print("N* ERROR ANALYSIS")
    print("=" * 72)
    print(f"  runs.csv: {len(runs_df)} rows across "
          f"{runs_df['dataset'].nunique()} datasets")
    print()

    nstar_results = compute_n_star_errors(runs_df)

    if not nstar_results:
        print("  No cells have enough runs for n_star_error yet.")
    else:
        all_reliable = True
        for r in nstar_results:
            err_pct = r["n_star_error"] * 100
            print(f"  {r['dataset']:20s} {r['architecture']:12s}  "
                  f"true_N*={r['true_n_star']:>8,}  "
                  f"pred_N*={r['predicted_n_star']:>8,}  "
                  f"error={err_pct:5.1f}%  {r['verdict']}")
            if r["n_star_error"] > 0.30:
                all_reliable = False

        print()
        errors = [r["n_star_error"] for r in nstar_results]
        print(f"  Mean n_star_error: {np.mean(errors):.3f} ({np.mean(errors)*100:.1f}%)")
        print(f"  Max  n_star_error: {np.max(errors):.3f} ({np.max(errors)*100:.1f}%)")

        if all_reliable:
            print()
            print("  VERDICT: All cells within acceptable error. "
                  "Curve fitting is reliable.")
        else:
            bad = [r["dataset"] for r in nstar_results if r["n_star_error"] > 0.30]
            print()
            print(f"  VERDICT: HIGH ERROR in {bad}. "
                  "Investigate power law specification before building prior.")

        # write n_star_errors.csv
        NSTAR_ERR_PATH.parent.mkdir(parents=True, exist_ok=True)
        nstar_df = pd.DataFrame(nstar_results)
        nstar_df.to_csv(NSTAR_ERR_PATH, index=False)
        print()
        print(f"[analyze_exponents] n_star_error results saved to {NSTAR_ERR_PATH}")


if __name__ == "__main__":
    main()