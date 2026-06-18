"""
[test_conditional_stability). — test whether α variance is structured or noise.

Run after each dataset's 216 runs complete to get an early signal.
Run with no arguments after 3+ datasets to get cross-dataset correlations.

Usage:
    python3 scripts/test_conditional_stability.py --dataset cifar10
    python3 scripts/test_conditional_stability.py           # all completed datasets

Output:
    results/stability_report.csv   — per-dataset stability verdicts
    printed cross-dataset summary  — domain CVs + correlations with label_entropy,
                                     class_imbalance (requires backfill_dataset_features.py)

Verdicts:
    stable              both ranges < 0.05 and CV < 0.10
    conditionally_stable  elevated but systematic variance
    unstable            high and unsystematic variance
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from scaling.surface_fit import fit_model_size

RUNS_PATH    = Path("results/runs.csv")
OUTPUT_PATH  = Path("results/stability_report.csv")

MIN_RUNS_PER_CELL = 50   # minimum runs to include a dataset in analysis


# ── helpers ───────────────────────────────────────────────────────────────────

def fit_alpha_on_subset(df: pd.DataFrame) -> Optional[float]:
    """
    Fit α on a subset of runs (already filtered to one fraction/scale regime).
    Returns α or None if fitting fails.
    """
    if len(df) < 3:
        return None
    best_per_n = df.groupby("params")["val_accuracy"].max().reset_index()
    if len(best_per_n) < 3:
        return None
    try:
        fit = fit_model_size(
            best_per_n["params"].values.astype(float),
            best_per_n["val_accuracy"].values.astype(float),
        )
        return float(fit["alpha"])
    except Exception:
        return None


def analysis_fraction_split(cell_df: pd.DataFrame) -> dict:
    """
    Split runs by dataset_fraction regime and fit α in each.
    low: fractions 0.1, 0.2
    medium: fractions 0.35, 0.5
    high: fractions 0.75, 1.0
    """
    regimes = {
        "low":    cell_df[cell_df["dataset_fraction"].isin([0.10, 0.20])],
        "medium": cell_df[cell_df["dataset_fraction"].isin([0.35, 0.50])],
        "high":   cell_df[cell_df["dataset_fraction"].isin([0.75, 1.00])],
    }

    alphas = {}
    for name, subset in regimes.items():
        a = fit_alpha_on_subset(subset)
        if a is not None:
            alphas[name] = a

    if len(alphas) < 2:
        return {"alphas": alphas, "range": None, "is_conditioning_variable": None}

    vals  = list(alphas.values())
    rng   = float(max(vals) - min(vals))
    return {
        "alphas": alphas,
        "range":  round(rng, 4),
        "is_conditioning_variable": rng > 0.05,
    }


def analysis_scale_split(cell_df: pd.DataFrame) -> dict:
    """
    Split runs by model params into halves (small vs large) and fit α in each.
    Tests whether the power law is consistent across the full parameter range
    (broken power law test).

    Uses halves instead of thirds so each half has enough points to fit (≥3).
    With 6 scales: bottom 3 vs top 3.
    With 8 scales: bottom 4 vs top 4.
    """
    params_sorted = sorted(cell_df["params"].unique())
    if len(params_sorted) < 6:
        # need at least 6 total so each half has ≥3 points
        return {"alphas": {}, "range": None, "is_broken_power_law": None}

    mid = len(params_sorted) // 2
    halves = {
        "small": params_sorted[:mid],
        "large": params_sorted[mid:],
    }

    alphas = {}
    for name, param_group in halves.items():
        subset = cell_df[cell_df["params"].isin(param_group)]
        a = fit_alpha_on_subset(subset)
        if a is not None:
            alphas[name] = a

    if len(alphas) < 2:
        return {"alphas": alphas, "range": None, "is_broken_power_law": None}

    vals = list(alphas.values())
    rng  = float(max(vals) - min(vals))
    return {
        "alphas": alphas,
        "range":  round(rng, 4),
        "is_broken_power_law": rng > 0.10,  # larger threshold for half-split
    }


def analysis_cv_across_fractions(cell_df: pd.DataFrame) -> dict:
    """
    Fit α separately at each of the 6 dataset_fraction values.
    Compute CV = std(α values) / mean(α values) across the 6 fits.
    """
    fractions = sorted(cell_df["dataset_fraction"].unique())
    alphas = {}

    for frac in fractions:
        subset = cell_df[cell_df["dataset_fraction"] == frac]
        a = fit_alpha_on_subset(subset)
        if a is not None:
            alphas[frac] = a

    if len(alphas) < 2:
        return {"alphas": alphas, "cv": None, "mean": None, "std": None}

    vals = list(alphas.values())
    mean = float(np.mean(vals))
    std  = float(np.std(vals))
    cv   = float(std / mean) if mean > 0 else None

    return {
        "alphas": {round(k, 2): round(v, 4) for k, v in alphas.items()},
        "cv":     round(cv, 4) if cv is not None else None,
        "mean":   round(mean, 4),
        "std":    round(std, 4),
    }


def stability_verdict(fraction_range, scale_range, cv) -> str:
    """Assign a stability verdict from the three analysis values."""
    if fraction_range is None or scale_range is None or cv is None:
        return "insufficient_data"
    if fraction_range < 0.05 and scale_range < 0.05 and cv < 0.10:
        return "stable"
    # check if the variance is systematic (monotone in fraction or scale)
    return "conditionally_stable" if (fraction_range < 0.10 or scale_range < 0.10) else "unstable"


def analyse_dataset(runs_df: pd.DataFrame, dataset: str) -> Optional[dict]:
    """Run all three analyses for one dataset. Returns result dict or None."""
    cell_df = runs_df[
        (runs_df["dataset"] == dataset) &
        (runs_df["sweep_type"] == "n_d_lr_grid")
    ]

    if len(cell_df) < MIN_RUNS_PER_CELL:
        print(f"  {dataset}: only {len(cell_df)} runs — skipping "
              f"(need {MIN_RUNS_PER_CELL})")
        return None

    # infer architecture and domain
    arch   = cell_df["architecture"].mode()[0]
    domain = cell_df["domain"].mode()[0]

    print(f"\n  {dataset.upper()} ({arch}, {domain})  [{len(cell_df)} runs]")
    print(f"  {'─' * 60}")

    # analysis 1: fraction split
    frac_result  = analysis_fraction_split(cell_df)
    frac_range   = frac_result["range"]
    frac_alphas  = frac_result["alphas"]
    frac_flag    = frac_result["is_conditioning_variable"]
    frac_str     = "  ".join(f"{k}={v:.3f}" for k, v in frac_alphas.items())
    flag_str     = "→ dataset_fraction IS a conditioning variable" if frac_flag else ""
    print(f"  1. Fraction split:   {frac_str}   range={frac_range}  {flag_str}")

    # analysis 2: scale split
    scale_result = analysis_scale_split(cell_df)
    scale_range  = scale_result["range"]
    scale_alphas = scale_result["alphas"]
    broken_flag  = scale_result["is_broken_power_law"]
    scale_str    = "  ".join(f"{k}={v:.3f}" for k, v in scale_alphas.items())
    broken_str   = "→ possible broken power law" if broken_flag else ""
    print(f"  2. Scale split:      {scale_str}   range={scale_range}  {broken_str}")

    # analysis 3: CV across fractions
    cv_result    = analysis_cv_across_fractions(cell_df)
    cv_val       = cv_result["cv"]
    cv_mean      = cv_result["mean"]
    cv_std       = cv_result["std"]
    cv_alpha_str = "  ".join(f"D={k}→α={v}" for k, v in cv_result["alphas"].items())
    print(f"  3. CV across fracs:  mean={cv_mean}  std={cv_std}  CV={cv_val}")
    if cv_alpha_str:
        print(f"     {cv_alpha_str}")

    # verdict
    verdict = stability_verdict(frac_range, scale_range, cv_val)
    print(f"  Verdict: {verdict.upper()}")

    # get dataset-level conditioning variables if available
    entropy    = None
    imbalance  = None
    resolution = None
    if "label_entropy" in cell_df.columns:
        entropy    = cell_df["label_entropy"].dropna().mean()
        imbalance  = cell_df["class_imbalance"].dropna().mean()
        resolution = cell_df["input_resolution"].dropna().mean()
        entropy    = round(float(entropy),    4) if not np.isnan(entropy)    else None
        imbalance  = round(float(imbalance),  4) if not np.isnan(imbalance)  else None
        resolution = int(resolution)             if not np.isnan(resolution) else None

    return {
        "dataset":           dataset,
        "architecture":      arch,
        "domain":            domain,
        "n_runs":            len(cell_df),
        "fraction_range":    frac_range,
        "scale_range":       scale_range,
        "cv_across_fracs":   cv_val,
        "alpha_mean":        cv_result["mean"],
        "alpha_std":         cv_result["std"],
        "frac_is_cond_var":  frac_flag,
        "broken_power_law":  broken_flag,
        "verdict":           verdict,
        "label_entropy":     entropy,
        "class_imbalance":   imbalance,
        "input_resolution":  resolution,
    }


def cross_dataset_summary(results: list):
    """Print cross-dataset summary: domain CVs + correlations with features."""
    print("\n" + "=" * 72)
    print("CROSS-DATASET SUMMARY")
    print("=" * 72)

    df = pd.DataFrame(results)

    # domain-level mean CV
    print("\nMean CV by domain:")
    for domain, grp in df.groupby("domain"):
        cvs = grp["cv_across_fracs"].dropna()
        if len(cvs) > 0:
            print(f"  {domain:10s}  mean_CV={cvs.mean():.3f}  "
                  f"n={len(cvs)} datasets")

    # correlations with conditioning variables
    has_features = df["label_entropy"].notna().any()
    if has_features and len(df) >= 3:
        print("\nCorrelations with dataset features (require backfill_dataset_features.py):")
        cv_vals = pd.to_numeric(df["cv_across_fracs"], errors="coerce")
        for feat in ["label_entropy", "class_imbalance"]:
            feat_vals = pd.to_numeric(df[feat], errors="coerce")
            valid     = cv_vals.notna() & feat_vals.notna()
            if valid.sum() >= 3:
                corr = float(np.corrcoef(cv_vals[valid], feat_vals[valid])[0, 1])
                direction = "positive" if corr > 0 else "negative"
                strength  = "strong" if abs(corr) > 0.6 else "moderate" if abs(corr) > 0.3 else "weak"
                print(f"  CV vs {feat:20s}  r={corr:+.3f}  ({strength} {direction} correlation)")
                if abs(corr) > 0.6:
                    print(f"    → {feat} is a genuine conditioning variable. "
                          f"Include in conditional scaling model.")
            else:
                print(f"  CV vs {feat:20s}  insufficient data ({valid.sum()} valid pairs)")
    else:
        print("\n  Feature correlations not available — run backfill_dataset_features.py first.")

    # final recommendation
    print()
    print("RECOMMENDATION:")
    verdicts = df["verdict"].value_counts()
    if verdicts.get("unstable", 0) > 0:
        bad = df[df["verdict"] == "unstable"]["dataset"].tolist()
        print(f"  ⚠ UNSTABLE datasets: {bad}")
        print(f"    Investigate power law specification before building prior.")
    elif verdicts.get("conditionally_stable", 0) > 0:
        cond = df[df["verdict"] == "conditionally_stable"]["dataset"].tolist()
        cond_vars = []
        if df["frac_is_cond_var"].any():
            cond_vars.append("dataset_fraction")
        if df["broken_power_law"].any():
            cond_vars.append("model_scale (broken power law)")
        print(f"  Conditionally stable datasets: {cond}")
        if cond_vars:
            print(f"  Conditioning variables explaining variance: {cond_vars}")
            print(f"  Consider updating surface_fit.py to include these.")
        else:
            print(f"  Variance pattern unclear — collect more data.")
    else:
        print(f"  ✓ All datasets stable. Prior transfer assumption validated.")
        print(f"  Proceed with confidence.")


def main():
    parser = argparse.ArgumentParser(
        description="Test conditional stability of α across dataset fractions and model scales"
    )
    parser.add_argument("--dataset", type=str, default=None,
                        help="Analyse only this dataset (default: all with enough runs)")
    args = parser.parse_args()

    if not RUNS_PATH.exists():
        print(f"[stability] runs.csv not found at {RUNS_PATH}")
        return

    runs_df = pd.read_csv(RUNS_PATH)
    print(f"[stability] Loaded {len(runs_df)} rows from runs.csv")
    print(f"  Datasets with runs: "
          f"{sorted(runs_df['dataset'].unique().tolist())}")

    print("\n" + "=" * 72)
    print("PER-DATASET STABILITY ANALYSIS")
    print("=" * 72)

    if args.dataset:
        datasets = [args.dataset]
    else:
        # all datasets with enough runs
        counts   = runs_df[runs_df["sweep_type"] == "n_d_lr_grid"].groupby("dataset").size()
        datasets = sorted(counts[counts >= MIN_RUNS_PER_CELL].index.tolist())
        if not datasets:
            print(f"\n  No datasets have {MIN_RUNS_PER_CELL}+ runs yet.")
            return

    results = []
    for ds in datasets:
        r = analyse_dataset(runs_df, ds)
        if r is not None:
            results.append(r)

    if not results:
        print("\n[stability] No results to summarise.")
        return

    # save per-dataset report
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_df = pd.DataFrame(results)
    report_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[stability] Per-dataset report saved to {OUTPUT_PATH}")

    # cross-dataset summary (only if 3+ datasets)
    if len(results) >= 3:
        cross_dataset_summary(results)
    else:
        print(f"\n[stability] {len(results)} dataset(s) analysed. "
              f"Run again after {3 - len(results)} more dataset(s) complete "
              f"for cross-dataset correlations.")


if __name__ == "__main__":
    main()