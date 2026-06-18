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
    prior/decomposition.json — additive decomposition: mu + arch_offset + domain_offset
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
from pathlib import Path
from utils.logger import PriorLogger


FITS_PATH       = Path("results/fits.csv")
RUNS_PATH       = Path("results/runs.csv")
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


def build_additive_decomposition(alpha_table: dict) -> dict:
    """
    Fit additive decomposition: alpha(arch, domain) = mu + arch_offset + domain_offset

    alpha_table: dict mapping (arch, domain) -> list of alpha values

    Returns dict with:
        mu:            global mean
        arch_offsets:  {arch: offset}
        domain_offsets: {domain: offset}
        predictions:   {(arch, domain): predicted_alpha}
        residuals:     {(arch, domain): observed - predicted}
        transfer:      {(arch, domain): bool (True if |residual| < 0.05)}
    """
    # compute mean alpha per (arch, domain) cell
    cell_means = {}
    for (arch, domain), values in alpha_table.items():
        arr = np.array([v for v in values if v is not None and np.isfinite(v)])
        if len(arr) > 0:
            cell_means[(arch, domain)] = float(np.mean(arr))

    if len(cell_means) == 0:
        return {}

    architectures = sorted(set(k[0] for k in cell_means))
    domains       = sorted(set(k[1] for k in cell_means))

    # global mean
    mu = float(np.mean(list(cell_means.values())))

    # iterative least-squares for offsets (2 iterations is enough for balance)
    arch_offsets   = {a: 0.0 for a in architectures}
    domain_offsets = {d: 0.0 for d in domains}

    for _ in range(10):
        # update arch offsets
        for arch in architectures:
            cells = [(arch, d) for d in domains if (arch, d) in cell_means]
            if cells:
                residuals = [
                    cell_means[c] - mu - domain_offsets[c[1]]
                    for c in cells
                ]
                arch_offsets[arch] = float(np.mean(residuals))

        # update domain offsets
        for domain in domains:
            cells = [(a, domain) for a in architectures if (a, domain) in cell_means]
            if cells:
                residuals = [
                    cell_means[c] - mu - arch_offsets[c[0]]
                    for c in cells
                ]
                domain_offsets[domain] = float(np.mean(residuals))

        # re-center: force offsets to sum to zero
        arch_mean   = np.mean(list(arch_offsets.values()))
        domain_mean = np.mean(list(domain_offsets.values()))
        mu += arch_mean + domain_mean
        arch_offsets   = {a: v - arch_mean   for a, v in arch_offsets.items()}
        domain_offsets = {d: v - domain_mean for d, v in domain_offsets.items()}

    # compute predictions and residuals
    predictions = {}
    residuals   = {}
    transfer    = {}

    # observed cells
    for (arch, domain), observed in cell_means.items():
        pred = mu + arch_offsets.get(arch, 0.0) + domain_offsets.get(domain, 0.0)
        res  = observed - pred
        predictions[(arch, domain)] = round(pred, 4)
        residuals[(arch, domain)]   = round(res, 4)
        transfer[(arch, domain)]    = bool(abs(res) < 0.05)

    # unobserved cells — predictions only, no residuals
    for arch in architectures:
        for domain in domains:
            if (arch, domain) not in cell_means:
                pred = mu + arch_offsets.get(arch, 0.0) + domain_offsets.get(domain, 0.0)
                predictions[(arch, domain)] = round(pred, 4)
                residuals[(arch, domain)]   = None   # not observed
                transfer[(arch, domain)]    = True   # assume transfer until proven otherwise

    # convert tuple keys to strings for JSON serialisation
    def str_key(d):
        return {f"{k[0]}__{k[1]}": v for k, v in d.items()}

    return {
        "mu":             round(mu, 4),
        "arch_offsets":   {k: round(v, 4) for k, v in arch_offsets.items()},
        "domain_offsets": {k: round(v, 4) for k, v in domain_offsets.items()},
        "cell_means":     {f"{k[0]}__{k[1]}": round(v, 4) for k, v in cell_means.items()},
        "predictions":    str_key(predictions),
        "residuals":      {f"{k[0]}__{k[1]}": v for k, v in residuals.items()},
        "transfer":       {f"{k[0]}__{k[1]}": v for k, v in transfer.items()},
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

    # ── load runs (for mean_source_dataset_size) ──────────────────────────────
    runs_df = None
    if RUNS_PATH.exists():
        runs_df = pd.read_csv(RUNS_PATH)
        print(f"[t-bound] Loaded {len(runs_df)} run rows from {RUNS_PATH}")
    else:
        print(f"[t-bound] runs.csv not found — mean_source_dataset_size will be null.")

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

    # collect alpha values per (arch, domain) for decomposition
    alpha_table = {}  # (arch, domain) -> [alpha values]

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

        # mean_source_dataset_size: average dataset_size of runs that fed this prior row
        mean_src_ds_size = None
        if runs_df is not None:
            contributing_runs = runs_df[
                (runs_df["architecture"] == arch) &
                (runs_df["domain"] == domain) &
                (runs_df["sweep_type"] == sweep)
            ]
            if len(contributing_runs) > 0 and "dataset_size" in contributing_runs.columns:
                sizes = pd.to_numeric(
                    contributing_runs["dataset_size"], errors="coerce"
                ).dropna()
                if len(sizes) > 0:
                    mean_src_ds_size = float(sizes.mean())

        prior_id = logger.log(
            architecture_family=arch,
            domain=domain,
            dataset_size_regime=regime,
            sweep_type=sweep,
            n_source_curves=len(alpha_vals),
            n_source_implementations=1,  # update when multiple impls added
            mean_source_dataset_size=mean_src_ds_size,
            **alpha_stats,
            **beta_stats,
            **gamma_stats,
            **delta_stats,
        )

        print(f"  {arch:12s} {domain:8s} {regime:8s} {sweep:15s} "
              f"→ α={alpha_stats.get('alpha_mean', 'N/A')} "
              f"({len(alpha_vals)} curves)")

        # accumulate for decomposition (model_size sweep only)
        if sweep == "model_size" and alpha_vals:
            key = (arch, domain)
            alpha_table.setdefault(key, [])
            alpha_table[key].extend(alpha_vals)

    print(f"\n[t-bound] prior.csv written to {prior_path}")

    # ── additive decomposition ─────────────────────────────────────────────────
    decomp_path = PRIOR_DIR / "decomposition.json"

    if len(alpha_table) >= 2:
        print("\n[t-bound] Building additive decomposition...")
        decomp = build_additive_decomposition(alpha_table)

        print(f"  mu = {decomp['mu']}")
        print(f"  arch_offsets:   {decomp['arch_offsets']}")
        print(f"  domain_offsets: {decomp['domain_offsets']}")
        print()

        for key, pred in decomp["predictions"].items():
            arch, domain = key.split("__")
            obs  = decomp["cell_means"].get(key, "—")
            res  = decomp["residuals"].get(key)
            tfr  = decomp["transfer"].get(key, True)
            obs_str = f"{obs:.3f}" if isinstance(obs, float) else obs
            res_str = f"{res:+.3f}" if res is not None else "  (unobserved)"
            tfr_str = "transfer=True" if tfr else "transfer=False (large residual)"
            print(f"  {arch:12s} × {domain:8s}  pred={pred:.3f}  "
                  f"obs={obs_str}  residual={res_str}  {tfr_str}")

        # update prior.csv with decomposition columns
        prior_df = pd.read_csv(prior_path)

        def get_decomp_alpha(row):
            key = f"{row['architecture_family']}__{row['domain']}"
            return decomp["predictions"].get(key)

        def get_decomp_residual(row):
            key = f"{row['architecture_family']}__{row['domain']}"
            return decomp["residuals"].get(key)

        def get_transfer(row):
            key = f"{row['architecture_family']}__{row['domain']}"
            return decomp["transfer"].get(key, True)

        prior_df["decomposition_alpha"]    = prior_df.apply(get_decomp_alpha,    axis=1)
        prior_df["decomposition_residual"] = prior_df.apply(get_decomp_residual, axis=1)
        prior_df["transfer"]               = prior_df.apply(get_transfer,         axis=1)
        prior_df.to_csv(prior_path, index=False)
        print(f"\n  decomposition columns written to {prior_path}")

        with open(decomp_path, "w") as f:
            json.dump(decomp, f, indent=2)
        print(f"  decomposition.json written to {decomp_path}")

    else:
        print(f"\n[t-bound] Not enough (arch, domain) cells for decomposition "
              f"(need ≥ 2, have {len(alpha_table)}). "
              f"Skipping — run more datasets first.")
        # write empty decomposition.json so prior_service.py doesn't crash
        with open(decomp_path, "w") as f:
            json.dump({}, f)

    print()
    print("[t-bound] Done.")
    print("  prior.csv and decomposition.json are ready for "
          "api/services/prior_service.py")


if __name__ == "__main__":
    main()