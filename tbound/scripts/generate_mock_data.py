"""
[generate_mock_data). — generate realistic mock runs.csv for SDK/API development.

Dayanch runs this on day one. Produces realistic mock data for all 9 datasets
so the API and dashboard can be built without waiting for real GPU results.

Usage:
    python scripts/generate_mock_data.py

Output:
    results/mock_runs.csv
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import uuid
from pathlib import Path
from datetime import datetime, timezone, timedelta
from utils.logger import RUNS_FIELDS
import csv

OUTPUT_PATH = Path("results/mock_runs.csv")

# Realistic scaling parameters per dataset
DATASET_CONFIGS = {
    "cifar10":   {"arch": "cnn",         "domain": "vision",   "nc": 10,  "alpha": 0.31, "a": 0.924, "ceil": 0.93},
    "cifar100":  {"arch": "cnn",         "domain": "vision",   "nc": 100, "alpha": 0.28, "a": 0.720, "ceil": 0.73},
    "stl10":     {"arch": "cnn",         "domain": "vision",   "nc": 10,  "alpha": 0.25, "a": 0.830, "ceil": 0.85},
    "yahoo":     {"arch": "transformer", "domain": "nlp",      "nc": 10,  "alpha": 0.22, "a": 0.750, "ceil": 0.76},
    "agnews":    {"arch": "transformer", "domain": "nlp",      "nc": 4,   "alpha": 0.24, "a": 0.940, "ceil": 0.95},
    "dbpedia":   {"arch": "transformer", "domain": "nlp",      "nc": 14,  "alpha": 0.20, "a": 0.990, "ceil": 0.99},
    "covertype": {"arch": "mlp",         "domain": "tabular",  "nc": 7,   "alpha": 0.29, "a": 0.970, "ceil": 0.97},
    "otto":      {"arch": "mlp",         "domain": "tabular",  "nc": 9,   "alpha": 0.26, "a": 0.840, "ceil": 0.85},
    "higgs":     {"arch": "mlp",         "domain": "tabular",  "nc": 2,   "alpha": 0.32, "a": 0.800, "ceil": 0.81},
}

# Parameter counts for each architecture scale
PARAM_COUNTS = {
    "cnn":         [45_000, 180_000, 400_000, 710_000, 1_600_000, 2_840_000],
    "transformer": [8_000,  32_000,  128_000, 260_000, 780_000,  2_050_000],
    "mlp":         [4_000,  16_000,  65_000,  260_000, 1_050_000, 4_200_000],
}

LR_VALUES    = [0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003]
D_FRACTIONS  = [0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
BATCH_VALUES = [32, 64, 128, 256, 512, 1024]


def power_law_acc(N, a, b, alpha, noise_std=0.008):
    """Accuracy = a - b·N^(-α) with noise."""
    rng = np.random.RandomState(abs(hash(str(N))) % (2**31))
    acc = a - b * (N ** (-alpha))
    acc += rng.normal(0, noise_std)
    return float(np.clip(acc, 0.0, 0.999))


def optimal_lr(N, base_lr=0.001, beta=0.18):
    """lr*(N) = base_lr · (N/N_ref)^(-β)"""
    N_ref = 300_000
    return float(base_lr * ((N / N_ref) ** (-beta)))


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    base_time = datetime.now(timezone.utc) - timedelta(days=30)
    rng = np.random.RandomState(42)

    for ds, cfg in DATASET_CONFIGS.items():
        arch   = cfg["arch"]
        domain = cfg["domain"]
        alpha  = cfg["alpha"]
        a_ceil = cfg["a"]
        nc     = cfg["nc"]
        params_list = PARAM_COUNTS[arch]
        b = rng.uniform(0.5, 2.0)  # scaling coefficient

        # ── n_d_lr_grid ───────────────────────────────────────────────────────
        for d_frac in D_FRACTIONS:
            d_scale = d_frac ** 0.25  # dataset size scaling factor
            for params in params_list:
                lr_star = optimal_lr(params)
                for lr in LR_VALUES:
                    # lr penalty: accuracy drops away from lr*
                    lr_ratio = lr / lr_star
                    lr_penalty = -0.3 * (np.log(lr_ratio) ** 2)

                    base_acc = power_law_acc(params, a_ceil, b, alpha)
                    val_acc  = float(np.clip(base_acc * d_scale + lr_penalty, 0.3, 0.999))
                    train_acc = float(np.clip(val_acc + rng.uniform(0.01, 0.08), 0, 1.0))

                    run_time = params * 0.0001 + rng.uniform(30, 120)
                    energy   = (run_time / 3600) * 250 / 1000

                    rows.append({
                        "run_id":              f"run_{uuid.uuid4().hex[:8]}",
                        "source":              "internal",
                        "project_id":          f"tbound_{ds}",
                        "timestamp":           (base_time + timedelta(
                                                 minutes=len(rows) * 20
                                               )).isoformat(),
                        "domain":              domain,
                        "dataset":             ds,
                        "architecture":        arch,
                        "num_classes":         nc,
                        "dataset_size":        int(45000 * d_frac),
                        "dataset_fraction":    d_frac,
                        "sweep_type":          "n_d_lr_grid",
                        "params":              params,
                        "learning_rate":       lr,
                        "batch_size":          128,
                        "weight_decay":        1e-4,
                        "optimizer":           "adam",
                        "num_steps":           10000,
                        "val_accuracy":        round(val_acc, 6),
                        "train_accuracy":      round(train_acc, 6),
                        "best_step":           rng.randint(7000, 10000),
                        "train_time_seconds":  round(run_time, 2),
                        "energy_kwh":          round(energy, 6),
                        "compute_flops":       params * 45000 * 6,
                        "generalization_gap":  round(train_acc - val_acc, 6),
                        "gen_warning":         "LOW" if train_acc - val_acc < 0.05
                                               else "MEDIUM" if train_acc - val_acc < 0.15
                                               else "HIGH",
                    })

        # ── batch_grid ────────────────────────────────────────────────────────
        # use N* ≈ middle params, lr* from optimal_lr
        n_star   = params_list[3]
        lr_n_star = optimal_lr(n_star)
        for d_frac in [0.25, 0.5, 1.0]:
            for batch in BATCH_VALUES:
                # batch penalty: sweet spot around 128-256
                batch_penalty = -0.01 * (np.log2(batch) - 7.5) ** 2
                base_acc = power_law_acc(n_star, a_ceil, b, alpha)
                val_acc  = float(np.clip(base_acc + batch_penalty, 0.3, 0.999))
                train_acc = float(np.clip(val_acc + rng.uniform(0.01, 0.05), 0, 1.0))
                run_time = n_star * 0.0001 + rng.uniform(30, 90)
                energy   = (run_time / 3600) * 250 / 1000

                rows.append({
                    "run_id":              f"run_{uuid.uuid4().hex[:8]}",
                    "source":              "internal",
                    "project_id":          f"tbound_{ds}",
                    "timestamp":           (base_time + timedelta(
                                             minutes=len(rows) * 20
                                           )).isoformat(),
                    "domain":              domain,
                    "dataset":             ds,
                    "architecture":        arch,
                    "num_classes":         nc,
                    "dataset_size":        int(45000 * d_frac),
                    "dataset_fraction":    d_frac,
                    "sweep_type":          "batch_grid",
                    "params":              n_star,
                    "learning_rate":       round(lr_n_star, 6),
                    "batch_size":          batch,
                    "weight_decay":        1e-4,
                    "optimizer":           "adam",
                    "num_steps":           10000,
                    "val_accuracy":        round(val_acc, 6),
                    "train_accuracy":      round(train_acc, 6),
                    "best_step":           rng.randint(7000, 10000),
                    "train_time_seconds":  round(run_time, 2),
                    "energy_kwh":          round(energy, 6),
                    "compute_flops":       n_star * 45000 * 6,
                    "generalization_gap":  round(train_acc - val_acc, 6),
                    "gen_warning":         "LOW",
                })

    # write CSV
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUNS_FIELDS)
        writer.writeheader()
        for row in rows:
            full_row = {k: row.get(k, "") for k in RUNS_FIELDS}
            writer.writerow(full_row)

    print(f"[t-bound] Generated {len(rows):,} mock runs → {OUTPUT_PATH}")
    print(f"  Datasets: {len(DATASET_CONFIGS)}")
    print(f"  n_d_lr_grid runs: {sum(1 for r in rows if r['sweep_type']=='n_d_lr_grid'):,}")
    print(f"  batch_grid runs:  {sum(1 for r in rows if r['sweep_type']=='batch_grid'):,}")
    print()
    print("  Give this file to Dayanch:")
    print(f"  cp {OUTPUT_PATH} <dayanch's api dir>/mock_data/runs.csv")


if __name__ == "__main__":
    main()
