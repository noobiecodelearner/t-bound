"""
[generate_val_splits). — generate and save deterministic val splits.

Run this ONCE before any experiments. Commit the output to git.
Every loader reads from these files. Never regenerates inline.

Usage:
    python scripts/generate_val_splits.py

Output:
    data/val_splits/<dataset>_seed42.npz for all 9 datasets
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("data/val_splits")
SEED = 42

# Dataset sizes (total training samples before split)
# Val fraction is 10% of total training set, minimum 1000 samples
DATASET_CONFIGS = {
    # vision
    "cifar10":   {"n_total": 50000, "val_fraction": 0.10},
    "cifar100":  {"n_total": 50000, "val_fraction": 0.10},
    "stl10":     {"n_total": 5000,  "val_fraction": 0.20},  # STL-10 has only 5K train
    # nlp
    "yahoo":     {"n_total": 1400000, "val_fraction": 0.005},  # 7K val from 1.4M
    "agnews":    {"n_total": 120000,  "val_fraction": 0.05},   # 6K val
    "dbpedia":   {"n_total": 560000,  "val_fraction": 0.01},   # 5.6K val
    # tabular
    "covertype": {"n_total": 581012, "val_fraction": 0.02},    # ~11.6K val
    "mnist_tabular": {"n_total": 70000,  "val_fraction": 0.10},   # 7K val
    "higgs":     {"n_total": 1000000,"val_fraction": 0.01},    # 10K val (from 1M subset)
}


def generate_split(dataset_name: str, n_total: int,
                   val_fraction: float, seed: int = 42):
    """
    Generate a deterministic train/val split.

    Strategy:
        1. Shuffle all indices with fixed seed
        2. First n_val indices → val set
        3. Remaining indices → train pool
        4. dataset_fraction is applied to train pool at load time

    This ensures the val set is always identical regardless of
    what dataset_fraction is used in experiments.
    """
    rng = np.random.RandomState(seed)
    all_idx = rng.permutation(n_total)
    n_val = max(1000, int(n_total * val_fraction))
    val_idx   = all_idx[:n_val]
    train_idx = all_idx[n_val:]

    return train_idx, val_idx


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[t-bound] Generating val splits...")
    print(f"  Output directory: {OUTPUT_DIR.resolve()}")
    print(f"  Seed: {SEED}")
    print()

    for dataset_name, cfg in DATASET_CONFIGS.items():
        out_path = OUTPUT_DIR / f"{dataset_name}_seed{SEED}.npz"

        if out_path.exists():
            print(f"  {dataset_name:12s}  SKIPPED (already exists) → {out_path.name}")
            continue

        train_idx, val_idx = generate_split(
            dataset_name, cfg["n_total"], cfg["val_fraction"], SEED
        )

        np.savez(
            out_path,
            train_idx=train_idx,
            val_idx=val_idx,
            seed=SEED,
            dataset=dataset_name,
            n_total=cfg["n_total"],
            n_val=len(val_idx),
        )

        print(f"  {dataset_name:12s}  total={cfg['n_total']:8,}  "
              f"train_pool={len(train_idx):8,}  val={len(val_idx):6,}  "
              f"→ {out_path.name}")

    print()
    print("[t-bound] Done. Commit data/val_splits/ to git.")
    print("  These files must NEVER be regenerated mid-experiment.")
    print()

    # verification: confirm splits are reproducible
    print("[t-bound] Verification (first 5 val indices per dataset):")
    for dataset_name in DATASET_CONFIGS:
        data = np.load(OUTPUT_DIR / f"{dataset_name}_seed{SEED}.npz")
        print(f"  {dataset_name:12s}  val[:5] = {data['val_idx'][:5].tolist()}")


if __name__ == "__main__":
    main()