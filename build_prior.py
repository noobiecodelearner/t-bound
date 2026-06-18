"""
[backfill_dataset_features). — add conditioning variables to runs.csv.

Adds three columns to runs.csv:
    label_entropy      Shannon entropy of class distribution (nats)
    class_imbalance    max_class_freq / min_class_freq
    input_resolution   H*W for vision | avg tokens for NLP | n_features for tabular

Run incrementally after each dataset's 216 runs complete.
Only fills rows where the three columns are currently null — safe to re-run.

Usage:
    python3 scripts/backfill_dataset_features.py              # all datasets
    python3 scripts/backfill_dataset_features.py --dataset cifar10
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

RUNS_PATH = Path("results/runs.csv")

# ── hard-coded dataset features ───────────────────────────────────────────────
# label_entropy: computed from official class distributions
# class_imbalance: max_count / min_count (1.0 = perfectly balanced)
# input_resolution: H*W | avg_tokens | n_features

def _entropy(counts):
    """Shannon entropy in nats from raw class counts."""
    counts = np.array(counts, dtype=float)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))

def _imbalance(counts):
    counts = np.array(counts, dtype=float)
    return float(counts.max() / counts.min())

# Vision — all balanced (equal samples per class)
_CIFAR10_COUNTS  = [5000] * 10
_CIFAR100_COUNTS = [500]  * 100
_STL10_COUNTS    = [500]  * 10   # 5000 train samples, 10 classes

# NLP — from official dataset statistics
# AG News: 4 classes, 30000 each
_AGNEWS_COUNTS   = [30000, 30000, 30000, 30000]
# Yahoo Answers: 10 classes, 140000 each (training set)
_YAHOO_COUNTS    = [140000] * 10
# DBpedia: 14 classes, 40000 each
_DBPEDIA_COUNTS  = [40000] * 14

# Tabular — from official dataset statistics
# Covertype: 7 classes, highly imbalanced
_COVERTYPE_COUNTS = [211840, 283301, 35754, 2747, 9493, 17367, 20510]
# MNIST (tabular): 10 classes, roughly balanced
_MNIST_TABULAR_COUNTS = [6903, 7877, 6990, 7141, 6824, 6313, 6876, 7293, 6825, 5958]
# HIGGS: 2 classes, roughly balanced (from 1M subset)
_HIGGS_COUNTS = [529000, 471000]  # approximate from published stats

DATASET_FEATURES = {
    # vision
    "cifar10": {
        "label_entropy":     _entropy(_CIFAR10_COUNTS),
        "class_imbalance":   _imbalance(_CIFAR10_COUNTS),
        "input_resolution":  32 * 32,        # 1024
    },
    "cifar100": {
        "label_entropy":     _entropy(_CIFAR100_COUNTS),
        "class_imbalance":   _imbalance(_CIFAR100_COUNTS),
        "input_resolution":  32 * 32,        # 1024
    },
    "stl10": {
        "label_entropy":     _entropy(_STL10_COUNTS),
        "class_imbalance":   _imbalance(_STL10_COUNTS),
        "input_resolution":  96 * 96,        # 9216
    },
    # nlp — avg token counts from official benchmarks (BERT tokenizer)
    "agnews": {
        "label_entropy":     _entropy(_AGNEWS_COUNTS),
        "class_imbalance":   _imbalance(_AGNEWS_COUNTS),
        "input_resolution":  38,             # avg tokens (title + description)
    },
    "yahoo": {
        "label_entropy":     _entropy(_YAHOO_COUNTS),
        "class_imbalance":   _imbalance(_YAHOO_COUNTS),
        "input_resolution":  87,             # avg tokens (question + answer)
    },
    "dbpedia": {
        "label_entropy":     _entropy(_DBPEDIA_COUNTS),
        "class_imbalance":   _imbalance(_DBPEDIA_COUNTS),
        "input_resolution":  52,             # avg tokens (title + abstract)
    },
    # tabular — input_resolution = number of features
    "covertype": {
        "label_entropy":     _entropy(_COVERTYPE_COUNTS),
        "class_imbalance":   _imbalance(_COVERTYPE_COUNTS),
        "input_resolution":  54,
    },
    "mnist_tabular": {
        "label_entropy":     _entropy(_MNIST_TABULAR_COUNTS),
        "class_imbalance":   _imbalance(_MNIST_TABULAR_COUNTS),
        "input_resolution":  784,
    },
    "higgs": {
        "label_entropy":     _entropy(_HIGGS_COUNTS),
        "class_imbalance":   _imbalance(_HIGGS_COUNTS),
        "input_resolution":  28,
    },
}


def backfill(dataset: str = None):
    if not RUNS_PATH.exists():
        print(f"[backfill] runs.csv not found at {RUNS_PATH}")
        return

    df = pd.read_csv(RUNS_PATH)
    original_len = len(df)

    # add columns if missing
    for col in ["label_entropy", "class_imbalance", "input_resolution"]:
        if col not in df.columns:
            df[col] = np.nan

    datasets_to_fill = [dataset] if dataset else list(DATASET_FEATURES.keys())
    total_filled = 0

    for ds in datasets_to_fill:
        if ds not in DATASET_FEATURES:
            print(f"[backfill] WARNING: {ds} not in DATASET_FEATURES — skipping.")
            continue

        features = DATASET_FEATURES[ds]

        # only fill rows where dataset matches AND columns are null
        mask = (df["dataset"] == ds) & df["label_entropy"].isna()
        n_rows = mask.sum()

        if n_rows == 0:
            already = (df["dataset"] == ds).sum()
            if already > 0:
                print(f"  {ds:20s}  already filled ({already} rows)")
            else:
                print(f"  {ds:20s}  no rows in runs.csv yet")
            continue

        df.loc[mask, "label_entropy"]    = round(features["label_entropy"],    4)
        df.loc[mask, "class_imbalance"]  = round(features["class_imbalance"],  4)
        df.loc[mask, "input_resolution"] = features["input_resolution"]

        total_filled += n_rows
        print(f"  {ds:20s}  filled {n_rows} rows  "
              f"entropy={features['label_entropy']:.3f}  "
              f"imbalance={features['class_imbalance']:.2f}  "
              f"resolution={features['input_resolution']}")

    if total_filled > 0:
        df.to_csv(RUNS_PATH, index=False)
        print(f"\n[backfill] Wrote {total_filled} new rows to {RUNS_PATH}")
    else:
        print(f"\n[backfill] Nothing to fill.")

    # summary: how many rows still have nulls
    null_count = df["label_entropy"].isna().sum()
    if null_count > 0:
        missing_ds = df[df["label_entropy"].isna()]["dataset"].unique().tolist()
        print(f"[backfill] {null_count} rows still null "
              f"(datasets not yet finished: {missing_ds})")
    else:
        print(f"[backfill] All {len(df)} rows have features. No nulls remaining.")


def main():
    parser = argparse.ArgumentParser(
        description="Backfill label_entropy, class_imbalance, input_resolution into runs.csv"
    )
    parser.add_argument("--dataset", type=str, default=None,
                        help="Fill only this dataset (default: all known datasets)")
    args = parser.parse_args()

    print(f"[backfill] Dataset features:")
    for ds, feat in DATASET_FEATURES.items():
        print(f"  {ds:20s}  entropy={feat['label_entropy']:.3f}  "
              f"imbalance={feat['class_imbalance']:.2f}  "
              f"resolution={feat['input_resolution']}")
    print()

    backfill(dataset=args.dataset)


if __name__ == "__main__":
    main()
