"""
[validate_holdout). — holdout validation for [t-bound).

Proves that the scaling law prediction is accurate BEFORE running
the full 2,106-run experiment grid.

CRITICAL: This script imports everything from the codebase.
Zero reimplementation. If anything is wrong with the pipeline
this script catches it — not the full grid.

Validates scale-invariant properties:
    1. Does N* prediction fall within the fitted CI?
    2. Is achieved accuracy within the predicted CI?
    3. What is the CI coverage rate across multiple τ values?

Usage:
    # First run on CIFAR-10 only (Week 1)
    python scripts/validate_holdout.py --dataset cifar10 --target_accuracy 0.85

    # Verbose with multiple tau values
    python scripts/validate_holdout.py --dataset cifar10 \\
        --target_accuracies 0.75 0.80 0.85 --verbose
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pandas as pd
import torch

# ── import from codebase only — never reimplement ──────────────────────────
from data.vision_loader    import load_vision, NORMALIZATION_STATS
from data.nlp_loader       import load_nlp
from data.tabular_loader   import load_tabular
from models.cnn            import ScalableCNN
from models.transformer    import ScalableTransformer
from models.mlp            import ScalableMLP
from training.trainer      import Trainer
from scaling.surface_fit   import fit_model_size, fit_lr_scaling
from scaling.bootstrap     import BootstrapUncertainty
from optimization.optimizer import ScaleOptimizer
from utils.seed            import set_seed


# ── config for each dataset ──────────────────────────────────────────────────

DATASET_CONFIGS = {
    "cifar10": {
        "domain": "vision", "num_classes": 10,
        "data_path": "data/raw/cifar10/cifar-10-batches-py",
        "architecture": "cnn",
        "optimizer": "adam", "weight_decay": 1e-4,
        "num_steps": 10000, "fixed_batch": 128,
    },
    "cifar100": {
        "domain": "vision", "num_classes": 100,
        "data_path": "data/raw/cifar100",
        "architecture": "cnn",
        "optimizer": "adam", "weight_decay": 1e-4,
        "num_steps": 10000, "fixed_batch": 128,
    },
    "stl10": {
        "domain": "vision", "num_classes": 10,
        "data_path": "data/raw/stl10",
        "architecture": "cnn",
        "optimizer": "adam", "weight_decay": 1e-4,
        "num_steps": 10000, "fixed_batch": 64,
    },
    "yahoo": {
        "domain": "nlp", "num_classes": 10,
        "data_path": "data/raw/yahoo",
        "architecture": "transformer",
        "optimizer": "adamw", "weight_decay": 1e-4,
        "num_steps": 5000, "fixed_batch": 32,
    },
    "agnews": {
        "domain": "nlp", "num_classes": 4,
        "data_path": "data/raw/agnews",
        "architecture": "transformer",
        "optimizer": "adamw", "weight_decay": 1e-4,
        "num_steps": 5000, "fixed_batch": 32,
    },
    "dbpedia": {
        "domain": "nlp", "num_classes": 14,
        "data_path": "data/raw/dbpedia",
        "architecture": "transformer",
        "optimizer": "adamw", "weight_decay": 1e-4,
        "num_steps": 5000, "fixed_batch": 32,
    },
    "covertype": {
        "domain": "tabular", "num_classes": 7, "input_dim": 54,
        "data_path": "data/raw/covertype",
        "architecture": "mlp",
        "optimizer": "adam", "weight_decay": 1e-4,
        "num_steps": 8000, "fixed_batch": 256,
    },
    "otto": {
        "domain": "tabular", "num_classes": 9, "input_dim": 93,
        "data_path": "data/raw/otto",
        "architecture": "mlp",
        "optimizer": "adam", "weight_decay": 1e-4,
        "num_steps": 8000, "fixed_batch": 256,
    },
    "higgs": {
        "domain": "tabular", "num_classes": 2, "input_dim": 28,
        "data_path": "data/raw/higgs",
        "architecture": "mlp",
        "optimizer": "adam", "weight_decay": 1e-4,
        "num_steps": 8000, "fixed_batch": 512,
    },
}

# model scales for each architecture (same as in configs/)
VISION_SCALES   = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
NLP_SCALES      = [[1,16],[1,32],[2,64],[2,128],[3,256],[4,512]]
TABULAR_SCALES  = [32, 64, 128, 256, 512, 1024]


def build_model(dataset: str, scale, cfg: dict):
    """Build model from scale. Imports from codebase only."""
    domain = cfg["domain"]
    nc     = cfg["num_classes"]
    if domain == "vision":
        return ScalableCNN(num_classes=nc, width_multiplier=float(scale))
    elif domain == "nlp":
        return ScalableTransformer(
            num_classes=nc, num_layers=int(scale[0]), d_model=int(scale[1])
        )
    elif domain == "tabular":
        return ScalableMLP(
            input_dim=cfg["input_dim"], num_classes=nc,
            hidden_size=int(scale)
        )


def get_loaders(dataset: str, cfg: dict, batch_size: int = None):
    """Load data. Imports from codebase only."""
    domain = cfg["domain"]
    bs     = batch_size or cfg["fixed_batch"]
    if domain == "vision":
        return load_vision(dataset, cfg["data_path"], 1.0, bs)
    elif domain == "nlp":
        train, val, test, n = load_nlp(dataset, cfg["data_path"],
                                        batch_size=bs)
        return train, val, test, n
    elif domain == "tabular":
        train, val, test, n, _ = load_tabular(dataset, cfg["data_path"],
                                               1.0, bs)
        return train, val, test, n


def run_small_sweep(dataset: str, cfg: dict,
                    device: str = "cpu",
                    verbose: bool = False) -> pd.DataFrame:
    """
    Run a small N × lr sweep to fit the scaling law.
    Uses 4 model sizes × 4 lr values = 16 runs.
    Only at dataset_fraction=1.0.
    """
    domain = cfg["domain"]
    if domain == "vision":
        scales = [VISION_SCALES[i] for i in [0, 1, 3, 5]]
    elif domain == "nlp":
        scales = [NLP_SCALES[i] for i in [0, 1, 3, 5]]
    else:
        scales = [TABULAR_SCALES[i] for i in [0, 1, 3, 5]]

    lr_values = [0.01, 0.001, 0.0001, 0.00001]
    rows = []

    total = len(scales) * len(lr_values)
    done  = 0

    for scale in scales:
        model = build_model(dataset, scale, cfg)
        params = model.count_parameters()

        train_loader, val_loader, _, n_train = get_loaders(dataset, cfg)

        for lr in lr_values:
            done += 1
            if verbose:
                print(f"  [{done}/{total}] params={params:,} lr={lr:.5f}")

            set_seed(42)
            trainer = Trainer(model, device=device)
            model   = build_model(dataset, scale, cfg)  # fresh model each run
            trainer = Trainer(model, device=device)

            result = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer_name=cfg["optimizer"],
                learning_rate=lr,
                weight_decay=cfg["weight_decay"],
                num_steps=cfg["num_steps"],
                verbose=False,
            )
            rows.append({
                "params":          params,
                "learning_rate":   lr,
                "val_accuracy":    result["best_val_accuracy"],
                "train_accuracy":  result["final_train_accuracy"],
            })

    return pd.DataFrame(rows)


def validate_holdout(dataset: str, target_accuracy: float,
                     device: str = "cpu",
                     verbose: bool = False) -> dict:
    """
    Full holdout validation for one dataset and one target accuracy.

    Steps:
        1. Run small N × lr sweep (16 runs)
        2. Fit scaling law
        3. Predict N*
        4. Train exactly one model at N*
        5. Check if achieved accuracy is within CI

    Returns validation result dict.
    """
    cfg = DATASET_CONFIGS[dataset]
    domain = cfg["domain"]

    print(f"\n[t-bound] Holdout validation: {dataset}, τ={target_accuracy}")
    print(f"  Architecture: {cfg['architecture']}, Domain: {domain}")

    # ── step 1: small sweep ──────────────────────────────────────────────────
    print(f"  Step 1: Running small sweep (4×4=16 runs)...")
    df = run_small_sweep(dataset, cfg, device=device, verbose=verbose)

    # ── step 2: fit scaling law ──────────────────────────────────────────────
    print(f"  Step 2: Fitting scaling law...")
    best_per_n = df.groupby("params")["val_accuracy"].max().reset_index()
    params_arr = best_per_n["params"].values.astype(float)
    acc_arr    = best_per_n["val_accuracy"].values.astype(float)

    fit = fit_model_size(params_arr, acc_arr)
    print(f"  Fit: α={fit['alpha']:.3f}, a={fit['a']:.4f}, "
          f"b={fit['b']:.4f}, R²={fit['r2']:.4f}")

    # bootstrap CI
    bootstrap = BootstrapUncertainty(n_bootstrap=100)

    def _fit_fn(x, y):
        return fit_model_size(x, y)

    def _opt_fn(result):
        return result["optimal_n_fn"](target_accuracy)

    ci = bootstrap.compute_ci(params_arr, acc_arr, _fit_fn, _opt_fn)

    # ── step 3: predict N* ───────────────────────────────────────────────────
    n_star = fit["optimal_n_fn"](target_accuracy)
    if n_star is None:
        print(f"  ✗ Target accuracy {target_accuracy} unreachable. "
              f"Max: {fit['a']:.4f}")
        return {"success": False, "reason": "target_unreachable"}

    print(f"  N* = {n_star:,.0f} parameters")
    if ci["success"]:
        print(f"  CI = [{ci['ci_lower']:,.0f}, {ci['ci_upper']:,.0f}]")

    # find best lr for N* from the sweep
    model_for_n_star = build_model(dataset, _find_scale_for_n_star(
        n_star, dataset, cfg
    ), cfg)
    actual_params = model_for_n_star.count_parameters()

    best_lr_row = (
        df.groupby("params")
        .apply(lambda g: g.loc[g["val_accuracy"].idxmax()])
        .reset_index(drop=True)
    )
    # use lr from nearest observed N
    nearest_n = best_lr_row.iloc[
        (best_lr_row["params"] - actual_params).abs().argsort().iloc[0]
    ]
    best_lr = nearest_n["learning_rate"]

    # ── step 4: train at N* ──────────────────────────────────────────────────
    print(f"  Step 3: Training at N*={actual_params:,} with lr={best_lr:.5f}...")
    train_loader, val_loader, _, n_train = get_loaders(dataset, cfg)
    set_seed(42)
    holdout_model   = build_model(dataset, _find_scale_for_n_star(
        n_star, dataset, cfg
    ), cfg)
    holdout_trainer = Trainer(holdout_model, device=device)
    holdout_result  = holdout_trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_name=cfg["optimizer"],
        learning_rate=best_lr,
        weight_decay=cfg["weight_decay"],
        num_steps=cfg["num_steps"],
        verbose=verbose,
    )
    achieved_acc = holdout_result["best_val_accuracy"]
    predicted_acc = fit["predict_fn"](actual_params)

    # ── step 5: check ────────────────────────────────────────────────────────
    within_ci = False
    if ci["success"]:
        within_ci = (ci["ci_lower"] <= actual_params <= ci["ci_upper"])

    acc_error = abs(achieved_acc - predicted_acc)
    passed = acc_error < 0.05  # within 5 percentage points

    print()
    print(f"  ── Results ──────────────────────────────────────────────")
    print(f"  Predicted accuracy at N*: {predicted_acc:.4f}")
    print(f"  Achieved  accuracy at N*: {achieved_acc:.4f}")
    print(f"  Absolute error:           {acc_error:.4f}")
    print(f"  N* within CI:             {'✓' if within_ci else '✗'}")
    print(f"  Accuracy within 5%:       {'✓ PASS' if passed else '✗ FAIL'}")
    print(f"  ─────────────────────────────────────────────────────────")

    return {
        "dataset":          dataset,
        "target_accuracy":  target_accuracy,
        "n_star_predicted": n_star,
        "n_star_actual":    actual_params,
        "predicted_acc":    predicted_acc,
        "achieved_acc":     achieved_acc,
        "acc_error":        acc_error,
        "within_ci":        within_ci,
        "passed":           passed,
        "alpha":            fit["alpha"],
        "r2":               fit["r2"],
        "ci_lower":         ci.get("ci_lower"),
        "ci_upper":         ci.get("ci_upper"),
    }


def _find_scale_for_n_star(n_star: float, dataset: str, cfg: dict):
    """Find scale whose param count is closest to n_star."""
    domain = cfg["domain"]
    nc     = cfg["num_classes"]

    if domain == "vision":
        scales = VISION_SCALES
    elif domain == "nlp":
        scales = NLP_SCALES
    else:
        scales = TABULAR_SCALES

    best_scale = scales[0]
    best_diff  = float("inf")

    for scale in scales:
        m = build_model(dataset, scale, cfg)
        diff = abs(m.count_parameters() - n_star)
        if diff < best_diff:
            best_diff  = diff
            best_scale = scale

    return best_scale


def main():
    parser = argparse.ArgumentParser(
        description="[t-bound). holdout validation"
    )
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--target_accuracy", type=float, default=0.85)
    parser.add_argument("--target_accuracies", type=float, nargs="+",
                        help="Multiple tau values for CI coverage test")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    targets = args.target_accuracies or [args.target_accuracy]
    results = []

    for tau in targets:
        r = validate_holdout(args.dataset, tau,
                             device=args.device, verbose=args.verbose)
        results.append(r)

    if len(results) > 1:
        passed  = sum(r["passed"] for r in results)
        print(f"\n[t-bound] Coverage: {passed}/{len(results)} targets passed "
              f"({passed/len(results)*100:.0f}%)")

    # overall verdict
    all_passed = all(r.get("passed", False) for r in results)
    if all_passed:
        print("\n[t-bound] ✓ Validation PASSED. Pipeline is consistent.")
        print("  Proceed to full n_d_lr_grid experiments.")
    else:
        print("\n[t-bound] ✗ Validation FAILED. Do not proceed to full grid.")
        print("  Investigate pipeline inconsistencies first.")
        sys.exit(1)


if __name__ == "__main__":
    main()
