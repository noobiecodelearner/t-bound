"""
[t-bound). — What if you had to train less?

CLI for running scaling experiments, fitting surfaces,
optimizing recommendations, and generating graphs.

Usage:
    # Step 0: generate val splits (run once before anything)
    python main.py --generate_splits

    # Step 1: holdout validation on CIFAR-10 before full grid
    python main.py --validate --dataset cifar10 --target_accuracy 0.85 --device cuda

    # Step 2: run n_d_lr_grid for one dataset
    python main.py --run_scaling --config configs/vision_cifar10_grid.yaml --device cuda

    # Step 2b: run all vision datasets
    python main.py --run_scaling --domain vision --device cuda

    # Step 3: fit scaling laws from runs.csv
    python main.py --fit_surface --config configs/vision_cifar10_grid.yaml

    # Step 4: get recommendation (accuracy target)
    python main.py --optimize --config configs/vision_cifar10_grid.yaml --target_accuracy 0.85

    # Step 4b: get Chinchilla recommendation (compute budget)
    python main.py --optimize_compute --config configs/vision_cifar10_grid.yaml --compute_hours 10

    # Step 5: after batch_grid, run batch sweep
    python main.py --run_scaling --config configs/vision_cifar10_batch.yaml --device cuda

    # Step 6: generate publication graphs
    python main.py --generate_graphs --domain all

    # Step 7: build prior.csv from all fits
    python main.py --build_prior

    # Mock data for Dayanch
    python main.py --generate_mock
"""

import argparse
import sys
import os
import yaml
import pandas as pd
from pathlib import Path


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_configs_for_domain(domain: str, sweep_type: str = None) -> list:
    """Return all config paths for a domain."""
    configs_dir = Path("configs")
    if domain == "all":
        pattern = "*.yaml"
    else:
        pattern = f"{domain}_*.yaml"

    paths = sorted(configs_dir.glob(pattern))
    if sweep_type:
        paths = [p for p in paths if sweep_type in p.stem]
    return [str(p) for p in paths]


def cmd_generate_splits(args):
    from scripts.generate_val_splits import main
    main()


def cmd_validate(args):
    from scripts.validate_holdout import validate_holdout
    targets = args.target_accuracies or [args.target_accuracy]
    results = []
    for tau in targets:
        r = validate_holdout(
            dataset=args.dataset,
            target_accuracy=tau,
            device=args.device,
            verbose=args.verbose,
        )
        results.append(r)
    all_passed = all(r.get("passed", False) for r in results)
    if not all_passed:
        sys.exit(1)


def cmd_run_scaling(args):
    from scaling.scaling_runner import ScalingRunner

    if args.config:
        configs = [args.config]
    elif args.domain:
        sweep = "grid" if args.sweep == "grid" else "batch" if args.sweep == "batch" else None
        configs = get_configs_for_domain(args.domain, sweep)
    else:
        print("[t-bound] Provide --config or --domain")
        sys.exit(1)

    for config_path in configs:
        print(f"\n[t-bound] Config: {config_path}")
        cfg = load_config(config_path)
        runner = ScalingRunner(
            config=cfg,
            device=args.device,
            verbose=args.verbose,
            results_dir=args.results_dir,
        )
        runner.run()


def cmd_fit_surface(args):
    from scaling.surface_fit import fit_model_size, fit_lr_scaling, fit_nd_surface
    from utils.logger import FitsLogger
    import numpy as np

    if not args.config:
        print("[t-bound] --fit_surface requires --config")
        sys.exit(1)

    cfg    = load_config(args.config)
    ds     = cfg["dataset"]
    arch   = cfg["architecture"]
    domain = cfg["domain"]

    runs_path = Path(args.results_dir) / "runs.csv"
    if not runs_path.exists():
        print(f"[t-bound] {runs_path} not found. Run experiments first.")
        sys.exit(1)

    df = pd.read_csv(runs_path)
    df = df[df["dataset"] == ds]
    grid_df = df[df["sweep_type"] == "n_d_lr_grid"]

    if len(grid_df) < 4:
        print(f"[t-bound] Not enough runs ({len(grid_df)}). Need at least 4.")
        sys.exit(1)

    fits_logger = FitsLogger(results_dir=args.results_dir)

    # ── fit α (model size) ──────────────────────────────────────────────────
    best_per_n = (grid_df
                  .groupby("params")["val_accuracy"]
                  .max()
                  .reset_index())
    params_arr = best_per_n["params"].values.astype(float)
    acc_arr    = best_per_n["val_accuracy"].values.astype(float)

    fit_ms = fit_model_size(params_arr, acc_arr)
    print(f"[t-bound] Model size fit: α={fit_ms['alpha']:.4f} "
          f"a={fit_ms['a']:.4f} R²={fit_ms['r2']:.4f}")

    fits_logger.log(
        dataset=ds, architecture_family=arch, domain=domain,
        sweep_type="model_size", exponent_type="alpha",
        exponent_value=fit_ms["alpha"],
        param_a=fit_ms["a"], param_b=fit_ms["b"],
        r2=fit_ms["r2"], mae=fit_ms["mae"],
        n_runs_used=len(best_per_n),
    )

    # ── fit β (lr scaling) ──────────────────────────────────────────────────
    best_lr_per_n = (
        grid_df.groupby("params")
        .apply(lambda g: g.loc[g["val_accuracy"].idxmax(), "learning_rate"])
        .reset_index()
    )
    best_lr_per_n.columns = ["params", "lr_star"]

    fit_lr = fit_lr_scaling(
        best_lr_per_n["params"].values.astype(float),
        best_lr_per_n["lr_star"].values.astype(float),
    )
    print(f"[t-bound] LR scaling fit: β={fit_lr['beta']:.4f} "
          f"c={fit_lr['c']:.6f} R²={fit_lr['r2']:.4f}")
    fits_logger.log(
        dataset=ds, architecture_family=arch, domain=domain,
        sweep_type="lr", exponent_type="beta",
        exponent_value=fit_lr["beta"], param_c=fit_lr["c"],
        r2=fit_lr["r2"], n_runs_used=len(best_lr_per_n),
    )

    # ── fit δ (N,D surface) if multiple D values ─────────────────────────────
    n_d_values = len(grid_df[["params", "dataset_size"]].drop_duplicates())
    if n_d_values >= 4:
        best_per_nd = (
            grid_df.groupby(["params", "dataset_size"])["val_accuracy"]
            .max()
            .reset_index()
        )
        fit_nd = fit_nd_surface(
            best_per_nd["params"].values.astype(float),
            best_per_nd["dataset_size"].values.astype(float),
            best_per_nd["val_accuracy"].values.astype(float),
        )
        print(f"[t-bound] N×D surface fit: α={fit_nd['alpha']:.4f} "
              f"δ={fit_nd['delta']:.4f} R²={fit_nd['r2']:.4f}")
        fits_logger.log(
            dataset=ds, architecture_family=arch, domain=domain,
            sweep_type="dataset", exponent_type="delta",
            exponent_value=fit_nd["delta"],
            param_a=fit_nd["a"], param_b=fit_nd["b"],
            r2=fit_nd["r2"], mae=fit_nd["mae"],
            n_runs_used=len(best_per_nd),
        )


def cmd_optimize(args):
    from optimization.optimizer import ScaleOptimizer

    if not args.config:
        print("[t-bound] --optimize requires --config")
        sys.exit(1)

    cfg = load_config(args.config)
    ds  = cfg["dataset"]

    runs_path = Path(args.results_dir) / "runs.csv"
    df = pd.read_csv(runs_path)

    optimizer = ScaleOptimizer(df, dataset=ds)
    rec = optimizer.optimize_accuracy(
        target_accuracy=args.target_accuracy,
        include_batch=True,
    )

    print(f"\n[t-bound] Recommendation for {ds} @ τ={args.target_accuracy}")
    print(f"  N*:               {rec.get('n_star', 'N/A'):,}")
    print(f"  lr*:              {rec.get('lr_star', 'N/A')}")
    print(f"  batch*:           {rec.get('batch_star', 'N/A')}")
    print(f"  Expected accuracy: {rec.get('expected_accuracy', 'N/A')}")
    print(f"  CI:               [{rec.get('ci_lower')}, {rec.get('ci_upper')}]")
    print(f"  Confidence:        {rec.get('confidence', 'N/A')}")
    print(f"  Compute saved:     {rec.get('compute_saved_fraction', 0)*100:.1f}%")
    print(f"  α:                 {rec.get('alpha', 'N/A')}")
    print(f"  R²:                {rec.get('fit_r2', 'N/A')}")


def cmd_optimize_compute(args):
    from optimization.optimizer import ScaleOptimizer

    if not args.config:
        print("[t-bound] --optimize_compute requires --config")
        sys.exit(1)

    cfg = load_config(args.config)
    ds  = cfg["dataset"]

    runs_path = Path(args.results_dir) / "runs.csv"
    df = pd.read_csv(runs_path)

    optimizer = ScaleOptimizer(df, dataset=ds)
    rec = optimizer.optimize_compute_budget(
        compute_budget_hours=args.compute_hours,
    )

    print(f"\n[t-bound] Chinchilla Recommendation for {ds} @ {args.compute_hours}h")
    print(f"  N*:               {rec.get('n_star', 'N/A'):,}")
    print(f"  D*:               {rec.get('d_star', 'N/A'):,} samples")
    print(f"  lr*:              {rec.get('lr_star', 'N/A')}")
    print(f"  Expected accuracy: {rec.get('expected_accuracy', 'N/A')}")
    print(f"  Energy:            {rec.get('energy_kwh', 'N/A')} kWh")
    print(f"  Carbon:            {rec.get('carbon_g', 'N/A')} g CO₂")


def cmd_generate_graphs(args):
    from scaling.graphs import generate_all_graphs

    runs_path = Path(args.results_dir) / "runs.csv"
    if args.config:
        cfg = load_config(args.config)
        generate_all_graphs(str(runs_path),
                            args.graphs_dir,
                            dataset=cfg["dataset"])
    else:
        generate_all_graphs(str(runs_path), args.graphs_dir)


def cmd_build_prior(args):
    from scripts.build_prior import main
    main()


def cmd_generate_mock(args):
    from scripts.generate_mock_data import main
    main()


def main():
    parser = argparse.ArgumentParser(
        description="[t-bound). — What if you had to train less?"
    )

    # commands
    parser.add_argument("--generate_splits",  action="store_true")
    parser.add_argument("--validate",         action="store_true")
    parser.add_argument("--run_scaling",      action="store_true")
    parser.add_argument("--fit_surface",      action="store_true")
    parser.add_argument("--optimize",         action="store_true")
    parser.add_argument("--optimize_compute", action="store_true")
    parser.add_argument("--generate_graphs",  action="store_true")
    parser.add_argument("--build_prior",      action="store_true")
    parser.add_argument("--generate_mock",    action="store_true")

    # config / domain
    parser.add_argument("--config",     type=str, default=None)
    parser.add_argument("--domain",     type=str, default=None,
                        choices=["vision", "nlp", "tabular", "all"])
    parser.add_argument("--sweep",      type=str, default="grid",
                        choices=["grid", "batch"],
                        help="Which sweep type to run when using --domain")
    parser.add_argument("--dataset",    type=str, default="cifar10")

    # recommendation params
    parser.add_argument("--target_accuracy",  type=float, default=0.85)
    parser.add_argument("--target_accuracies", type=float, nargs="+")
    parser.add_argument("--compute_hours",    type=float, default=10.0)

    # infrastructure
    parser.add_argument("--device",      type=str, default="cpu")
    parser.add_argument("--verbose",     action="store_true")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--graphs_dir",  type=str, default="results/graphs")
    parser.add_argument("--bootstrap",   action="store_true")

    args = parser.parse_args()

    if args.generate_splits:
        cmd_generate_splits(args)
    elif args.validate:
        cmd_validate(args)
    elif args.run_scaling:
        cmd_run_scaling(args)
    elif args.fit_surface:
        cmd_fit_surface(args)
    elif args.optimize:
        cmd_optimize(args)
    elif args.optimize_compute:
        cmd_optimize_compute(args)
    elif args.generate_graphs:
        cmd_generate_graphs(args)
    elif args.build_prior:
        cmd_build_prior(args)
    elif args.generate_mock:
        cmd_generate_mock(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
