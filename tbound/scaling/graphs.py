"""[graphs). — publication-quality scaling law graphs for [t-bound)."""

import numpy as np
from pathlib import Path


def generate_all_graphs(runs_csv: str = "results/runs.csv",
                        output_dir: str = "results/graphs",
                        dataset: str = None):
    """
    Generate all scaling law graphs from runs.csv.
    Requires matplotlib. Saves PNG files to output_dir.

    Graphs generated:
        1. Accuracy vs params (model size scaling) per dataset per D fraction
        2. lr* vs params (lr scaling) per dataset
        3. Accuracy vs batch size per dataset
        4. (N, D) surface heatmap per dataset (Chinchilla frontier)
        5. Compute-optimal frontier per dataset
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("[t-bound] matplotlib or pandas not installed. Skipping graphs.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(runs_csv)
    if dataset:
        df = df[df["dataset"] == dataset]

    datasets = df["dataset"].unique()

    for ds in datasets:
        df_ds = df[df["dataset"] == ds]

        # ── graph 1: accuracy vs params per D fraction ─────────────────────
        grid_df = df_ds[df_ds["sweep_type"] == "n_d_lr_grid"]
        if not grid_df.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            for frac, grp in grid_df.groupby("dataset_fraction"):
                # best acc per param count (optimized over lr)
                best = grp.groupby("params")["val_accuracy"].max().reset_index()
                best = best.sort_values("params")
                ax.plot(best["params"], best["val_accuracy"],
                        marker="o", label=f"D={frac:.0%}")
            ax.set_xscale("log")
            ax.set_xlabel("Parameters (N)")
            ax.set_ylabel("Val Accuracy")
            ax.set_title(f"[t-bound) Scaling Law — {ds}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(output_dir / f"{ds}_model_size_scaling.png", dpi=150)
            plt.close(fig)

        # ── graph 2: optimal lr vs params ──────────────────────────────────
        if not grid_df.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            # for each (params, D) find the lr that maximized val_acc
            best_lr = (
                grid_df.groupby(["params", "dataset_fraction"])
                .apply(lambda g: g.loc[g["val_accuracy"].idxmax()])
                .reset_index(drop=True)
            )
            full_d = best_lr[best_lr["dataset_fraction"] == best_lr["dataset_fraction"].max()]
            full_d = full_d.sort_values("params")
            ax.loglog(full_d["params"], full_d["learning_rate"],
                      marker="o", color="steelblue")
            ax.set_xlabel("Parameters (N)")
            ax.set_ylabel("Optimal lr*(N)")
            ax.set_title(f"[t-bound) LR Scaling — {ds}")
            ax.grid(True, alpha=0.3, which="both")
            fig.tight_layout()
            fig.savefig(output_dir / f"{ds}_lr_scaling.png", dpi=150)
            plt.close(fig)

        # ── graph 3: accuracy vs batch size ────────────────────────────────
        batch_df = df_ds[df_ds["sweep_type"] == "batch_grid"]
        if not batch_df.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            full_d = batch_df[
                batch_df["dataset_fraction"] == batch_df["dataset_fraction"].max()
            ].sort_values("batch_size")
            ax.semilogx(full_d["batch_size"], full_d["val_accuracy"],
                        marker="o", color="darkorange", base=2)
            ax.set_xlabel("Batch Size")
            ax.set_ylabel("Val Accuracy")
            ax.set_title(f"[t-bound) Batch Scaling — {ds}")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(output_dir / f"{ds}_batch_scaling.png", dpi=150)
            plt.close(fig)

    print(f"[t-bound] graphs saved to {output_dir}/")
