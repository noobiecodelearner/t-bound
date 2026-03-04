"""[tabular_loader). — all tabular datasets for [t-bound).

CRITICAL DESIGN RULES:
    1. Val splits loaded from data/val_splits/*.npz — never regenerated inline.
    2. StandardScaler fit on training pool only, applied to val and test.
       Scaler cached per dataset — fitted once, reused across fractions.
    3. load_tabular() is the only public function.
    4. Returns input_dim alongside loaders — needed to build ScalableMLP.

Supported datasets:
    covertype: Forest Cover Type, 581K samples, 7 classes,  54 features
    otto:      Otto Group Product, 61K samples,  9 classes,  93 features
    higgs:     HIGGS, 11M samples (use 1M), 2 classes, 28 features
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

VAL_SPLITS_DIR = Path(__file__).parent / "val_splits"
_cache: Dict[str, Dict] = {}


def _load_val_split(dataset: str, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    split_path = VAL_SPLITS_DIR / f"{dataset}_seed{seed}.npz"
    if not split_path.exists():
        raise FileNotFoundError(
            f"Val split not found: {split_path}\n"
            f"Run: python scripts/generate_val_splits.py"
        )
    data = np.load(split_path)
    return data["train_idx"], data["val_idx"]


class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.X[idx], int(self.y[idx])


def _make_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=2, pin_memory=True)


def _build_covertype_cache(data_path: Path, seed: int) -> None:
    from sklearn.datasets import fetch_covtype
    print("  [covertype] Loading...", flush=True)
    data = fetch_covtype(data_home=str(data_path))
    X = data.data.astype(np.float32)
    y = (data.target - 1).astype(np.int64)
    train_pool_idx, val_idx = _load_val_split("covertype", seed)
    scaler = StandardScaler()
    scaler.fit(X[train_pool_idx])
    X_scaled = scaler.transform(X)
    all_idx = np.arange(len(y))
    pool_and_val = np.concatenate([train_pool_idx, val_idx])
    test_idx = all_idx[~np.isin(all_idx, pool_and_val)][:50_000]
    _cache["covertype"] = {
        "X": X_scaled, "y": y,
        "train_pool_idx": train_pool_idx,
        "val_idx": val_idx, "test_idx": test_idx,
        "input_dim": X.shape[1],
    }
    print(f"  [covertype] {len(y):,} samples, {X.shape[1]} features.", flush=True)


def _build_otto_cache(data_path: Path, seed: int) -> None:
    print("  [otto] Loading CSV...", flush=True)
    df = pd.read_csv(data_path / "train.csv")
    X = df.drop(columns=["id", "target"]).values.astype(np.float32)
    y = (df["target"].str.replace("Class_", "").astype(int) - 1).values.astype(np.int64)
    train_pool_idx, val_idx = _load_val_split("otto", seed)
    scaler = StandardScaler()
    scaler.fit(X[train_pool_idx])
    X_scaled = scaler.transform(X)
    all_idx = np.arange(len(y))
    pool_and_val = np.concatenate([train_pool_idx, val_idx])
    test_idx = all_idx[~np.isin(all_idx, pool_and_val)][:10_000]
    _cache["otto"] = {
        "X": X_scaled, "y": y,
        "train_pool_idx": train_pool_idx,
        "val_idx": val_idx, "test_idx": test_idx,
        "input_dim": X.shape[1],
    }
    print(f"  [otto] {len(y):,} samples, {X.shape[1]} features.", flush=True)


def _build_higgs_cache(data_path: Path, seed: int, max_samples: int = 1_000_000) -> None:
    print(f"  [higgs] Loading up to {max_samples:,} samples...", flush=True)
    df = pd.read_csv(data_path / "HIGGS.csv.gz", header=None, nrows=max_samples)
    y = df.iloc[:, 0].values.astype(np.int64)
    X = df.iloc[:, 1:].values.astype(np.float32)
    train_pool_idx, val_idx = _load_val_split("higgs", seed)
    scaler = StandardScaler()
    scaler.fit(X[train_pool_idx])
    X_scaled = scaler.transform(X)
    all_idx = np.arange(len(y))
    pool_and_val = np.concatenate([train_pool_idx, val_idx])
    test_idx = all_idx[~np.isin(all_idx, pool_and_val)][:50_000]
    _cache["higgs"] = {
        "X": X_scaled, "y": y,
        "train_pool_idx": train_pool_idx,
        "val_idx": val_idx, "test_idx": test_idx,
        "input_dim": X.shape[1],
    }
    print(f"  [higgs] {len(y):,} samples, {X.shape[1]} features.", flush=True)


def _make_tabular_loaders(
    dataset: str, dataset_fraction: float, batch_size: int, seed: int
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    c = _cache[dataset]
    train_pool_idx = c["train_pool_idx"]
    val_idx = c["val_idx"]
    test_idx = c["test_idx"]
    X, y = c["X"], c["y"]

    n_train = max(1, int(dataset_fraction * len(train_pool_idx)))
    rng = np.random.RandomState(seed)
    selected = rng.choice(len(train_pool_idx), size=n_train, replace=False)
    train_indices = train_pool_idx[selected].tolist()

    print(
        f"  [{dataset}] fraction={dataset_fraction:.2f} → "
        f"{len(train_indices):,} train / {len(val_idx):,} val / {len(test_idx):,} test",
        flush=True,
    )

    return (
        _make_loader(TabularDataset(X[train_indices], y[train_indices]), batch_size, shuffle=True),
        _make_loader(TabularDataset(X[val_idx], y[val_idx]), batch_size, shuffle=False),
        _make_loader(TabularDataset(X[test_idx], y[test_idx]), batch_size, shuffle=False),
        c["input_dim"],
    )


def load_tabular(
    dataset: str,
    data_path: str,
    dataset_fraction: float = 1.0,
    batch_size: int = 256,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """Load any tabular dataset. Returns (train, val, test, input_dim).

    Args:
        dataset: 'covertype' | 'otto' | 'higgs'.
        data_path: Path to raw data directory.
        dataset_fraction: Fraction of training pool to use.
        batch_size: DataLoader batch size.
        seed: Random seed.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, input_dim).
    """
    data_path = Path(data_path)
    builders = {"covertype": _build_covertype_cache,
                "otto": _build_otto_cache,
                "higgs": _build_higgs_cache}

    if dataset not in builders:
        raise ValueError(f"Unknown tabular dataset: '{dataset}'. Supported: covertype, otto, higgs.")

    if dataset not in _cache:
        builders[dataset](data_path, seed)
    else:
        print(f"  [{dataset}] Using cached data.", flush=True)

    return _make_tabular_loaders(dataset, dataset_fraction, batch_size, seed)
