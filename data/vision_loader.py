"""[vision_loader). — unified vision data loader for [t-bound).

All normalization constants are defined once here.
Val splits are loaded from data/val_splits/*.npz — never regenerated inline.
Raw data is loaded ONCE per Python process and cached in _cache.
Subsequent calls slice from cache — no re-reading from disk.

Run scripts/generate_val_splits.py once before any experiments.
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from pathlib import Path
from typing import Dict, Tuple
import PIL.Image

# ── normalization constants — single source of truth ─────────────────────────
# validate_holdout.py imports NORMALIZATION_STATS from here.
# Never hardcode these values anywhere else.

NORMALIZATION_STATS: Dict[str, Dict] = {
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std":  (0.2470, 0.2435, 0.2616),
    },
    "cifar100": {
        "mean": (0.5071, 0.4867, 0.4408),
        "std":  (0.2675, 0.2565, 0.2761),
    },
    "stl10": {
        "mean": (0.4467, 0.4398, 0.4066),
        "std":  (0.2603, 0.2566, 0.2713),
    },
}

VAL_SPLITS_DIR = Path(__file__).parent / "val_splits"

# ── module-level cache — raw arrays loaded once per process ──────────────────
# Structure per dataset:
#   cifar10:
#       "images":      np.ndarray (50000, 32, 32, 3) uint8
#       "labels":      np.ndarray (50000,) int64
#       "test_images": np.ndarray (10000, 32, 32, 3) uint8
#       "test_labels": np.ndarray (10000,) int64
#       "train_idx":   np.ndarray — train pool indices
#       "val_idx":     np.ndarray — val indices
#   cifar100 / stl10:
#       torchvision dataset objects + split indices

_cache: Dict[str, Dict] = {}


# ── val split loader ──────────────────────────────────────────────────────────

def _load_val_split(dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    path = VAL_SPLITS_DIR / f"{dataset_name}_seed42.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Val split not found: {path}\n"
            f"Run: python scripts/generate_val_splits.py"
        )
    data = np.load(path)
    return data["train_idx"], data["val_idx"]


# ── dataset wrapper ───────────────────────────────────────────────────────────

class _ArrayDataset(Dataset):
    """Wraps (H, W, C) uint8 numpy arrays as a torch Dataset."""
    def __init__(self, images: np.ndarray, labels: np.ndarray,
                 transform=None) -> None:
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = PIL.Image.fromarray(self.images[idx])
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, int(self.labels[idx])


def _make_loader(ds: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=2, pin_memory=True, persistent_workers=False)


# ── cache builders — called once per dataset per process ─────────────────────

def _build_cifar10_cache(data_path: str) -> None:
    print("  [cifar10] Loading raw pickle files into cache...", flush=True)

    def _load_batch(fpath):
        with open(fpath, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        images = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = np.array(d[b"labels"], dtype=np.int64)
        return images, labels

    imgs, lbls = [], []
    for i in range(1, 6):
        im, lb = _load_batch(os.path.join(data_path, f"data_batch_{i}"))
        imgs.append(im)
        lbls.append(lb)

    train_imgs = np.concatenate(imgs)   # (50000, 32, 32, 3) uint8
    train_lbls = np.concatenate(lbls)
    test_imgs, test_lbls = _load_batch(os.path.join(data_path, "test_batch"))

    train_idx, val_idx = _load_val_split("cifar10")

    _cache["cifar10"] = {
        "images":      train_imgs,
        "labels":      train_lbls,
        "test_images": test_imgs,
        "test_labels": test_lbls,
        "train_idx":   train_idx,
        "val_idx":     val_idx,
        "data_path":   data_path,
    }
    print(f"  [cifar10] Cached {len(train_imgs):,} train + {len(test_imgs):,} test.",
          flush=True)


def _build_cifar100_cache(data_path: str) -> None:
    print("  [cifar100] Loading into cache...", flush=True)
    stats = NORMALIZATION_STATS["cifar100"]
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(stats["mean"], stats["std"]),
    ])
    ds_train = datasets.CIFAR100(data_path, train=True,  download=False, transform=val_tf)
    ds_test  = datasets.CIFAR100(data_path, train=False, download=False, transform=val_tf)
    train_idx, val_idx = _load_val_split("cifar100")
    _cache["cifar100"] = {
        "ds_train":  ds_train,
        "ds_test":   ds_test,
        "train_idx": train_idx,
        "val_idx":   val_idx,
        "data_path": data_path,
    }
    print(f"  [cifar100] Cached {len(ds_train):,} train + {len(ds_test):,} test.",
          flush=True)


def _build_stl10_cache(data_path: str) -> None:
    """
    Load STL-10 from raw binary files, same pattern as cifar10.
    Expects:
        data_path/stl10_binary/train_X.bin   — 5000 × 3 × 96 × 96 uint8
        data_path/stl10_binary/train_y.bin   — 5000 uint8 labels (1-indexed)
        data_path/stl10_binary/test_X.bin    — 8000 × 3 × 96 × 96 uint8
        data_path/stl10_binary/test_y.bin    — 8000 uint8 labels (1-indexed)
    """
    print("  [stl10] Loading from raw binary files...", flush=True)
    base = Path(data_path) / "stl10_binary"

    def _read_images(path):
        with open(path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        # STL-10 layout: (N, 3, 96, 96) in CHW order, then transpose to HWC
        n = len(data) // (3 * 96 * 96)
        data = data.reshape(n, 3, 96, 96)
        return data.transpose(0, 2, 3, 1)  # → (N, 96, 96, 3)

    def _read_labels(path):
        with open(path, "rb") as f:
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return (labels - 1).astype(np.int64)  # 1-indexed → 0-indexed

    images_train = _read_images(base / "train_X.bin")
    labels_train = _read_labels(base / "train_y.bin")
    images_test  = _read_images(base / "test_X.bin")
    labels_test  = _read_labels(base / "test_y.bin")

    train_idx, val_idx = _load_val_split("stl10")

    _cache["stl10"] = {
        "images":      images_train,
        "labels":      labels_train,
        "test_images": images_test,
        "test_labels": labels_test,
        "train_idx":   train_idx,
        "val_idx":     val_idx,
    }
    print(f"  [stl10] Cached {len(images_train):,} train + "
          f"{len(images_test):,} test. Shape: {images_train.shape[1:]}", flush=True)


# ── loader builders — called every run, slices from cache ────────────────────

def _loaders_cifar10(dataset_fraction: float, batch_size: int,
                     seed: int) -> Tuple:
    c = _cache["cifar10"]
    stats = NORMALIZATION_STATS["cifar10"]

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(stats["mean"], stats["std"]),
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(stats["mean"], stats["std"]),
    ])

    pool = c["train_idx"]
    n_train = max(1, int(len(pool) * dataset_fraction))
    rng = np.random.RandomState(seed)
    chosen = rng.choice(pool, n_train, replace=False)

    train_ds = _ArrayDataset(c["images"][chosen],        c["labels"][chosen],        train_tf)
    val_ds   = _ArrayDataset(c["images"][c["val_idx"]],  c["labels"][c["val_idx"]],  val_tf)
    test_ds  = _ArrayDataset(c["test_images"],           c["test_labels"],            val_tf)

    return (
        _make_loader(train_ds, batch_size,     shuffle=True),
        _make_loader(val_ds,   batch_size * 2, shuffle=False),
        _make_loader(test_ds,  batch_size * 2, shuffle=False),
        len(chosen),
    )


def _loaders_cifar100(dataset_fraction: float, batch_size: int,
                      seed: int) -> Tuple:
    c = _cache["cifar100"]
    stats = NORMALIZATION_STATS["cifar100"]

    # Build train-augmented version fresh each call — dataset object is cheap
    # raw pixel data is already in memory via torchvision's internal cache
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(stats["mean"], stats["std"]),
    ])
    ds_train_aug = datasets.CIFAR100(
        c["data_path"], train=True, download=False, transform=train_tf
    )

    pool = c["train_idx"]
    n_train = max(1, int(len(pool) * dataset_fraction))
    rng = np.random.RandomState(seed)
    chosen = rng.choice(pool, n_train, replace=False)

    return (
        _make_loader(Subset(ds_train_aug,  chosen),        batch_size,     shuffle=True),
        _make_loader(Subset(c["ds_train"], c["val_idx"]),  batch_size * 2, shuffle=False),
        _make_loader(c["ds_test"],                         batch_size * 2, shuffle=False),
        len(chosen),
    )


def _loaders_stl10(dataset_fraction: float, batch_size: int,
                   seed: int) -> Tuple:
    """Raw binary loader — no torchvision.datasets.STL10 dependency."""
    c = _cache["stl10"]
    stats = NORMALIZATION_STATS["stl10"]

    train_tf = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(stats["mean"], stats["std"]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(stats["mean"], stats["std"]),
    ])

    pool = c["train_idx"]
    n_train = max(1, int(len(pool) * dataset_fraction))
    rng = np.random.RandomState(seed)
    chosen = rng.choice(pool, n_train, replace=False)

    train_ds = _ArrayDataset(c["images"][chosen],        c["labels"][chosen],        train_tf)
    val_ds   = _ArrayDataset(c["images"][c["val_idx"]],  c["labels"][c["val_idx"]],  val_tf)
    test_ds  = _ArrayDataset(c["test_images"],           c["test_labels"],            val_tf)

    return (
        _make_loader(train_ds, batch_size,     shuffle=True),
        _make_loader(val_ds,   batch_size * 2, shuffle=False),
        _make_loader(test_ds,  batch_size * 2, shuffle=False),
        len(chosen),
    )


# ── public dispatcher ─────────────────────────────────────────────────────────

def load_vision(dataset: str, data_path: str,
                dataset_fraction: float = 1.0,
                batch_size: int = 128,
                seed: int = 42) -> Tuple:
    """Unified vision loader. Returns (train_loader, val_loader, test_loader, n_train).

    First call per dataset loads raw data into memory cache.
    All subsequent calls slice from cache — no disk reads.

    dataset_fraction applies to training pool only.
    Val split is always the full fixed set from data/val_splits/.

    Args:
        dataset:          'cifar10' | 'cifar100' | 'stl10'
        data_path:        path to raw data directory
        dataset_fraction: fraction of training pool to use (0, 1]
        batch_size:       loader batch size
        seed:             random seed for subsampling training pool

    Returns:
        (train_loader, val_loader, test_loader, n_train)
    """
    dataset = dataset.lower()

    builders = {
        "cifar10":  lambda: _build_cifar10_cache(data_path),
        "cifar100": lambda: _build_cifar100_cache(data_path),
        "stl10":    lambda: _build_stl10_cache(data_path),
    }
    loaders_fn = {
        "cifar10":  _loaders_cifar10,
        "cifar100": _loaders_cifar100,
        "stl10":    _loaders_stl10,
    }

    if dataset not in builders:
        raise ValueError(
            f"Unknown vision dataset: '{dataset}'. "
            f"Supported: cifar10, cifar100, stl10"
        )

    if dataset not in _cache:
        builders[dataset]()

    return loaders_fn[dataset](dataset_fraction, batch_size, seed)