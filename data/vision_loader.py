"""[vision_loader). — unified vision data loader for [t-bound).

All normalization constants are defined once here.
Val splits are loaded from data/val_splits/*.npz — never regenerated inline.
Run scripts/generate_val_splits.py once before any experiments.
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import datasets, transforms
from pathlib import Path
from typing import Tuple

# ── normalization constants — defined once, imported everywhere ───────────────
# These are the exact values used in all scaling experiments.
# validate_holdout.py imports from here. Never hardcode elsewhere.

NORMALIZATION_STATS = {
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


# ── val split loader ──────────────────────────────────────────────────────────

def _load_val_split(dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load saved val split indices from data/val_splits/.
    Raises FileNotFoundError if generate_val_splits.py has not been run.
    """
    path = VAL_SPLITS_DIR / f"{dataset_name}_seed42.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Val split not found: {path}\n"
            f"Run: python scripts/generate_val_splits.py"
        )
    data = np.load(path)
    return data["train_idx"], data["val_idx"]


# ── dataset helpers ───────────────────────────────────────────────────────────

class _ArrayDataset(Dataset):
    """Wraps numpy arrays as a torch Dataset."""
    def __init__(self, images: np.ndarray, labels: np.ndarray,
                 transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = int(self.labels[idx])
        if self.transform:
            import PIL.Image
            img = PIL.Image.fromarray(img)
            img = self.transform(img)
        else:
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return img, label


# ── CIFAR-10 raw pickle loader ────────────────────────────────────────────────

def _load_cifar10_raw(data_path: str) -> Tuple[np.ndarray, np.ndarray,
                                                np.ndarray, np.ndarray]:
    """Load CIFAR-10 from raw pickle batches."""
    def _load_batch(fpath):
        with open(fpath, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        images = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = np.array(d[b"labels"])
        return images, labels

    train_images, train_labels = [], []
    for i in range(1, 6):
        batch_path = os.path.join(data_path, f"data_batch_{i}")
        imgs, lbls = _load_batch(batch_path)
        train_images.append(imgs)
        train_labels.append(lbls)
    train_images = np.concatenate(train_images)
    train_labels = np.concatenate(train_labels)

    test_images, test_labels = _load_batch(
        os.path.join(data_path, "test_batch")
    )
    return train_images, train_labels, test_images, test_labels


def _load_cifar10(data_path: str, dataset_fraction: float,
                  batch_size: int, seed: int) -> Tuple:
    stats = NORMALIZATION_STATS["cifar10"]
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(stats["mean"], stats["std"]),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(stats["mean"], stats["std"]),
    ])

    all_train_img, all_train_lbl, test_img, test_lbl = \
        _load_cifar10_raw(data_path)

    train_pool_idx, val_idx = _load_val_split("cifar10")

    # apply dataset fraction to training pool only
    n_train = int(len(train_pool_idx) * dataset_fraction)
    rng = np.random.RandomState(seed)
    chosen_train_idx = rng.choice(train_pool_idx, n_train, replace=False)

    train_ds = _ArrayDataset(all_train_img[chosen_train_idx],
                             all_train_lbl[chosen_train_idx], train_transform)
    val_ds   = _ArrayDataset(all_train_img[val_idx],
                             all_train_lbl[val_idx], val_transform)
    test_ds  = _ArrayDataset(test_img, test_lbl, val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size * 2,
                              shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size * 2,
                              shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader, len(chosen_train_idx)


# ── CIFAR-100 ─────────────────────────────────────────────────────────────────

def _load_cifar100(data_path: str, dataset_fraction: float,
                   batch_size: int, seed: int) -> Tuple:
    stats = NORMALIZATION_STATS["cifar100"]
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(stats["mean"], stats["std"]),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(stats["mean"], stats["std"]),
    ])

    full_train = datasets.CIFAR100(data_path, train=True, download=False,
                                   transform=train_transform)
    full_test  = datasets.CIFAR100(data_path, train=False, download=False,
                                   transform=val_transform)
    full_val   = datasets.CIFAR100(data_path, train=True, download=False,
                                   transform=val_transform)

    train_pool_idx, val_idx = _load_val_split("cifar100")
    n_train = int(len(train_pool_idx) * dataset_fraction)
    rng = np.random.RandomState(seed)
    chosen = rng.choice(train_pool_idx, n_train, replace=False)

    train_loader = DataLoader(Subset(full_train, chosen),
                              batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(Subset(full_val, val_idx),
                              batch_size=batch_size * 2, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(full_test,
                              batch_size=batch_size * 2, shuffle=False,
                              num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader, len(chosen)


# ── STL-10 ────────────────────────────────────────────────────────────────────

def _load_stl10(data_path: str, dataset_fraction: float,
                batch_size: int, seed: int) -> Tuple:
    stats = NORMALIZATION_STATS["stl10"]
    # resize to 32x32 to match CNN input
    train_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(stats["mean"], stats["std"]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(stats["mean"], stats["std"]),
    ])

    full_train = datasets.STL10(data_path, split="train", download=False,
                                transform=train_transform)
    full_test  = datasets.STL10(data_path, split="test",  download=False,
                                transform=val_transform)
    full_val   = datasets.STL10(data_path, split="train", download=False,
                                transform=val_transform)

    train_pool_idx, val_idx = _load_val_split("stl10")
    n_train = int(len(train_pool_idx) * dataset_fraction)
    rng = np.random.RandomState(seed)
    chosen = rng.choice(train_pool_idx, n_train, replace=False)

    train_loader = DataLoader(Subset(full_train, chosen),
                              batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(Subset(full_val, val_idx),
                              batch_size=batch_size * 2, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(full_test,
                              batch_size=batch_size * 2, shuffle=False,
                              num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader, len(chosen)


# ── public dispatcher ─────────────────────────────────────────────────────────

def load_vision(dataset: str, data_path: str,
                dataset_fraction: float = 1.0,
                batch_size: int = 128,
                seed: int = 42) -> Tuple:
    """
    Unified vision loader. Returns (train_loader, val_loader, test_loader, n_train).

    dataset_fraction applies to the training pool only.
    Val split is always the full fixed val set from val_splits/.

    Supported datasets: cifar10, cifar100, stl10
    """
    dataset = dataset.lower()
    if dataset == "cifar10":
        return _load_cifar10(data_path, dataset_fraction, batch_size, seed)
    elif dataset == "cifar100":
        return _load_cifar100(data_path, dataset_fraction, batch_size, seed)
    elif dataset == "stl10":
        return _load_stl10(data_path, dataset_fraction, batch_size, seed)
    else:
        raise ValueError(f"Unknown vision dataset: {dataset}. "
                         f"Supported: cifar10, cifar100, stl10")
