"""[nlp_loader). — unified NLP data loader for [t-bound).

Val splits loaded from data/val_splits/*.npz — never regenerated inline.
BERT tokenizer loaded from data/raw/bert-base-uncased/ (offline).
Module-level cache avoids re-tokenizing on repeated calls.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from pathlib import Path
from typing import Dict, Tuple, Optional

VAL_SPLITS_DIR = Path(__file__).parent / "val_splits"

# module-level tokenization cache — one-time cost per dataset
_CACHE: Dict[str, dict] = {}


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


# ── tokenized dataset ─────────────────────────────────────────────────────────

class _TokenizedDataset(Dataset):
    def __init__(self, input_ids: torch.Tensor,
                 attention_mask: torch.Tensor,
                 labels: torch.Tensor):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }, self.labels[idx]


def _make_loaders(input_ids, attention_mask, labels,
                  train_idx, val_idx, n_train, batch_size):
    full_ds = _TokenizedDataset(input_ids, attention_mask, labels)
    train_ds = Subset(full_ds, train_idx[:n_train])
    val_ds   = Subset(full_ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size * 2,
                              shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader


# ── tokenizer helper ──────────────────────────────────────────────────────────

def _get_tokenizer(tokenizer_path: str):
    from transformers import BertTokenizerFast
    return BertTokenizerFast.from_pretrained(tokenizer_path)


def _tokenize_texts(texts, tokenizer, max_length: int = 128):
    enc = tokenizer(
        texts, padding="max_length", truncation=True,
        max_length=max_length, return_tensors="pt"
    )
    return enc["input_ids"], enc["attention_mask"]


# ── Yahoo Answers ─────────────────────────────────────────────────────────────
# Format: parquet files (train-00000-of-00002.parquet, train-00001-of-00002.parquet,
#                        test-00000-of-00001.parquet)
# Columns: id, topic, question_title, question_content, best_answer
# Labels:  topic (0-indexed, 0-9, 10 classes)

def _load_yahoo(data_path: str, tokenizer_path: str,
                dataset_fraction: float, batch_size: int,
                seed: int) -> Tuple:
    cache_key = f"yahoo_{dataset_fraction}_{seed}"
    if cache_key not in _CACHE:
        import pandas as pd
        from pathlib import Path
        p = Path(data_path)

        # load all train parquet shards
        train_shards = sorted(p.glob("train-*.parquet"))
        if not train_shards:
            raise FileNotFoundError(
                f"No train parquet files found in {data_path}\n"
                f"Expected: train-00000-of-*.parquet, train-00001-of-*.parquet ..."
            )
        train_df = pd.concat([pd.read_parquet(f) for f in train_shards],
                             ignore_index=True)

        test_shards = sorted(p.glob("test-*.parquet"))
        if not test_shards:
            raise FileNotFoundError(f"No test parquet files found in {data_path}")
        test_df = pd.concat([pd.read_parquet(f) for f in test_shards],
                            ignore_index=True)

        tokenizer = _get_tokenizer(tokenizer_path)

        def _process(df):
            texts  = [f"{q} {a}" for q, a in
                      zip(df["question_title"].tolist(),
                          df["best_answer"].tolist())]
            labels = torch.tensor(df["topic"].tolist(), dtype=torch.long)
            ids, mask = _tokenize_texts(texts, tokenizer)
            return ids, mask, labels

        tr_ids, tr_mask, tr_lbl = _process(train_df)
        _CACHE[cache_key] = {
            "train_ids": tr_ids, "train_mask": tr_mask, "train_lbl": tr_lbl,
        }
        te_ids, te_mask, te_lbl = _process(test_df)
        _CACHE["yahoo_test"] = {
            "ids": te_ids, "mask": te_mask, "lbl": te_lbl
        }
        print(f"  [yahoo] Loaded {len(train_df):,} train + {len(test_df):,} test.", flush=True)

    c = _CACHE[cache_key]
    train_pool_idx, val_idx = _load_val_split("yahoo")
    n_train = int(len(train_pool_idx) * dataset_fraction)
    rng = np.random.RandomState(seed)
    chosen = rng.choice(train_pool_idx, n_train, replace=False)

    train_loader, val_loader = _make_loaders(
        c["train_ids"], c["train_mask"], c["train_lbl"],
        chosen, val_idx, len(chosen), batch_size
    )
    tc = _CACHE.get("yahoo_test", {})
    test_ds = _TokenizedDataset(tc["ids"], tc["mask"], tc["lbl"])
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2,
                             shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader, len(chosen)


# ── AG News ───────────────────────────────────────────────────────────────────
# Format: CSV files (train.csv, test.csv)
# Columns: Class Index, Title, Description
# Labels:  Class Index (1-indexed, 1-4) → subtract 1 → (0-3, 4 classes)

def _load_agnews(data_path: str, tokenizer_path: str,
                 dataset_fraction: float, batch_size: int,
                 seed: int) -> Tuple:
    cache_key = f"agnews_{dataset_fraction}_{seed}"
    if cache_key not in _CACHE:
        import pandas as pd
        from pathlib import Path
        p = Path(data_path)

        train_df = pd.read_csv(p / "train.csv")
        test_df  = pd.read_csv(p / "test.csv")

        tokenizer = _get_tokenizer(tokenizer_path)

        def _process(df):
            texts  = [f"{t} {d}" for t, d in
                      zip(df["Title"].tolist(),
                          df["Description"].tolist())]
            labels = torch.tensor(
                [l - 1 for l in df["Class Index"].tolist()], dtype=torch.long
            )
            ids, mask = _tokenize_texts(texts, tokenizer)
            return ids, mask, labels

        tr_ids, tr_mask, tr_lbl = _process(train_df)
        _CACHE[cache_key] = {
            "train_ids": tr_ids, "train_mask": tr_mask, "train_lbl": tr_lbl,
        }
        te_ids, te_mask, te_lbl = _process(test_df)
        _CACHE["agnews_test"] = {"ids": te_ids, "mask": te_mask, "lbl": te_lbl}
        print(f"  [agnews] Loaded {len(train_df):,} train + {len(test_df):,} test.", flush=True)

    c = _CACHE[cache_key]
    train_pool_idx, val_idx = _load_val_split("agnews")
    n_train = int(len(train_pool_idx) * dataset_fraction)
    rng = np.random.RandomState(seed)
    chosen = rng.choice(train_pool_idx, n_train, replace=False)

    train_loader, val_loader = _make_loaders(
        c["train_ids"], c["train_mask"], c["train_lbl"],
        chosen, val_idx, len(chosen), batch_size
    )
    tc = _CACHE.get("agnews_test", {})
    test_ds = _TokenizedDataset(tc["ids"], tc["mask"], tc["lbl"])
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2,
                             shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader, len(chosen)


# ── DBpedia ───────────────────────────────────────────────────────────────────
# Format: parquet files (train.parquet, test.parquet)
# Columns: label, title, content
# Labels:  label (0-indexed, 0-13, 14 classes)

def _load_dbpedia(data_path: str, tokenizer_path: str,
                  dataset_fraction: float, batch_size: int,
                  seed: int) -> Tuple:
    cache_key = f"dbpedia_{dataset_fraction}_{seed}"
    if cache_key not in _CACHE:
        import pandas as pd
        from pathlib import Path
        p = Path(data_path)

        train_df = pd.read_parquet(p / "train.parquet")
        test_df  = pd.read_parquet(p / "test.parquet")

        tokenizer = _get_tokenizer(tokenizer_path)

        def _process(df):
            texts  = [f"{t} {c}" for t, c in
                      zip(df["title"].tolist(),
                          df["content"].tolist())]
            labels = torch.tensor(df["label"].tolist(), dtype=torch.long)
            ids, mask = _tokenize_texts(texts, tokenizer)
            return ids, mask, labels

        tr_ids, tr_mask, tr_lbl = _process(train_df)
        _CACHE[cache_key] = {
            "train_ids": tr_ids, "train_mask": tr_mask, "train_lbl": tr_lbl,
        }
        te_ids, te_mask, te_lbl = _process(test_df)
        _CACHE["dbpedia_test"] = {
            "ids": te_ids, "mask": te_mask, "lbl": te_lbl
        }
        print(f"  [dbpedia] Loaded {len(train_df):,} train + {len(test_df):,} test.", flush=True)

    c = _CACHE[cache_key]
    train_pool_idx, val_idx = _load_val_split("dbpedia")
    n_train = int(len(train_pool_idx) * dataset_fraction)
    rng = np.random.RandomState(seed)
    chosen = rng.choice(train_pool_idx, n_train, replace=False)

    train_loader, val_loader = _make_loaders(
        c["train_ids"], c["train_mask"], c["train_lbl"],
        chosen, val_idx, len(chosen), batch_size
    )
    tc = _CACHE.get("dbpedia_test", {})
    test_ds = _TokenizedDataset(tc["ids"], tc["mask"], tc["lbl"])
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2,
                             shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader, len(chosen)


# ── public dispatcher ─────────────────────────────────────────────────────────

def load_nlp(dataset: str, data_path: str,
             tokenizer_path: str = "data/raw/bert-base-uncased",
             dataset_fraction: float = 1.0,
             batch_size: int = 32,
             seed: int = 42) -> Tuple:
    """
    Unified NLP loader. Returns (train_loader, val_loader, test_loader, n_train).

    dataset_fraction applies to the training pool only.
    Val split is always the full fixed val set from val_splits/.

    Supported datasets: yahoo, agnews, dbpedia
    """
    dataset = dataset.lower()
    if dataset == "yahoo":
        return _load_yahoo(data_path, tokenizer_path,
                           dataset_fraction, batch_size, seed)
    elif dataset == "agnews":
        return _load_agnews(data_path, tokenizer_path,
                            dataset_fraction, batch_size, seed)
    elif dataset == "dbpedia":
        return _load_dbpedia(data_path, tokenizer_path,
                             dataset_fraction, batch_size, seed)
    else:
        raise ValueError(f"Unknown NLP dataset: {dataset}. "
                         f"Supported: yahoo, agnews, dbpedia")