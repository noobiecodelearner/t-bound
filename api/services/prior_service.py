"""[api/services/prior_service). — prior database lookup."""

import csv
import os
from typing import Dict

_prior_cache: Dict = {}
_literature_cache = []

_PRIOR_CSV = os.path.join(os.path.dirname(__file__), "../../prior/prior.csv")
_LIT_CSV = os.path.join(os.path.dirname(__file__), "../../prior/literature_values.csv")

_GLOBAL_FALLBACK = {
    "alpha_mean": 0.30,
    "alpha_std": 0.15,
    "beta_mean": 0.15,
    "beta_std": 0.08,
    "gamma_mean": 0.10,
    "gamma_std": 0.05,
    "delta_mean": 0.30,
    "delta_std": 0.15,
    "n_source_curves": 0,
    "source": "global_fallback",
    "confidence_in_prior": "low",
}


def _load_prior():
    global _prior_cache
    if _prior_cache:
        return _prior_cache
    path = os.path.abspath(_PRIOR_CSV)
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (
                row.get("architecture_family", ""),
                row.get("domain", ""),
                row.get("dataset_size_regime", ""),
                row.get("sweep_type", ""),
            )
            _prior_cache[key] = row
    return _prior_cache


def _load_literature():
    global _literature_cache
    if _literature_cache:
        return _literature_cache
    path = os.path.abspath(_LIT_CSV)
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        _literature_cache = list(reader)
    return _literature_cache


def _row_to_prior(row: dict, source: str) -> dict:
    def _f(key, default=0.0):
        v = row.get(key, "")
        try:
            return float(v) if v != "" else default
        except (ValueError, TypeError):
            return default

    n_src = row.get("n_source_curves", "0")
    try:
        n_src = int(float(n_src))
    except (ValueError, TypeError):
        n_src = 0

    confidence = "low"
    if n_src >= 5:
        confidence = "high"
    elif n_src >= 2:
        confidence = "medium"

    return {
        "alpha_mean": _f("alpha_mean", 0.30),
        "alpha_std": _f("alpha_std", 0.15),
        "beta_mean": _f("beta_mean", 0.15),
        "beta_std": _f("beta_std", 0.08),
        "gamma_mean": _f("gamma_mean", 0.10),
        "gamma_std": _f("gamma_std", 0.05),
        "delta_mean": _f("delta_mean", 0.30),
        "delta_std": _f("delta_std", 0.15),
        "n_source_curves": n_src,
        "source": source,
        "confidence_in_prior": confidence,
    }


def get_prior(
    architecture_family: str,
    domain: str,
    dataset_size_regime: str = "medium",
    sweep_type: str = "n_d_lr_grid",
) -> dict:
    prior = _load_prior()

    # 1. Exact match
    key = (architecture_family, domain, dataset_size_regime, sweep_type)
    if key in prior:
        return _row_to_prior(prior[key], "internal")

    # 2. Family + domain (drop regime)
    for k, v in prior.items():
        if k[0] == architecture_family and k[1] == domain:
            return _row_to_prior(v, "internal")

    # 3. Domain only
    for k, v in prior.items():
        if k[1] == domain:
            return _row_to_prior(v, "internal")

    # 4. Literature values
    lit = _load_literature()
    for row in lit:
        if row.get("architecture_family") == architecture_family and row.get("domain") == domain:
            r = dict(row)
            r["n_source_curves"] = "1"
            return _row_to_prior(r, "literature")

    for row in lit:
        if row.get("domain") == domain:
            r = dict(row)
            r["n_source_curves"] = "1"
            return _row_to_prior(r, "literature")

    # 5. Global fallback
    return dict(_GLOBAL_FALLBACK)
