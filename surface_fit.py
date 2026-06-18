"""[bootstrap). — bootstrap uncertainty quantification for [t-bound)."""

import numpy as np
from typing import Callable, Dict, Optional, Tuple


def extrapolation_ci_multiplier(extrapolation_ratio: float) -> float:
    """
    Continuous CI inflation based on how far the customer is subsampling.

    extrapolation_ratio = dataset_size / full_dataset_size
        1.0   → no subsampling, no inflation (benchmark customers)
        0.05  → 5% subsample, mild inflation
        0.01  → 1% subsample, moderate inflation
        0.001 → 0.1% subsample, heavy inflation

    Formula: 1.0 / sqrt(extrapolation_ratio)
    Statistical justification: standard error scales as 1/sqrt(n),
    so halving the effective dataset doubles uncertainty.

    Examples:
        ratio=1.0   → multiplier=1.00 (no change)
        ratio=0.25  → multiplier=2.00
        ratio=0.05  → multiplier=4.47
        ratio=0.01  → multiplier=10.0
    """
    ratio = max(extrapolation_ratio, 1e-6)
    return float(1.0 / np.sqrt(ratio))


class BootstrapUncertainty:
    """
    Bootstrap confidence intervals for scaling law parameters.

    Given observed (x, y) pairs and a fitting function,
    resamples with replacement n_bootstrap times and reports
    parameter distributions and CI on optimal_value.

    If extrapolation_ratio < 1.0, CI bands are inflated continuously
    using extrapolation_ci_multiplier() to reflect reduced effective
    dataset coverage.
    """

    def __init__(self, n_bootstrap: int = 200, ci_level: float = 0.95,
                 seed: int = 42):
        self.n_bootstrap = n_bootstrap
        self.ci_level = ci_level
        self.rng = np.random.RandomState(seed)

    def compute_ci(
        self,
        x: np.ndarray,
        y: np.ndarray,
        fit_fn: Callable,
        optimal_fn: Callable,
        extrapolation_ratio: float = 1.0,
    ) -> Dict:
        """
        Args:
            x:                   independent variable (e.g. params counts)
            y:                   dependent variable (e.g. val_accuracy)
            fit_fn:              function(x, y) → fit_result dict
            optimal_fn:          function(fit_result) → scalar optimal value
            extrapolation_ratio: dataset_size / full_dataset_size.
                                 1.0 for benchmark customers (no inflation).
                                 < 1.0 inflates CI bands continuously.

        Returns dict with:
            ci_lower, ci_upper, success, optimal_samples,
            ci_multiplier, extrapolation_ratio
        """
        n = len(x)
        optimal_samples = []

        for _ in range(self.n_bootstrap):
            idx = self.rng.randint(0, n, size=n)
            x_boot = x[idx]
            y_boot = y[idx]
            try:
                result = fit_fn(x_boot, y_boot)
                opt = optimal_fn(result)
                if opt is not None and np.isfinite(opt):
                    optimal_samples.append(opt)
            except Exception:
                continue

        if len(optimal_samples) < self.n_bootstrap * 0.5:
            return {
                "ci_lower":           None,
                "ci_upper":           None,
                "success":            False,
                "optimal_samples":    [],
                "ci_multiplier":      None,
                "extrapolation_ratio": extrapolation_ratio,
            }

        alpha_tail = (1 - self.ci_level) / 2
        ci_lower_raw = float(np.quantile(optimal_samples, alpha_tail))
        ci_upper_raw = float(np.quantile(optimal_samples, 1 - alpha_tail))

        # apply continuous CI inflation for subsampled datasets
        multiplier = extrapolation_ci_multiplier(extrapolation_ratio)
        if multiplier > 1.0:
            center   = float(np.median(optimal_samples))
            ci_lower = center - (center - ci_lower_raw) * multiplier
            ci_upper = center + (ci_upper_raw - center) * multiplier
        else:
            ci_lower = ci_lower_raw
            ci_upper = ci_upper_raw

        return {
            "ci_lower":            ci_lower,
            "ci_upper":            ci_upper,
            "success":             True,
            "optimal_samples":     optimal_samples,
            "ci_multiplier":       round(multiplier, 3),
            "extrapolation_ratio": extrapolation_ratio,
        }