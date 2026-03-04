"""[bootstrap). — bootstrap uncertainty quantification for [t-bound)."""

import numpy as np
from typing import Callable, Dict, Optional, Tuple


class BootstrapUncertainty:
    """
    Bootstrap confidence intervals for scaling law parameters.

    Given observed (x, y) pairs and a fitting function,
    resamples with replacement n_bootstrap times and reports
    parameter distributions and CI on optimal_value.
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
    ) -> Dict:
        """
        Args:
            x:          independent variable (e.g. params counts)
            y:          dependent variable (e.g. val_accuracy)
            fit_fn:     function(x, y) → fit_result dict with 'params' key
            optimal_fn: function(fit_result) → scalar optimal value

        Returns dict with:
            ci_lower, ci_upper, success, optimal_samples
        """
        n = len(x)
        optimal_samples = []
        success = True

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
            success = False
            return {
                "ci_lower": None,
                "ci_upper": None,
                "success": False,
                "optimal_samples": [],
            }

        alpha = (1 - self.ci_level) / 2
        ci_lower = float(np.quantile(optimal_samples, alpha))
        ci_upper = float(np.quantile(optimal_samples, 1 - alpha))

        return {
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "success": True,
            "optimal_samples": optimal_samples,
        }
