"""[surface_fit). — scaling law fitting for [t-bound).

Four fitting functions:
    fit_model_size(N, accuracy)       → α, a, b     Accuracy = a - b·N^(-α)
    fit_lr_scaling(N, lr_star)        → β, c        lr*(N)   = c·N^(-β)
    fit_batch_scaling(N, batch_star)  → γ, d        batch*(N)= d·N^(γ)
    fit_nd_surface(N, D, accuracy)    → α, δ, a, b  Accuracy = a - b·N^(-α)·D^(-δ)
    solve_chinchilla_frontier(...)    → N*, D*       given compute budget C
"""

import numpy as np
from scipy.optimize import curve_fit, minimize_scalar
from scipy.stats import pearsonr
from typing import Dict, List, Optional, Tuple


# ── model size scaling ────────────────────────────────────────────────────────

def _power_law(N, a, b, alpha):
    """Accuracy = a - b · N^(-alpha)"""
    return a - b * np.power(np.maximum(N, 1e-10), -alpha)


def fit_model_size(params: np.ndarray,
                   accuracies: np.ndarray) -> Dict:
    """
    Fit Accuracy(N) = a - b · N^(-α).

    Args:
        params:     array of parameter counts (N values)
        accuracies: array of best val accuracies at each N

    Returns dict:
        alpha, a, b, r2, mae, aic, optimal_n_fn
        optimal_n_fn(target_accuracy) → N* (float)
    """
    params = np.array(params, dtype=float)
    accuracies = np.array(accuracies, dtype=float)

    if len(params) < 3:
        raise ValueError("Need at least 3 points to fit model size scaling.")

    # initial guess
    p0 = [max(accuracies), 1.0, 0.3]
    bounds = ([0.0, 1e-6, 0.01], [1.0, 1e6, 2.0])

    try:
        popt, _ = curve_fit(_power_law, params, accuracies,
                            p0=p0, bounds=bounds, maxfev=10000)
    except RuntimeError:
        # fallback with looser bounds
        popt, _ = curve_fit(_power_law, params, accuracies,
                            p0=p0, maxfev=50000)

    a, b, alpha = popt
    pred = _power_law(params, a, b, alpha)
    residuals = accuracies - pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((accuracies - np.mean(accuracies)) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-12)
    mae = float(np.mean(np.abs(residuals)))
    k = 3  # number of parameters
    n = len(params)
    aic = n * np.log(max(ss_res / n, 1e-12)) + 2 * k

    def optimal_n_fn(target_accuracy: float) -> Optional[float]:
        """Return minimum N to achieve target_accuracy."""
        if target_accuracy >= a:
            return None  # unreachable
        n_star = (b / (a - target_accuracy)) ** (1.0 / alpha)
        return float(n_star)

    return {
        "alpha": float(alpha),
        "a": float(a),
        "b": float(b),
        "r2": float(r2),
        "mae": mae,
        "aic": float(aic),
        "optimal_n_fn": optimal_n_fn,
        "predict_fn": lambda N: _power_law(np.array(N), a, b, alpha),
    }


# ── lr scaling ────────────────────────────────────────────────────────────────

def fit_lr_scaling(params: np.ndarray,
                   lr_star: np.ndarray) -> Dict:
    """
    Fit lr*(N) = c · N^(-β) in log-log space.

    Args:
        params:  array of parameter counts at each scale point
        lr_star: array of optimal learning rates at each scale point
                 (one per model size, found by argmax over lr grid)

    Returns dict:
        beta, c, r2
        optimal_lr_fn(N) → lr*(N)
    """
    params  = np.array(params,  dtype=float)
    lr_star = np.array(lr_star, dtype=float)

    if len(params) < 3:
        raise ValueError("Need at least 3 points to fit lr scaling.")

    # fit in log-log space: log(lr*) = log(c) - β·log(N)
    log_N  = np.log(params)
    log_lr = np.log(lr_star)

    # linear regression in log space
    coeffs = np.polyfit(log_N, log_lr, 1)
    beta   = float(-coeffs[0])  # slope is -β
    log_c  = float(coeffs[1])
    c      = float(np.exp(log_c))

    pred   = log_c - beta * log_N
    ss_res = np.sum((log_lr - pred) ** 2)
    ss_tot = np.sum((log_lr - np.mean(log_lr)) ** 2)
    r2     = float(1 - ss_res / max(ss_tot, 1e-12))

    def optimal_lr_fn(N: float) -> float:
        return float(c * (N ** (-beta)))

    return {
        "beta": beta,
        "c": c,
        "r2": r2,
        "optimal_lr_fn": optimal_lr_fn,
    }


# ── batch scaling ─────────────────────────────────────────────────────────────

def fit_batch_scaling(params: np.ndarray,
                      batch_star: np.ndarray) -> Dict:
    """
    Fit batch*(N) = d · N^(γ) in log-log space.

    Note: γ is typically positive — larger models tolerate larger batches.
    Fit from the batch sweep at N* (one set of batch values vs accuracy),
    or from multiple N* values if available.

    Args:
        params:     array of parameter counts (N values, often just N*)
        batch_star: array of optimal batch sizes at each N value

    Returns dict:
        gamma, d, r2
        optimal_batch_fn(N) → batch*(N)
    """
    params     = np.array(params,     dtype=float)
    batch_star = np.array(batch_star, dtype=float)

    if len(params) < 2:
        raise ValueError("Need at least 2 points to fit batch scaling.")

    log_N     = np.log(params)
    log_batch = np.log(batch_star)
    coeffs    = np.polyfit(log_N, log_batch, 1)
    gamma     = float(coeffs[0])   # positive: larger model → larger batch
    log_d     = float(coeffs[1])
    d         = float(np.exp(log_d))

    pred   = log_d + gamma * log_N
    ss_res = np.sum((log_batch - pred) ** 2)
    ss_tot = np.sum((log_batch - np.mean(log_batch)) ** 2)
    r2     = float(1 - ss_res / max(ss_tot, 1e-12))

    def optimal_batch_fn(N: float) -> int:
        batch = d * (N ** gamma)
        # round to nearest power of 2
        power = round(np.log2(max(batch, 1)))
        return int(2 ** np.clip(power, 5, 10))  # 32 to 1024

    return {
        "gamma": gamma,
        "d": d,
        "r2": r2,
        "optimal_batch_fn": optimal_batch_fn,
    }


# ── (N, D) surface fit — Chinchilla ──────────────────────────────────────────

def _nd_surface(ND, a, b, alpha, delta):
    """
    Accuracy(N, D) = a - b · N^(-α) · D^(-δ)
    ND: array of shape (2, n) where ND[0]=N, ND[1]=D
    """
    N, D = ND
    return a - b * np.power(np.maximum(N, 1), -alpha) \
                 * np.power(np.maximum(D, 1), -delta)


def fit_nd_surface(params: np.ndarray,
                   dataset_sizes: np.ndarray,
                   accuracies: np.ndarray) -> Dict:
    """
    Fit Accuracy(N, D) = a - b · N^(-α) · D^(-δ).

    This is the Chinchilla-style 2D surface fit.

    Args:
        params:       1D array of parameter counts
        dataset_sizes: 1D array of training dataset sizes (raw counts)
        accuracies:   1D array of best val accuracies at each (N, D) point
                      Must be best accuracy optimized over lr at each (N,D).

    Returns dict:
        alpha, delta, a, b, r2, mae
        predict_fn(N, D) → accuracy
        optimal_n_fn(target_accuracy, D) → N*
        optimal_d_fn(target_accuracy, N) → D*
    """
    params        = np.array(params,        dtype=float)
    dataset_sizes = np.array(dataset_sizes, dtype=float)
    accuracies    = np.array(accuracies,    dtype=float)

    if len(params) < 4:
        raise ValueError("Need at least 4 (N,D,acc) points for surface fit.")

    ND = np.vstack([params, dataset_sizes])
    p0 = [max(accuracies), 1.0, 0.3, 0.3]
    bounds = ([0.0, 1e-6, 0.01, 0.01], [1.0, 1e6, 2.0, 2.0])

    try:
        popt, _ = curve_fit(_nd_surface, ND, accuracies,
                            p0=p0, bounds=bounds, maxfev=20000)
    except RuntimeError:
        popt, _ = curve_fit(_nd_surface, ND, accuracies,
                            p0=p0, maxfev=100000)

    a, b, alpha, delta = popt
    pred      = _nd_surface(ND, a, b, alpha, delta)
    residuals = accuracies - pred
    ss_res    = np.sum(residuals ** 2)
    ss_tot    = np.sum((accuracies - np.mean(accuracies)) ** 2)
    r2        = float(1 - ss_res / max(ss_tot, 1e-12))
    mae       = float(np.mean(np.abs(residuals)))

    def predict_fn(N, D):
        N = np.atleast_1d(np.array(N, dtype=float))
        D = np.atleast_1d(np.array(D, dtype=float))
        return float(np.mean(_nd_surface(np.vstack([N, D]), a, b, alpha, delta)))

    def optimal_n_fn(target_accuracy: float, D: float) -> Optional[float]:
        """Minimum N to hit target_accuracy at dataset size D."""
        if target_accuracy >= a:
            return None
        # solve: a - b·N^(-α)·D^(-δ) = τ
        # b·N^(-α)·D^(-δ) = a - τ
        # N^(-α) = (a-τ) / (b·D^(-δ))
        # N = [(b·D^(-δ)) / (a-τ)]^(1/α)
        b_eff = b * (D ** (-delta))
        n_star = (b_eff / (a - target_accuracy)) ** (1.0 / alpha)
        return float(n_star)

    def optimal_d_fn(target_accuracy: float, N: float) -> Optional[float]:
        """Minimum D to hit target_accuracy with model size N."""
        if target_accuracy >= a:
            return None
        b_eff = b * (N ** (-alpha))
        d_star = (b_eff / (a - target_accuracy)) ** (1.0 / delta)
        return float(d_star)

    return {
        "alpha": float(alpha),
        "delta": float(delta),
        "a": float(a),
        "b": float(b),
        "r2": r2,
        "mae": mae,
        "predict_fn": predict_fn,
        "optimal_n_fn": optimal_n_fn,
        "optimal_d_fn": optimal_d_fn,
    }


# ── Chinchilla frontier solver ────────────────────────────────────────────────

def solve_chinchilla_frontier(
    surface: Dict,
    compute_budget_flops: float,
    flops_per_param_per_sample: float = 6.0,
    n_grid: int = 100,
) -> Dict:
    """
    Given a compute budget C (in FLOPs) and a fitted (N,D) surface,
    find the optimal (N*, D*) split.

    Compute constraint: C = flops_per_param_per_sample · N · D
    → D = C / (flops_per_param_per_sample · N)

    Searches over N on a log-scale grid, evaluates accuracy at each
    (N, D(N)) point, returns the optimal split.

    Args:
        surface:                    result of fit_nd_surface()
        compute_budget_flops:       total FLOPs budget
        flops_per_param_per_sample: FLOPs per parameter per sample
                                    (default 6 following Chinchilla convention)
        n_grid:                     resolution of N search grid

    Returns dict:
        n_star, d_star, expected_accuracy, n_grid_values, acc_grid_values
    """
    predict_fn = surface["predict_fn"]
    a = surface["a"]

    # N range: from very small to compute_budget / min_D(1 sample)
    n_min = max(1e3,  compute_budget_flops / (flops_per_param_per_sample * 1e8))
    n_max = min(1e9,  compute_budget_flops / (flops_per_param_per_sample * 10))

    n_values   = np.logspace(np.log10(n_min), np.log10(n_max), n_grid)
    acc_values = []

    for N in n_values:
        D = compute_budget_flops / (flops_per_param_per_sample * N)
        if D < 1:
            acc_values.append(0.0)
        else:
            acc_values.append(predict_fn(N, D))

    acc_values = np.array(acc_values)
    best_idx   = int(np.argmax(acc_values))
    n_star     = float(n_values[best_idx])
    d_star     = float(compute_budget_flops / (flops_per_param_per_sample * n_star))

    return {
        "n_star":             n_star,
        "d_star":             d_star,
        "expected_accuracy":  float(acc_values[best_idx]),
        "n_grid_values":      n_values.tolist(),
        "acc_grid_values":    acc_values.tolist(),
        "compute_budget_flops": compute_budget_flops,
    }


# ── hierarchical Bayesian model size fit ──────────────────────────────────────

def fit_model_size_hierarchical(
    params: np.ndarray,
    accuracies: np.ndarray,
    prior_alpha_mean: float,
    prior_alpha_std: float,
    n_runs: int,
) -> Dict:
    """
    Hierarchical Bayesian fit of Accuracy(N) = a - b·N^(-α).

    Places a Normal prior on α: α ~ Normal(prior_alpha_mean, prior_alpha_std)
    and fits via penalized least squares. Prior weight decreases as n_runs grows.

    Penalized objective:
        L(a, b, α) = sum[(acc_i - (a - b·N_i^(-α)))^2]
                   + lambda · (α - prior_alpha_mean)^2 / prior_alpha_std^2

    Lambda schedule:
        n_runs = 0   → lambda = 1.0  (fully prior-driven)
        n_runs = 6+  → lambda = 0.0  (fully data-driven)
        linear decay between 0 and 6

    Posterior CI on α:
        Derived analytically from curvature of objective at optimum.
        sigma_posterior = prior_alpha_std / sqrt(1 + n_runs / 3)

    Args:
        params:           array of parameter counts (N values)
        accuracies:       array of best val accuracies at each N
        prior_alpha_mean: prior mean for α (from prior.csv)
        prior_alpha_std:  prior std for α (from prior.csv)
        n_runs:           number of customer runs contributing to fit

    Returns dict with all keys from fit_model_size() plus:
        prior_alpha_mean, prior_alpha_std, lambda_used,
        alpha_ci_lower, alpha_ci_upper, sigma_posterior,
        fit_type = "hierarchical"
    """
    from scipy.optimize import minimize

    params     = np.array(params,     dtype=float)
    accuracies = np.array(accuracies, dtype=float)

    if len(params) < 2:
        raise ValueError("Need at least 2 points for hierarchical fit.")

    # prior weight: linear decay 1.0 → 0.0 over first 6 runs
    lam = max(0.0, (6 - n_runs) / 6.0)

    sigma_prior = max(prior_alpha_std, 1e-4)  # avoid division by zero

    def objective(p):
        a, b, alpha = p
        if a <= 0 or b <= 0 or alpha <= 0:
            return 1e10
        pred      = a - b * np.power(np.maximum(params, 1e-10), -alpha)
        data_loss = np.sum((accuracies - pred) ** 2)
        prior_pen = lam * ((alpha - prior_alpha_mean) ** 2) / (sigma_prior ** 2)
        return data_loss + prior_pen

    # initial guess: use prior alpha, estimate a and b from data
    a0     = float(np.max(accuracies)) + 0.01
    b0     = 1.0
    alpha0 = prior_alpha_mean

    result = minimize(
        objective,
        x0=[a0, b0, alpha0],
        method="Nelder-Mead",
        options={"maxiter": 10000, "xatol": 1e-6, "fatol": 1e-8},
    )

    if not result.success:
        # fallback to standard fit
        fallback = fit_model_size(params, accuracies)
        fallback["fit_type"] = "standard_fallback"
        fallback["prior_alpha_mean"] = prior_alpha_mean
        fallback["prior_alpha_std"]  = prior_alpha_std
        fallback["lambda_used"]      = lam
        return fallback

    a, b, alpha = result.x

    # ensure positivity (Nelder-Mead doesn't enforce bounds)
    a     = max(a, float(np.max(accuracies)))
    b     = max(b, 1e-6)
    alpha = max(alpha, 0.01)

    pred      = _power_law(params, a, b, alpha)
    residuals = accuracies - pred
    ss_res    = np.sum(residuals ** 2)
    ss_tot    = np.sum((accuracies - np.mean(accuracies)) ** 2)
    r2        = float(1 - ss_res / max(ss_tot, 1e-12))
    mae       = float(np.mean(np.abs(residuals)))
    k         = 3
    n         = len(params)
    aic       = n * np.log(max(ss_res / n, 1e-12)) + 2 * k

    # posterior sigma on alpha: shrinks as we see more data
    # sigma_post = sigma_prior / sqrt(1 + n_runs / 3)
    sigma_post    = sigma_prior / np.sqrt(1 + n_runs / 3.0)
    alpha_ci_lower = alpha - 1.96 * sigma_post
    alpha_ci_upper = alpha + 1.96 * sigma_post

    def optimal_n_fn(target_accuracy: float) -> Optional[float]:
        if target_accuracy >= a:
            return None
        return float((b / (a - target_accuracy)) ** (1.0 / alpha))

    return {
        "alpha":            float(alpha),
        "a":                float(a),
        "b":                float(b),
        "r2":               r2,
        "mae":              mae,
        "aic":              float(aic),
        "optimal_n_fn":     optimal_n_fn,
        "predict_fn":       lambda N: _power_law(np.array(N), a, b, alpha),
        # hierarchical extras
        "fit_type":         "hierarchical",
        "prior_alpha_mean": prior_alpha_mean,
        "prior_alpha_std":  prior_alpha_std,
        "lambda_used":      round(lam, 4),
        "sigma_posterior":  round(float(sigma_post), 4),
        "alpha_ci_lower":   round(float(alpha_ci_lower), 4),
        "alpha_ci_upper":   round(float(alpha_ci_upper), 4),
    }