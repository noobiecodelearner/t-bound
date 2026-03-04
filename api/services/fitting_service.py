"""[api/services/fitting_service). — core scaling law fitting."""

import numpy as np
from sqlalchemy.orm import Session

from api.db import crud
from api.services.prior_service import get_prior


def fit_project(project_id: str, db: Session) -> dict:
    """Fit scaling curve for a project. Called after every logged run."""
    runs = crud.get_runs_for_project(db, project_id)
    project = crud.get_project(db, project_id)

    arch = project.architecture_family if project else "cnn"
    domain = project.domain if project else "vision"
    prior = get_prior(arch, domain)

    n_runs = len(runs)

    def _store(result: dict) -> dict:
        crud.create_or_update_fit(
            db,
            project_id=project_id,
            alpha=result.get("alpha"),
            a=result.get("a"),
            b=result.get("b"),
            r2=result.get("r2"),
            mae=result.get("mae"),
            ci_lower=result.get("ci_lower"),
            ci_upper=result.get("ci_upper"),
            confidence=result.get("confidence"),
            n_runs_used=n_runs,
        )
        crud.update_project_confidence(db, project_id, result["confidence"])
        return result

    # ── 0 runs ────────────────────────────────────────────────────────────────
    if n_runs == 0:
        return _store({
            "alpha": prior["alpha_mean"],
            "a": None,
            "b": None,
            "r2": None,
            "mae": None,
            "ci_lower": prior["alpha_mean"] - 2 * prior["alpha_std"],
            "ci_upper": prior["alpha_mean"] + 2 * prior["alpha_std"],
            "confidence": "very_low",
            "n_runs": 0,
            "prior_weight": 1.0,
        })

    params = np.array([r.params for r in runs], dtype=float)
    accuracies = np.array([r.val_accuracy for r in runs], dtype=float)

    # ── 1–2 runs ──────────────────────────────────────────────────────────────
    if n_runs <= 2:
        alpha = prior["alpha_mean"]
        # rough estimate of a from observed max accuracy
        a_est = min(float(np.max(accuracies)) + 0.05, 0.999)
        return _store({
            "alpha": alpha,
            "a": a_est,
            "b": None,
            "r2": None,
            "mae": None,
            "ci_lower": alpha - 1.5 * prior["alpha_std"],
            "ci_upper": alpha + 1.5 * prior["alpha_std"],
            "confidence": "low",
            "n_runs": n_runs,
            "prior_weight": 0.8,
        })

    # ── 3–5 runs ──────────────────────────────────────────────────────────────
    try:
        from scaling.surface_fit import fit_model_size
        from scaling.bootstrap import BootstrapUncertainty

        fit_result = fit_model_size(params, accuracies)
        alpha_data = fit_result["alpha"]
        a = fit_result["a"]
        b = fit_result["b"]
        r2 = fit_result.get("r2")
        mae = fit_result.get("mae")

        if n_runs <= 5:
            w = max(0.0, (6 - n_runs) / 6)
            alpha_final = (1 - w) * alpha_data + w * prior["alpha_mean"]
            confidence = "medium"
        else:
            w = 0.0
            alpha_final = alpha_data
            confidence = "high"

        # Bootstrap CI
        ci_lower, ci_upper = None, None
        if n_runs >= 3:
            try:
                bs = BootstrapUncertainty(n_bootstrap=200)
                target = float(np.mean(accuracies)) + 0.05

                def _fit_fn(x, y):
                    return fit_model_size(x, y)

                def _opt_fn(res):
                    fn = res.get("optimal_n_fn")
                    if fn:
                        v = fn(target)
                        return v
                    return None

                ci_result = bs.compute_ci(params, accuracies, _fit_fn, _opt_fn)
                if ci_result["success"]:
                    ci_lower = ci_result["ci_lower"]
                    ci_upper = ci_result["ci_upper"]
            except Exception:
                pass

        if ci_lower is None:
            ci_lower = alpha_final - prior["alpha_std"]
            ci_upper = alpha_final + prior["alpha_std"]

        return _store({
            "alpha": alpha_final,
            "a": float(a),
            "b": float(b),
            "r2": float(r2) if r2 is not None else None,
            "mae": float(mae) if mae is not None else None,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "confidence": confidence,
            "n_runs": n_runs,
            "prior_weight": w if n_runs <= 5 else 0.0,
        })

    except Exception as e:
        # Fallback to prior on any fitting failure
        alpha = prior["alpha_mean"]
        return _store({
            "alpha": alpha,
            "a": None,
            "b": None,
            "r2": None,
            "mae": None,
            "ci_lower": alpha - 2 * prior["alpha_std"],
            "ci_upper": alpha + 2 * prior["alpha_std"],
            "confidence": "very_low",
            "n_runs": n_runs,
            "prior_weight": 1.0,
        })
