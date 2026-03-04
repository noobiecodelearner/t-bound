"""[api/services/recommendation_service). — recommendation engine."""

from typing import Optional
from sqlalchemy.orm import Session

from api.db import crud
from api.services.fitting_service import fit_project


def get_recommendation(
    project_id: str,
    db: Session,
    target_accuracy: float = 0.85,
    compute_budget_hours: Optional[float] = None,
) -> dict:
    """Return recommendation for a project."""
    fit = crud.get_fit_for_project(db, project_id)
    runs = crud.get_runs_for_project(db, project_id)

    if fit is None:
        fit_result = fit_project(project_id, db)
        fit = crud.get_fit_for_project(db, project_id)

    alpha = fit.alpha or 0.30
    a = fit.a
    b = fit.b
    confidence = fit.confidence or "very_low"
    n_runs = fit.n_runs_used or 0

    # prior weight (stored implicitly — estimate from confidence)
    prior_weight_map = {"very_low": 1.0, "low": 0.8, "medium": 0.4, "high": 0.0}
    prior_weight = prior_weight_map.get(confidence, 1.0)

    # N* from scaling law: N* = (b / (a - target))^(1/alpha)
    optimal_n = None
    expected_accuracy = None
    if a is not None and b is not None and a > target_accuracy:
        try:
            optimal_n = int((b / (a - target_accuracy)) ** (1.0 / alpha))
            expected_accuracy = float(a - b * (optimal_n ** -alpha))
        except Exception:
            pass

    if optimal_n is None:
        # Fallback estimate using prior alpha only
        if runs:
            import numpy as np
            max_n = max(r.params for r in runs)
            max_acc = max(r.val_accuracy for r in runs)
            # rough estimate
            if max_acc < target_accuracy:
                scale = (1 - max_acc) / (1 - target_accuracy)
                optimal_n = int(max_n * (scale ** (1.0 / alpha)))
            else:
                optimal_n = max_n
        else:
            optimal_n = 1_000_000
        expected_accuracy = target_accuracy

    # Optimal LR and batch (simple scaling rules from literature)
    # lr*(N) ~ c * N^(-beta), beta ~ 0.15, c calibrated to typical lr range
    optimal_lr = round(0.1 * (optimal_n ** -0.15), 6)
    optimal_batch = max(16, min(512, int(32 * (optimal_n ** 0.10))))

    # CI on N*
    ci_lower = fit.ci_lower
    ci_upper = fit.ci_upper
    if ci_lower is None or ci_upper is None:
        ci_lower = optimal_n * 0.5
        ci_upper = optimal_n * 2.0

    # Compute savings vs max observed params
    compute_saved = 0.0
    if runs:
        max_observed = max(r.params for r in runs)
        if max_observed > 0:
            compute_saved = max(0.0, min(1.0, 1.0 - (optimal_n / max_observed)))

    # Energy / carbon
    try:
        from training.energy import estimate_energy_kwh, estimate_carbon_grams
        energy_saved_kwh = estimate_energy_kwh(optimal_n) * compute_saved
        carbon_saved_g = estimate_carbon_grams(energy_saved_kwh)
    except Exception:
        # simple estimates: ~1e-9 kWh per param, 400g CO2/kWh
        energy_saved_kwh = optimal_n * 1e-9 * compute_saved
        carbon_saved_g = energy_saved_kwh * 400

    message = ""
    if confidence in ("very_low",):
        message = "Log more runs to improve confidence. Recommendation based on prior only."
    elif confidence == "low":
        message = "Confidence is low. Log at least 3 runs for a medium confidence recommendation."

    return {
        "optimal_n": optimal_n,
        "optimal_lr": optimal_lr,
        "optimal_batch": optimal_batch,
        "optimal_dataset_fraction": None,
        "expected_accuracy": round(expected_accuracy or target_accuracy, 4),
        "ci_lower": round(float(ci_lower), 1),
        "ci_upper": round(float(ci_upper), 1),
        "confidence": confidence,
        "compute_saved": round(compute_saved, 4),
        "energy_saved_kwh": round(energy_saved_kwh, 6),
        "carbon_saved_g": round(carbon_saved_g, 4),
        "runs_used": n_runs,
        "prior_weight": prior_weight,
        "message": message,
    }
