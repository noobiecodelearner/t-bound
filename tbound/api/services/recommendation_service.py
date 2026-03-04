"""
[api/services/recommendation_service). — recommendation engine.

DAYANCH — implement this file.

Takes a fitted scaling curve and a customer request.
Returns a complete recommendation object.

CRITICAL: Import Naeem's optimizer directly.
    from optimization.optimizer import ScaleOptimizer

Two recommendation paths:

PATH A — target accuracy (standard):
    Customer provides target_accuracy (e.g., 0.85).
    Returns: N*, lr*, batch*, CI bands, compute_saved, carbon_saved.

PATH B — compute budget (Chinchilla):
    Customer provides compute_budget_hours.
    Requires delta to be fitted (needs N × D grid data).
    Returns: N*, D*, lr*, batch*, expected_accuracy, compute_saved, carbon_saved.

Main function:
    get_recommendation(project_id, db, target_accuracy=None, compute_budget_hours=None)
        → validates: at least one of target_accuracy or compute_budget_hours must be set
        → gets fit from database via crud.get_fit_for_project()
        → if no fit exists or confidence is very_low:
              returns prior-based rough estimate with very wide CI
              and message: "Log more runs for a reliable recommendation."
        → if Path A: calls ScaleOptimizer.optimize_accuracy()
        → if Path B: calls ScaleOptimizer.optimize_compute_budget()
        → computes compute_saved and carbon_saved
        → returns Recommendation object (see below)

Recommendation object fields:
    optimal_n:                 int    — minimum params
    optimal_lr:                float  — optimal learning rate
    optimal_batch:             int    — optimal batch size
    optimal_dataset_fraction:  float  — D* (Path B only, else None)
    expected_accuracy:         float  — predicted accuracy at N*
    ci_lower:                  float  — lower bound 95% CI
    ci_upper:                  float  — upper bound 95% CI
    confidence:                str    — very_low | low | medium | high
    compute_saved:             float  — fraction 0-1 (vs training at 10× N*)
    energy_saved_kwh:          float
    carbon_saved_g:            float
    runs_used:                 int    — how many customer runs contributed
    prior_weight:              float  — 0.0 = all data, 1.0 = all prior
    message:                   str    — human-readable status message

Compute saved calculation:
    Naive approach = training at 10× N* for the same num_steps
    Savings = 1 - (N* / (10 × N*)) = 0.9 always? No —
    Better: compare to training at the largest model the customer has tried.
    compute_saved = 1 - (N* × num_steps) / (max_observed_params × num_steps)
                  = 1 - (N* / max_observed_params)

Carbon saved:
    Use training/energy.py estimate_carbon_grams() with the saved compute.

Notes:
    - Always return a recommendation even if confidence is very_low
      — just make the CI very wide and include the message
    - If customer has not run batch sweep, optimal_batch = 128 (sensible default)
    - If customer has not run lr sweep, optimal_lr = best lr observed so far

--- LARGE DATA / SUBSAMPLING EXTRAPOLATION ---

If fit["subsampling_extrapolation"] is True:

    1. Get full_dataset_size from the project's run records
       (customer passed this in tbound.init → stored on runs)

    2. Extrapolate D to full_dataset_size:
           D_extrap = full_dataset_size
           N_extrap, acc_extrap = ScaleOptimizer.optimize_compute_budget(
               compute_budget_hours, D_max=full_dataset_size
           )
           or for Path A:
           N_star = ScaleOptimizer.optimize_accuracy(
               target_accuracy, D=full_dataset_size
           )

    3. Inflate CI bands proportionally to extrapolation distance:
           extrap_factor = full_dataset_size / max(observed_dataset_sizes)
           ci_inflation = 1 + 0.1 * log10(extrap_factor)
           ci_lower = expected_accuracy - (expected_accuracy - ci_lower) * ci_inflation
           ci_upper = expected_accuracy + (ci_upper - expected_accuracy) * ci_inflation

    4. Add to recommendation:
           subsampling_warning = True
           message += " Warning: accuracy extrapolated beyond observed dataset sizes.
                        Use stratified sampling for best results."

    5. Add to recommendation object:
           subsampling_warning:   bool   — True if extrapolated beyond observed D
           extrapolation_factor:  float  — full_dataset_size / max_observed_dataset_size
"""