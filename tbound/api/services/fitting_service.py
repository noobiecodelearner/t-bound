"""
[api/services/fitting_service). — core scaling law fitting logic.

DAYANCH — implement this file.

This is the most important service. It is called every time a new run is logged.
It combines the prior with customer data to produce a fitted scaling curve.

CRITICAL: Import Naeem's code directly. Do not reimplement anything.
    from scaling.surface_fit import ScalingSurfaceFitter
    from scaling.bootstrap import BootstrapUncertainty
    from api.services.prior_service import get_prior

Main function to implement:
    fit_project(project_id, db)
        → reads all runs for project from database
        → determines confidence level based on run count
        → fits scaling curve (blending prior and data per confidence level)
        → stores result to fits table
        → returns fitted curve dict

--- DATASET SIZE HANDLING ---

Runs may have either dataset_fraction (relative) or dataset_size (absolute)
or both. Normalize before fitting:

    For each run:
        if dataset_fraction is not None:
            use dataset_fraction directly
        elif dataset_size is not None and full_dataset_size is not None:
            dataset_fraction = dataset_size / full_dataset_size
        elif dataset_size is not None:
            # unknown full size — use raw dataset_size for delta fitting
            # fitting_service uses dataset_size values directly (not fractions)
            # This works because delta is scale-invariant w.r.t. D units
            pass

For N×D surface fitting (delta):
    If customer has runs at multiple dataset sizes (large-data protocol):
        extract (params, dataset_size_or_fraction, accuracy) triples
        call ScalingSurfaceFitter.fit_nd_surface()
        this gives you both alpha and delta from their subsampling runs

For N-only fitting (alpha, no delta):
    Take best accuracy per params value (across all dataset sizes)
    call ScalingSurfaceFitter.fit_model_size()

--- SUBSAMPLING EXTRAPOLATION FLAG ---

If any run has subsampling_extrapolation=True (dataset_size < 1% of full):
    After fitting delta, flag the fit:
        fit["subsampling_extrapolation"] = True
    The recommendation_service uses this to:
        - Extrapolate D beyond observed range to full_dataset_size
        - Inflate CI bands at extrapolation distance (see recommendation_service)
        - Pass warning to dashboard

--- FITTING LOGIC BY RUN COUNT ---

    0 runs (very_low confidence):
        alpha = prior.alpha_mean
        a = None, b = None
        CI from prior.alpha_std (wide)

    1-2 runs (low confidence):
        alpha = prior.alpha_mean
        estimate a from observed points
        b = None
        CI still wide

    3-5 runs (medium confidence):
        fit alpha, a, b from data via ScalingSurfaceFitter.fit_model_size()
        regularize: alpha_final = (1-w)*alpha_data + w*prior.alpha_mean
        w = max(0, (6 - n_runs) / 6)
        bootstrap CI via BootstrapUncertainty

    6+ runs (high confidence):
        fit fully from data, prior_weight = 0.0
        full bootstrap CI

Confidence:
    0 runs:      "very_low"
    1-2 runs:    "low"
    3-5 runs:    "medium"
    6+ runs:     "high"

What fitting_service stores in fits table:
    alpha, a, b  (project-specific — never shared with other projects)
    beta, gamma, delta  (None until enough runs for those sweeps)
    r2, mae
    ci_lower, ci_upper
    confidence
    n_runs_used
    subsampling_extrapolation  (bool — True if any run used large-data protocol)

What fitting_service returns (dict):
    {
        "alpha":                    float,
        "a":                        float or None,
        "b":                        float or None,
        "delta":                    float or None,
        "r2":                       float or None,
        "ci_lower":                 float or None,
        "ci_upper":                 float or None,
        "confidence":               str,
        "n_runs":                   int,
        "prior_weight":             float,
        "subsampling_extrapolation": bool,
    }

Notes:
    - ScalingSurfaceFitter.fit_model_size() expects arrays: params_list, accuracy_list
    - ScalingSurfaceFitter.fit_nd_surface() expects: params_list, dataset_sizes_list, accuracy_list
    - Extract from Run objects returned by crud.get_runs_for_project()
    - Sort runs by params before fitting
    - If fitting fails fall back to prior — never crash
    - Log fitting errors with project_id for debugging
"""