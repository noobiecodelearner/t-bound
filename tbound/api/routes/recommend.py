"""
[api/routes/recommend). — recommendation endpoint.

DAYANCH: Returns N*, lr*, batch*, CI, confidence for a project.

WHAT TO IMPLEMENT:

GET /v1/recommend
    Query params:
        project_id:           str    (required)
        target_accuracy:      float  (optional, one of these two required)
        compute_budget_hours: float  (optional, one of these two required)

    Response (Path A — target accuracy):
        n_star:                   int
        lr_star:                  float
        batch_star:               int
        expected_accuracy:        float
        ci_lower:                 float
        ci_upper:                 float
        confidence:               str
        compute_saved_fraction:   float
        energy_saved_kwh:         float
        carbon_saved_g:           float
        runs_used:                int
        prior_weight:             float   (0=all prior, 1=all data)
        alpha:                    float
        fit_r2:                   float
        warning:                  str     (null or warning message)

    Response (Path B — compute budget / Chinchilla):
        n_star:                   int
        d_star:                   int     (optimal dataset size)
        lr_star:                  float
        batch_star:               int
        expected_accuracy:        float
        compute_budget_hours:     float
        energy_kwh:               float
        carbon_g:                 float
        alpha:                    float
        delta:                    float
        surface_r2:               float

    Errors:
        400 if neither target_accuracy nor compute_budget_hours provided
        404 if project_id not found
        422 if < 3 runs logged (return n_runs_available in error body)

LOGIC:
    1. Load project from database
    2. Load runs from database
    3. Call api/services/recommendation_service.py
    4. Recommendation service calls:
           Prior path (0-5 runs): api/services/prior_service.py
           Data path  (6+ runs):  optimization/optimizer.py (import from Naeem's code)
    5. Return formatted response

IMPORTANT:
    Import ScaleOptimizer directly from optimization/optimizer.py
    Never reimplement the fitting logic here — use Naeem's code.
"""

# TODO: implement this file
raise NotImplementedError("api/routes/recommend.py not yet implemented — see docstring")
