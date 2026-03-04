"""
[api/routes/projects). — project management endpoints.

DAYANCH: CRUD for projects. Dashboard reads from these.

WHAT TO IMPLEMENT:

GET /v1/projects/{project_id}
    Response:
        project_id, api_key_prefix (first 8 chars only),
        architecture, domain, created_at, last_updated,
        run_count, confidence_level, latest_alpha, latest_r2

GET /v1/projects/{project_id}/runs
    Query params: page (default 1), page_size (default 50)
    Response: list of runs, sorted by params ascending
              each run: run_id, params, val_accuracy, learning_rate,
              batch_size, dataset_fraction, num_steps, logged_at,
              generalization_gap, gen_warning

GET /v1/projects/{project_id}/curve
    Response: fitted curve data for visualization
        param_counts:     list of N values (log-spaced grid)
        mean_predictions: list of predicted accuracies at each N
        ci_lower:         list of lower CI bound at each N
        ci_upper:         list of upper CI bound at each N
        observed_params:  list of actual logged N values
        observed_accs:    list of actual logged val accuracies
        alpha:            float
        a:                float (ceiling)
        b:                float (coefficient)
        r2:               float
        confidence:       str

DELETE /v1/projects/{project_id}
    Deletes all runs and fits for project.
    Requires matching API key.
    Response: {deleted: true, runs_deleted: int}

NOTES:
- /curve endpoint is called by dashboard every 30 seconds (polling)
- Make it fast: cache the fit result in database, recompute only when new runs arrive
- For CI bands on the curve: evaluate bootstrap CI at each N in the grid
"""

# TODO: implement this file
raise NotImplementedError("api/routes/projects.py not yet implemented — see docstring")
