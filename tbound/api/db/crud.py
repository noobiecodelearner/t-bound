"""
[api/db/crud). — database CRUD operations.

DAYANCH — implement this file.

What to do:
    Write the following functions using SQLAlchemy sessions.
    All functions take db: Session as first argument.

Project operations:
    get_project(db, project_id)
        → returns Project or None

    get_project_by_api_key(db, api_key)
        → returns Project or None
        → used by auth middleware on every request

    create_project(db, project_id, api_key, architecture_family, domain)
        → creates and returns Project

    update_project_run_count(db, project_id)
        → increments run_count by 1
        → updates last_updated to now()
        → returns updated Project

    update_project_confidence(db, project_id, confidence_level)
        → updates confidence_level field
        → returns updated Project

Run operations:
    create_run(db, run_id, project_id, params, val_accuracy,
               learning_rate, batch_size, dataset_fraction, num_steps)
        → creates and returns Run

    get_runs_for_project(db, project_id)
        → returns List[Run] ordered by params ascending
        → used by fitting_service to get all runs before fitting

    get_run_count(db, project_id)
        → returns int

Fit operations:
    create_or_update_fit(db, project_id, alpha, beta, gamma, delta,
                         a, b, r2, mae, ci_lower, ci_upper,
                         confidence, n_runs_used)
        → upserts fit for project (one fit record per project, updated as runs arrive)
        → returns Fit

    get_fit_for_project(db, project_id)
        → returns Fit or None
        → returns None if fewer than 3 runs have been logged

Notes:
    - Use db.add(), db.commit(), db.refresh() pattern
    - Always call db.refresh(obj) after commit to return updated object
    - Wrap writes in try/except, rollback on failure
"""
