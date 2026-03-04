"""
[api/routes/runs). — run logging endpoint.

DAYANCH: Receives run data from SDK and stores it.

WHAT TO IMPLEMENT:

POST /v1/runs
    Request body:
        project_id:       str
        params:           int      (required)
        val_accuracy:     float    (required, 0-1)
        learning_rate:    float    (optional)
        batch_size:       int      (optional)
        num_steps:        int      (optional — warn if missing)
        train_accuracy:   float    (optional)

        --- dataset size fields — accept one or both ---
        dataset_fraction:         float    (optional — relative, 0-1)
        dataset_size:             int      (optional — absolute sample count)
        full_dataset_size:        int      (optional — total dataset size, for normalization)

        --- subsampling flag ---
        subsampling_extrapolation: bool    (optional — set by SDK when dataset_size < 1% of full)

    Normalization logic:
        If dataset_size provided and full_dataset_size provided:
            dataset_fraction = dataset_size / full_dataset_size
        If only dataset_fraction provided:
            dataset_size = None (unknown absolute size)
        If only dataset_size provided (no full_dataset_size):
            dataset_fraction = None (unknown relative size)
            store dataset_size as-is — fitting_service handles raw sizes

    Response:
        run_id:                   str
        message:                  str
        runs_in_project:          int
        confidence:               str      (very_low | low | medium | high)
        subsampling_warning:      bool     (True if extrapolation flag set)

    Side effects:
        1. Store run in database via api/db/crud.py
        2. Always trigger fitting_service.fit(project_id) via BackgroundTasks
        3. Update project.run_count in database
        4. Update project.confidence_level in database

NOTES:
- Validate params > 0
- Validate val_accuracy between 0 and 1
- Warn (but accept) if num_steps missing — add warning to response
- Use FastAPI BackgroundTasks for async fit so POST returns in < 100ms always
- If subsampling_extrapolation=True, store flag in run record and return
  subsampling_warning=True in response — dashboard reads this to show warning
"""

# TODO: implement this file
raise NotImplementedError("api/routes/runs.py not yet implemented — see docstring")