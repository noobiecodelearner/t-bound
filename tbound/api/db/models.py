"""
[api/db/models). — SQLAlchemy ORM models.

DAYANCH — implement this file.

What to do:
    Define three SQLAlchemy ORM model classes using Base from database.py.

Table 1 — Project:
    __tablename__ = "projects"
    Columns:
        project_id:          String, primary key
        api_key:             String, indexed (for fast lookup on every request)
        architecture_family: String (cnn | transformer | mlp | etc.)
        domain:              String (vision | nlp | tabular)
        created_at:          String (ISO 8601 UTC)
        last_updated:        String (ISO 8601 UTC)
        run_count:           Integer, default 0
        confidence_level:    String (very_low | low | medium | high)

Table 2 — Run:
    __tablename__ = "runs"
    Columns:
        run_id:              String, primary key
        project_id:          String, ForeignKey("projects.project_id"), indexed
        params:              Integer
        val_accuracy:        Float
        learning_rate:       Float
        batch_size:          Integer
        dataset_fraction:    Float   (nullable — relative size 0-1)
        dataset_size:        Integer (nullable — absolute sample count)
        full_dataset_size:   Integer (nullable — total dataset size, set once in tbound.init)
        num_steps:           Integer
        logged_at:           String (ISO 8601 UTC)
        subsampling_extrapolation: Boolean, default False (True when dataset_size < 1% of full)

Table 3 — Fit:
    __tablename__ = "fits"
    Columns:
        fit_id:              String, primary key
        project_id:          String, ForeignKey("projects.project_id"), indexed
        alpha:               Float (nullable)
        beta:                Float (nullable)
        gamma:               Float (nullable)
        delta:               Float (nullable)
        a:                   Float (nullable — accuracy ceiling, customer-specific)
        b:                   Float (nullable — scaling coefficient, customer-specific)
        r2:                  Float (nullable)
        mae:                 Float (nullable)
        ci_lower:            Float (nullable)
        ci_upper:            Float (nullable)
        confidence:          String
        n_runs_used:         Integer
        fitted_at:           String (ISO 8601 UTC)

Notes:
    - a and b ARE stored here (per-project, never shared across projects)
    - Only exponents (alpha, beta, gamma, delta) are shared via prior.csv
    - After defining models, call Base.metadata.create_all(bind=engine) in api/main.py
"""