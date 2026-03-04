"""[api/routes/projects). — project management endpoints."""

import uuid
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.db.database import get_db
from api.db import crud
from api.middleware.auth import verify_api_key

router = APIRouter()


class CreateProjectRequest(BaseModel):
    project_id: str
    architecture_family: str
    domain: str


@router.post("/projects")
def create_project(body: CreateProjectRequest, db: Session = Depends(get_db)):
    existing = crud.get_project(db, body.project_id)
    if existing:
        raise HTTPException(status_code=409, detail="Project already exists")
    api_key = "tb_" + uuid.uuid4().hex[:24]
    project = crud.create_project(
        db,
        project_id=body.project_id,
        api_key=api_key,
        architecture_family=body.architecture_family,
        domain=body.domain,
    )
    return {"project_id": project.project_id, "api_key": project.api_key}


@router.get("/projects/{project_id}/runs")
def get_project_runs(
    project_id: str,
    db: Session = Depends(get_db),
    _pid: str = Depends(verify_api_key),
):
    runs = crud.get_runs_for_project(db, project_id)
    return {
        "runs": [
            {
                "run_id": r.run_id,
                "params": r.params,
                "val_accuracy": r.val_accuracy,
                "learning_rate": r.learning_rate,
                "batch_size": r.batch_size,
                "dataset_fraction": r.dataset_fraction,
                "num_steps": r.num_steps,
                "logged_at": r.logged_at,
            }
            for r in runs
        ],
        "total_runs": len(runs),
    }


@router.get("/projects/{project_id}/curve")
def get_project_curve(
    project_id: str,
    target_accuracy: float = 0.85,
    db: Session = Depends(get_db),
    _pid: str = Depends(verify_api_key),
):
    runs = crud.get_runs_for_project(db, project_id)
    fit = crud.get_fit_for_project(db, project_id)

    params_list = [r.params for r in runs]
    accuracies_list = [r.val_accuracy for r in runs]

    if not params_list or fit is None or fit.alpha is None:
        return {
            "params": params_list,
            "accuracies": accuracies_list,
            "curve_params": [],
            "curve_mean": [],
            "ci_lower": [],
            "ci_upper": [],
            "n_star": None,
            "target_accuracy": target_accuracy,
            "alpha": fit.alpha if fit else None,
            "confidence": fit.confidence if fit else "very_low",
            "n_runs": len(runs),
        }

    min_n = min(params_list) * 0.5
    max_n = max(params_list) * 5.0
    curve_n = np.logspace(np.log10(min_n), np.log10(max_n), 100).tolist()

    a = fit.a or 0.95
    b = fit.b or 1.0
    alpha = fit.alpha

    curve_mean = [float(a - b * (n ** -alpha)) for n in curve_n]

    # CI bands using alpha uncertainty
    alpha_lo = max(0.01, alpha * 0.8)
    alpha_hi = alpha * 1.2
    ci_lower_curve = [float(a - b * (n ** -alpha_lo)) for n in curve_n]
    ci_upper_curve = [float(a - b * (n ** -alpha_hi)) for n in curve_n]

    n_star = None
    if target_accuracy < a:
        try:
            n_star = int((b / (a - target_accuracy)) ** (1.0 / alpha))
        except Exception:
            pass

    return {
        "params": params_list,
        "accuracies": accuracies_list,
        "curve_params": curve_n,
        "curve_mean": curve_mean,
        "ci_lower": ci_lower_curve,
        "ci_upper": ci_upper_curve,
        "n_star": n_star,
        "target_accuracy": target_accuracy,
        "alpha": alpha,
        "confidence": fit.confidence,
        "n_runs": len(runs),
    }
