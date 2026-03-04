"""[api/routes/runs). — run logging endpoint."""

import uuid
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.db.database import get_db
from api.db import crud
from api.middleware.auth import verify_api_key
from api.services import fitting_service

router = APIRouter()


class LogRunRequest(BaseModel):
    params: int
    val_accuracy: float
    learning_rate: float
    batch_size: int
    dataset_fraction: float = 1.0
    num_steps: int


@router.post("/runs")
def log_run(
    body: LogRunRequest,
    db: Session = Depends(get_db),
    project_id: str = Depends(verify_api_key),
):
    run_id = "run_" + uuid.uuid4().hex[:8]
    crud.create_run(
        db,
        run_id=run_id,
        project_id=project_id,
        params=body.params,
        val_accuracy=body.val_accuracy,
        learning_rate=body.learning_rate,
        batch_size=body.batch_size,
        dataset_fraction=body.dataset_fraction,
        num_steps=body.num_steps,
    )
    crud.update_project_run_count(db, project_id)

    # Synchronous fitting — SDK waits for this
    fit_result = fitting_service.fit_project(project_id, db)

    n_runs = crud.get_run_count(db, project_id)
    return {
        "run_id": run_id,
        "message": "Run logged successfully.",
        "n_runs": n_runs,
        "confidence": fit_result.get("confidence", "very_low"),
    }
