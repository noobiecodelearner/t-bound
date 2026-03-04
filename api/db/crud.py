"""[api/db/crud). — database CRUD operations."""

from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy.orm import Session
from api.db.models import Project, Run, Fit


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Projects ──────────────────────────────────────────────────────────────────

def get_project(db: Session, project_id: str) -> Optional[Project]:
    return db.query(Project).filter(Project.project_id == project_id).first()


def get_project_by_api_key(db: Session, api_key: str) -> Optional[Project]:
    return db.query(Project).filter(Project.api_key == api_key).first()


def create_project(db: Session, project_id: str, api_key: str,
                   architecture_family: str, domain: str) -> Project:
    now = _now()
    project = Project(
        project_id=project_id,
        api_key=api_key,
        architecture_family=architecture_family,
        domain=domain,
        created_at=now,
        last_updated=now,
        run_count=0,
        confidence_level="very_low",
    )
    try:
        db.add(project)
        db.commit()
        db.refresh(project)
        return project
    except Exception:
        db.rollback()
        raise


def update_project_run_count(db: Session, project_id: str) -> Project:
    project = get_project(db, project_id)
    try:
        project.run_count = (project.run_count or 0) + 1
        project.last_updated = _now()
        db.commit()
        db.refresh(project)
        return project
    except Exception:
        db.rollback()
        raise


def update_project_confidence(db: Session, project_id: str,
                               confidence_level: str) -> Project:
    project = get_project(db, project_id)
    try:
        project.confidence_level = confidence_level
        project.last_updated = _now()
        db.commit()
        db.refresh(project)
        return project
    except Exception:
        db.rollback()
        raise


# ── Runs ──────────────────────────────────────────────────────────────────────

def create_run(db: Session, run_id: str, project_id: str, params: int,
               val_accuracy: float, learning_rate: float, batch_size: int,
               dataset_fraction: float, num_steps: int) -> Run:
    run = Run(
        run_id=run_id,
        project_id=project_id,
        params=params,
        val_accuracy=val_accuracy,
        learning_rate=learning_rate,
        batch_size=batch_size,
        dataset_fraction=dataset_fraction,
        num_steps=num_steps,
        logged_at=_now(),
    )
    try:
        db.add(run)
        db.commit()
        db.refresh(run)
        return run
    except Exception:
        db.rollback()
        raise


def get_runs_for_project(db: Session, project_id: str) -> List[Run]:
    return db.query(Run).filter(Run.project_id == project_id).all()


def get_run_count(db: Session, project_id: str) -> int:
    return db.query(Run).filter(Run.project_id == project_id).count()


# ── Fits ──────────────────────────────────────────────────────────────────────

def create_or_update_fit(db: Session, project_id: str, **fit_fields) -> Fit:
    fit = db.query(Fit).filter(Fit.project_id == project_id).first()
    try:
        if fit is None:
            import uuid
            fit = Fit(
                fit_id=str(uuid.uuid4()),
                project_id=project_id,
                fitted_at=_now(),
                **fit_fields,
            )
            db.add(fit)
        else:
            for k, v in fit_fields.items():
                setattr(fit, k, v)
            fit.fitted_at = _now()
        db.commit()
        db.refresh(fit)
        return fit
    except Exception:
        db.rollback()
        raise


def get_fit_for_project(db: Session, project_id: str) -> Optional[Fit]:
    return db.query(Fit).filter(Fit.project_id == project_id).first()
