"""[api/db/models). — SQLAlchemy ORM models."""

from sqlalchemy import Column, String, Float, Integer, ForeignKey
from api.db.database import Base


class Project(Base):
    __tablename__ = "projects"

    project_id = Column(String, primary_key=True)
    api_key = Column(String, index=True)
    architecture_family = Column(String)
    domain = Column(String)
    created_at = Column(String)
    last_updated = Column(String)
    run_count = Column(Integer, default=0)
    confidence_level = Column(String, default="very_low")


class Run(Base):
    __tablename__ = "runs"

    run_id = Column(String, primary_key=True)
    project_id = Column(String, ForeignKey("projects.project_id"), index=True)
    params = Column(Integer)
    val_accuracy = Column(Float)
    learning_rate = Column(Float)
    batch_size = Column(Integer)
    dataset_fraction = Column(Float)
    num_steps = Column(Integer)
    logged_at = Column(String)


class Fit(Base):
    __tablename__ = "fits"

    fit_id = Column(String, primary_key=True)
    project_id = Column(String, ForeignKey("projects.project_id"), index=True)
    alpha = Column(Float, nullable=True)
    beta = Column(Float, nullable=True)
    gamma = Column(Float, nullable=True)
    delta = Column(Float, nullable=True)
    a = Column(Float, nullable=True)
    b = Column(Float, nullable=True)
    r2 = Column(Float, nullable=True)
    mae = Column(Float, nullable=True)
    ci_lower = Column(Float, nullable=True)
    ci_upper = Column(Float, nullable=True)
    confidence = Column(String)
    n_runs_used = Column(Integer)
    fitted_at = Column(String)
