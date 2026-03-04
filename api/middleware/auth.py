"""[api/middleware/auth). — API key authentication."""

from fastapi import Header, HTTPException, Depends
from sqlalchemy.orm import Session
from api.db.database import get_db
from api.db import crud


def verify_api_key(
    x_tbound_key: str = Header(..., alias="X-TBound-Key"),
    db: Session = Depends(get_db),
) -> str:
    project = crud.get_project_by_api_key(db, x_tbound_key)
    if not project:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return project.project_id
