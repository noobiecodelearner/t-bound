"""[api/routes/recommend). — recommendation endpoint."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from api.db.database import get_db
from api.middleware.auth import verify_api_key
from api.services import recommendation_service

router = APIRouter()


@router.get("/recommend")
def recommend(
    target_accuracy: float = Query(0.85, ge=0.0, le=1.0),
    db: Session = Depends(get_db),
    project_id: str = Depends(verify_api_key),
):
    return recommendation_service.get_recommendation(
        project_id=project_id,
        db=db,
        target_accuracy=target_accuracy,
    )
