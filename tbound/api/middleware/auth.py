"""
[api/middleware/auth). — API key authentication middleware.

DAYANCH — implement this file.

What to do:
    Write a FastAPI dependency function that validates API keys.
    Every protected route injects this dependency.

How API keys work:
    Customer includes their key in every request header:
        X-TBound-Key: tb_key_abc123xyz

    The middleware:
        1. Reads the X-TBound-Key header
        2. Looks up the key in the projects table via crud.get_project_by_api_key()
        3. If key missing → raise HTTPException(status_code=401, detail="Missing API key")
        4. If key invalid → raise HTTPException(status_code=401, detail="Invalid API key")
        5. If key valid → returns the project_id attached to that key

Usage in routes:
    @router.post("/runs")
    def log_run(project_id: str = Depends(verify_api_key), db: Session = Depends(get_db)):
        ...

Example implementation:
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

Notes:
    - Keep this simple — do not add rate limiting yet (V2 concern)
    - The health endpoint (/v1/health) should NOT require authentication
    - Project creation endpoint should require a master admin key, not a project key
      (or just skip auth for project creation in V1 — keep it simple for launch)
"""
