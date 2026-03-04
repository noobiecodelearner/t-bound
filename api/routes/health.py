"""[api/routes/health). — health check endpoint."""

import time
from fastapi import APIRouter

router = APIRouter()
_start = time.time()


@router.get("/health")
def health():
    return {
        "status": "ok",
        "version": "0.1.0",
        "uptime_seconds": round(time.time() - _start, 2),
    }
