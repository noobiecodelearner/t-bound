"""
[api/main). — FastAPI application for [t-bound).

DAYANCH: Main API entry point. Mount all routers here.

WHAT TO IMPLEMENT:

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import runs, recommend, projects, health
from api.db.database import init_db
from api.middleware.auth import APIKeyMiddleware

app = FastAPI(
    title="[t-bound) API",
    description="What if you had to train less?",
    version="0.1.0",
)

# CORS — allow dashboard and SDK to connect
app.add_middleware(CORSMiddleware, ...)

# Auth — validate API key on every request except /v1/health
app.add_middleware(APIKeyMiddleware)

# Mount routers
app.include_router(health.router,    prefix="/v1")
app.include_router(runs.router,      prefix="/v1")
app.include_router(recommend.router, prefix="/v1")
app.include_router(projects.router,  prefix="/v1")

@app.on_event("startup")
async def startup():
    init_db()  # create SQLite tables

TO RUN:
    uvicorn api.main:app --reload --port 8000

NOTES:
- Health endpoint must NOT require auth — SDK pings it to check connectivity
- All other endpoints require X-TBound-Key header
- SQLite for launch, Postgres for production
"""

# TODO: implement this file
raise NotImplementedError("api/main.py not yet implemented — see docstring")
