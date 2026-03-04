"""[api/main). — FastAPI application for [t-bound)."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import runs, recommend, projects, health
from api.db.database import init_db

app = FastAPI(
    title="[t-bound) API",
    description="What if you had to train less?",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/v1")
app.include_router(runs.router, prefix="/v1")
app.include_router(recommend.router, prefix="/v1")
app.include_router(projects.router, prefix="/v1")


@app.on_event("startup")
async def startup():
    init_db()
