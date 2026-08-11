"""Phase 8 -- FastAPI application entrypoint.

Run locally: uvicorn app.main:app --reload --port ${API_PORT:-8000}
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.demo_routes import router as demo_router
from app.api.routes import router

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="Alert Intelligence Engine",
    description=(
        "Unsupervised/self-supervised anomaly scoring for AML alert triage. "
        "No LLM, no paid AI API -- see docs/ for the full design rationale. "
        "Scores are novelty percentiles against a training population, "
        "never a probability of true match / false positive."
    ),
    version="0.1.0-poc",
)

app.include_router(router)
# Demo-support routes (app/api/demo_routes.py) -- NOT part of the master
# plan section 13 production API contract, returns real PII by design for
# local demo use. Mounted under a distinct /api/v1/demo prefix specifically
# so it is trivial to exclude before any real deployment.
app.include_router(demo_router)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def root() -> dict:
    return {
        "service": "alert-intelligence-engine",
        "docs": "/docs",
        "health": "/api/v1/health",
        "demo": "/demo",
    }


@app.get("/demo")
def demo_ui() -> FileResponse:
    return FileResponse(STATIC_DIR / "demo.html")
