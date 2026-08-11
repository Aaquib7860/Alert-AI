"""Phase 8 -- FastAPI application entrypoint.

Run locally: uvicorn app.main:app --reload --port ${API_PORT:-8000}
"""
from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router

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


@app.get("/")
def root() -> dict:
    return {"service": "alert-intelligence-engine", "docs": "/docs", "health": "/api/v1/health"}
