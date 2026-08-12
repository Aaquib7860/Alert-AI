"""Phase 9 -- Demo-support routes.

NOT part of the master plan section 13 API contract -- these exist only to
power the local client-demonstration UI (master plan section 12) and
return real PII by design (see app/services/demo_data.py docstring).
Mounted under /api/v1/demo/* specifically so they are trivial to strip out
or gate behind auth/network controls before any real deployment -- they
must never ship to a production/internet-facing environment.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.services.demo_data import ALERT_TYPE_TO_SHEET, build_review_queue, load_data_overview, sample_alert

router = APIRouter(prefix="/api/v1/demo", tags=["demo (local use only -- not production API)"])


@router.get("/data-overview")
def data_overview() -> dict:
    return load_data_overview()


@router.get("/sample-alert")
def get_sample_alert(alert_type: str = Query(...), seed: int | None = None) -> dict:
    if alert_type not in ALERT_TYPE_TO_SHEET:
        raise HTTPException(status_code=400, detail=f"Unknown alert_type {alert_type!r}")
    try:
        return sample_alert(alert_type, seed=seed)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/review-queue")
def get_review_queue(alert_type: str = Query(...), n: int = 10, seed: int | None = None) -> dict:
    # seed=None -> build_review_queue seeds from system entropy, so each
    # "Build review queue" click in the UI draws a fresh random sample
    # instead of the same fixed set every time. Callers who want a
    # reproducible queue (e.g. tests) still pass an explicit ?seed=.
    if alert_type not in ALERT_TYPE_TO_SHEET:
        raise HTTPException(status_code=400, detail=f"Unknown alert_type {alert_type!r}")
    try:
        queue = build_review_queue(alert_type, n=n, seed=seed)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"alert_type": alert_type, "n": len(queue), "queue": queue}
