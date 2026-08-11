"""Phase 8 -- Minimal feedback capture.

Master plan section 17: every production alert should eventually record
the ML recommendation and the final compliance outcome, independently
validated before becoming a supervised training label. This is the
capture endpoint only -- validation, label-quality checks, and the
supervised calibration pipeline itself are Phase 12 (Feedback Learning),
explicitly out of scope here. This just appends one JSON line per
feedback event to a local file; nothing here treats the outcome as a
validated label.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def get_feedback_path() -> Path:
    configured = os.environ.get("FEEDBACK_STORE_PATH")
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else REPO_ROOT / configured
    return REPO_ROOT / "feedback" / "feedback_log.jsonl"


def record_feedback(
    alert_id: str, alert_type: str, model_version: str,
    compliance_outcome: str, notes: str | None = None,
) -> dict:
    path = get_feedback_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "alert_id": alert_id,
        "alert_type": alert_type,
        "model_version": model_version,
        "compliance_outcome": compliance_outcome,
        "notes": notes,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return entry
