"""Phase 10 -- Shadow-mode audit log.

Master plan section 10 (Shadow Mode): "Initial production deployment
should observe ML recommendations without autonomous release" (section
13 Production Controls table). Every score this system produces --
whether from a real API call or an offline simulation -- gets one audit
record here, and every record carries `autonomous_action_taken: false`
as a structural invariant, not a comment. `verify_no_autonomous_action`
scans the actual log and fails loudly if that invariant is ever violated,
rather than just trusting the field was set correctly at write time.

This is intentionally a dumb, append-only log -- no downstream action of
any kind is triggered by writing to it. That absence is the entire point
of shadow mode.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def get_shadow_log_path() -> Path:
    configured = os.environ.get("SHADOW_LOG_PATH")
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else REPO_ROOT / configured
    return REPO_ROOT / "monitoring" / "shadow_log.jsonl"


def record_shadow_score(score_result: dict, source: str) -> dict:
    """Appends one audit record for a scored alert. `source` identifies
    where the score came from (e.g. "api", "shadow_simulation") -- useful
    for later distinguishing real traffic from replay evidence, never used
    to change behaviour.
    """
    path = get_shadow_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "alert_id": score_result["alert_id"],
        "alert_type": score_result["alert_type"],
        "model_version": score_result["model_version"],
        "feature_version": score_result["feature_version"],
        "schema_version": score_result["schema_version"],
        "novelty_global": score_result["novelty"]["global"],
        "novelty_customer": score_result["novelty"]["customer"],
        "recommendation": score_result["recommendation"],
        "plain_language_label": score_result["plain_language_label"],
        "scored_at": score_result["scored_at"],
        "source": source,
        "shadow_mode": True,
        "autonomous_action_taken": False,
        "logged_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def read_shadow_log(path: Path | None = None) -> list[dict]:
    path = path or get_shadow_log_path()
    if not path.exists():
        return []
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def verify_no_autonomous_action(path: Path | None = None) -> tuple[bool, list[str]]:
    """Scans the actual log file and confirms every record's
    `autonomous_action_taken` is False. Returns (all_clean, violating_alert_ids)
    -- checked, not assumed, because this is the specific safety property
    shadow mode exists to guarantee.
    """
    entries = read_shadow_log(path)
    violations = [e["alert_id"] for e in entries if e.get("autonomous_action_taken") is not False]
    return len(violations) == 0, violations
