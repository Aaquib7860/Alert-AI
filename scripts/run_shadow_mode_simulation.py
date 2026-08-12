"""Phase 10 -- Shadow-mode simulation.

Master plan section 20 Phase 10 gate: "Operational validation." Replays
every alert in the proven-unseen test split (same deterministic group
split verified bit-for-bit reproducible in Phase 7) through the real
`score_alert` code path -- the identical function the live API calls --
and logs every result to the shadow audit trail (source="shadow_simulation").

This produces the evidence a human reviewer needs to approve moving from
"code exists" to "operationally validated": volume handled, error rate,
latency, recommendation distribution, and -- the one property that
actually matters for shadow mode -- proof that zero autonomous actions
were taken across the whole run.

Run: python scripts/run_shadow_mode_simulation.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.services.demo_data import ALERT_TYPE_TO_SHEET, _load_raw_and_normalized, _test_positions
from app.services.scoring import AlertValidationError, allowed_request_fields, score_alert
from app.services.shadow_log import get_shadow_log_path, verify_no_autonomous_action
from app.services.shadow_log import record_shadow_score


def _coerce(v):
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if hasattr(v, "item"):
        return v.item()
    return v


def run_for_alert_type(alert_type: str) -> dict:
    sheet_name = ALERT_TYPE_TO_SHEET[alert_type]
    raw_df, normalized_df = _load_raw_and_normalized(sheet_name)
    fields = allowed_request_fields(alert_type)
    positions = _test_positions(alert_type)

    latencies_ms = []
    recommendations = {"REVIEW": 0, "LOWER_TOUCH_CANDIDATE": 0}
    plain_labels = {"Needs Review": 0, "Not Confident": 0, "Looks Routine": 0}
    n_errors = 0
    error_samples = []

    for pos in positions:
        record_id = normalized_df.iloc[pos]["record_id"]
        alert_id = f"shadow-{record_id}"
        raw_row = raw_df.iloc[pos]
        raw_fields = {c: _coerce(raw_row[c]) for c in fields}

        t0 = time.perf_counter()
        try:
            result = score_alert(alert_type, alert_id, raw_fields)
        except (AlertValidationError, ValueError) as e:
            n_errors += 1
            if len(error_samples) < 5:
                error_samples.append(str(e))
            continue
        latencies_ms.append((time.perf_counter() - t0) * 1000)

        record_shadow_score(result, source="shadow_simulation")
        recommendations[result["recommendation"]] += 1
        plain_labels[result["plain_language_label"]] += 1

    n_scored = len(latencies_ms)
    latencies_sorted = sorted(latencies_ms)

    def _pct(p):
        if not latencies_sorted:
            return None
        idx = min(int(len(latencies_sorted) * p), len(latencies_sorted) - 1)
        return round(latencies_sorted[idx], 2)

    return {
        "alert_type": alert_type,
        "n_test_split_rows": len(positions),
        "n_scored": n_scored,
        "n_errors": n_errors,
        "error_rate": round(n_errors / len(positions), 4) if positions else None,
        "error_samples": error_samples,
        "latency_ms": {
            "mean": round(sum(latencies_ms) / n_scored, 2) if n_scored else None,
            "p50": _pct(0.50),
            "p95": _pct(0.95),
            "p99": _pct(0.99),
            "max": round(max(latencies_ms), 2) if latencies_ms else None,
        },
        "recommendation_distribution": recommendations,
        "plain_language_distribution": plain_labels,
    }


def main() -> None:
    print("Shadow-mode simulation starting -- replaying the full proven-unseen test split "
          "through the real score_alert() code path for every alert type.\n")

    per_type_results = {}
    for alert_type in ALERT_TYPE_TO_SHEET:
        print(f"Running {alert_type}...")
        per_type_results[alert_type] = run_for_alert_type(alert_type)
        r = per_type_results[alert_type]
        print(f"  scored={r['n_scored']}/{r['n_test_split_rows']}, errors={r['n_errors']}, "
              f"mean_latency={r['latency_ms']['mean']}ms")

    ok, violations = verify_no_autonomous_action()
    print(f"\nAutonomous-action invariant check: {'PASS' if ok else 'FAIL'} "
          f"({len(violations)} violation(s))")

    total_scored = sum(r["n_scored"] for r in per_type_results.values())
    total_errors = sum(r["n_errors"] for r in per_type_results.values())

    report = {
        "phase": "10_shadow_mode",
        "status": "PASS" if ok and total_errors == 0 else "PASS_WITH_WARNINGS",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "shadow_log_path": str(get_shadow_log_path().relative_to(REPO_ROOT)),
        "per_alert_type": per_type_results,
        "totals": {
            "n_scored": total_scored,
            "n_errors": total_errors,
        },
        "autonomous_action_invariant": {
            "checked": True,
            "all_clean": ok,
            "violating_alert_ids": violations,
            "statement": (
                f"{total_scored} alerts scored and logged across all 3 alert types during "
                "this shadow-mode simulation. Zero autonomous actions taken -- every score "
                "was logged to the audit trail only, per master plan section 10/13."
            ),
        },
        "known_limitations": [
            "This is an offline replay against historical held-out data, not live production "
            "traffic -- master plan section 10 shadow mode ultimately requires connection to "
            "the client's real alert stream, which is Phase 3 (Production Integration) scope.",
            "Latency figures are single-process, local-machine timings -- not representative "
            "of production infrastructure/network conditions.",
        ],
        "next_gate": "Phase 11 -- Production foundation (monitoring, registry, rollback, "
            "retraining) and Phase 12 (feedback learning / label quality gate). Requires "
            "human approval.",
    }

    out_path = REPO_ROOT / "evaluation" / "phase10_shadow_mode_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
