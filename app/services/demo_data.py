"""Phase 9 -- Demo-only data access.

Everything here reads REAL client data (`data/raw/Alerts_Samples.xlsx`,
gitignored) to power a local demo -- it exists so the client demonstration
(master plan section 12) can "select a previously unseen historical
alert" and "send it through the same API path intended for production."

**This module must never be exposed outside a local/controlled demo
environment.** It returns raw PII (names, DOB, nationality) over HTTP by
design -- that is what makes it useful for a demo, and exactly why it is
kept in a separate, clearly-labeled module from the production scoring API
(app/services/scoring.py), not mixed into it. Master plan section 12
gate is "client demo ready", not "production ready" -- see
docs/phase9 report for the explicit removal note before any real
deployment.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

ALERT_TYPE_TO_SHEET = {
    "customer_name": "CustomerViolation",
    "transaction_name": "TransactionNameViolation",
    "transaction_rule": "Rule",
}

STATUS_COLUMN = {
    "CustomerViolation": "Alert Status",
    "TransactionNameViolation": "Alert Status",
    "Rule": "Status",
}


def load_data_overview() -> dict:
    """Aggregate-only summary (row counts, duplicates, missingness
    highlights, label-audit finding) from the already-committed Phase 1/2
    reports -- no PII, safe for any environment.
    """
    phase1 = json.load(open(REPO_ROOT / "evaluation" / "phase1_data_audit_report.json"))
    phase2 = json.load(open(REPO_ROOT / "evaluation" / "phase2_data_pipeline_report.json"))

    sheets = []
    for sheet_name in ("CustomerViolation", "TransactionNameViolation", "Rule"):
        p1 = phase1["schema_check"][sheet_name]
        sheets.append({
            "sheet_name": sheet_name,
            "rows": p1["rows"],
            "columns": p1["cols"],
            "exact_duplicate_rows": phase1["duplicate_rows"][sheet_name],
            "near_duplicate_candidate_rows": phase2["sheets"][sheet_name]["near_duplicate_candidate_rows"],
        })

    return {
        "sheets": sheets,
        "total_rows": sum(s["rows"] for s in sheets),
        "label_audit_finding": (
            "Operational status (Released/UPS/Followup) is NOT a trustworthy ground-truth "
            "label. A large share of UPS-status rows contain the literal phrase 'false "
            "positive' in the post-review comment (Phase 1 finding) -- this is why the "
            "engine is unsupervised/self-supervised, not a trained true-match classifier."
        ),
        "warnings": phase1["warnings"],
        "dataset_version": phase2["dataset_version"],
        "schema_version": phase2["schema_version"],
    }


def _load_raw_and_normalized(sheet_name: str):
    from pipelines.ingestion.load_alerts import load_raw_alerts
    from pipelines.normalization.pipeline import run_phase2_pipeline

    raw_sheets = load_raw_alerts()
    normalized_sheets, _ = run_phase2_pipeline(REPO_ROOT, persist=False)
    return raw_sheets[sheet_name], normalized_sheets[sheet_name]


def _test_positions(alert_type: str) -> list[int]:
    """Positional row indices belonging to the proven-unseen test split for
    this alert type -- same deterministic split (seed=42) verified
    bit-for-bit reproducible in Phase 7.
    """
    from pipelines.entity.combined_dataset import build_combined_entity_dataset
    from pipelines.entity.validation_splits import group_split_by_customer
    from pipelines.normalization.pipeline import run_phase2_pipeline

    normalized_sheets, _ = run_phase2_pipeline(REPO_ROOT, persist=False)

    if alert_type in ("customer_name", "transaction_name"):
        combined = build_combined_entity_dataset(
            normalized_sheets["CustomerViolation"], normalized_sheets["TransactionNameViolation"]
        )
        _, test_idx = group_split_by_customer(combined, test_size=0.25, random_state=42)
        sheet_name = ALERT_TYPE_TO_SHEET[alert_type]
        test_df = combined.iloc[test_idx]
        matching_record_ids = set(test_df[test_df["alert_source_sheet"] == sheet_name]["record_id"])
        sheet_df = normalized_sheets[sheet_name]
        return sheet_df.index[sheet_df["record_id"].isin(matching_record_ids)].tolist()

    rule_df = normalized_sheets["Rule"]
    _, test_idx = group_split_by_customer(rule_df, test_size=0.25, random_state=42)
    return list(test_idx)


def sample_alert(alert_type: str, seed: int | None = None) -> dict:
    """Returns a REAL, previously-unseen (test-split) historical alert:
    {alert_id, alert_type, raw_fields, historical_operational_outcome}.
    `historical_operational_outcome` is returned SEPARATELY from
    raw_fields specifically so a caller cannot accidentally feed it back
    into scoring -- it is evaluation context only (master plan section 12:
    "Show the historical operational outcome separately, clearly marked as
    evaluation context rather than model input").
    """
    from app.services.scoring import allowed_request_fields

    sheet_name = ALERT_TYPE_TO_SHEET[alert_type]
    raw_df, normalized_df = _load_raw_and_normalized(sheet_name)

    positions = _test_positions(alert_type)
    if not positions:
        raise ValueError(f"No test-split rows available for alert_type={alert_type!r}")

    rng = random.Random(seed)
    pos = rng.choice(positions)

    raw_row = raw_df.iloc[pos]
    fields = allowed_request_fields(alert_type)
    raw_fields = {}
    for c in fields:
        v = raw_row[c]
        if pd.isna(v):
            v = None
        elif hasattr(v, "item"):
            v = v.item()
        elif hasattr(v, "isoformat"):
            v = v.isoformat()
        raw_fields[c] = v

    status_col = STATUS_COLUMN[sheet_name]
    outcome = normalized_df.iloc[pos][status_col]
    if pd.isna(outcome):
        outcome = None

    record_id = normalized_df.iloc[pos]["record_id"]

    return {
        "alert_id": f"demo-{record_id}",
        "alert_type": alert_type,
        "raw_fields": raw_fields,
        "historical_operational_outcome": outcome,
        "_demo_note": (
            "Real historical alert from the proven-unseen test split (not used to train "
            "the active model). historical_operational_outcome is shown for evaluation "
            "context only -- it is Status/UPS/Released, which Phase 1 found is not a "
            "trustworthy ground-truth label, and it was never a model input."
        ),
    }


# Identity fields shown in the demo review-queue table (NOT part of the
# production /alerts/score API contract -- that stays PII-free by design,
# see app/services/scoring.py). Requested explicitly for local review use:
# a compliance reviewer scanning the queue needs to see who the alert is
# actually about, same as they'd already see in the single-alert panel's
# raw-fields table. Real PII, real names -- same warning as the rest of
# this module: local demo use only, never expose this route outside a
# controlled environment.
IDENTITY_FIELDS = {
    "CustomerViolation": {"name": "Alerted Party Name", "dob": "Alerted Party DOB", "id": "UIN", "country": "Alerted Party Nationality"},
    "TransactionNameViolation": {"name": "Alerted Party Name", "dob": "Alerted Party DOB", "id": "UIN", "country": "Alerted Party Nationality"},
    "Rule": {"name": "Customer Name", "dob": None, "id": "Customer Number", "country": "Customer Nationality"},
}


def build_review_queue(alert_type: str, n: int = 10, seed: int | None = 42) -> list[dict]:
    """Scores `n` real test-split alerts and returns them ranked by global
    novelty percentile, descending -- master plan section 12: "Show batch
    scoring and a simulated review-queue ranking." Each result also carries
    an `identity` block (name/DOB/ID/country) for display -- see
    IDENTITY_FIELDS docstring above for the PII scope note.
    """
    from app.services.scoring import score_alert

    positions = _test_positions(alert_type)
    if not positions:
        raise ValueError(f"No test-split rows available for alert_type={alert_type!r}")

    rng = random.Random(seed)
    chosen = rng.sample(positions, k=min(n, len(positions)))

    sheet_name = ALERT_TYPE_TO_SHEET[alert_type]
    raw_df, normalized_df = _load_raw_and_normalized(sheet_name)

    from app.services.scoring import allowed_request_fields

    fields = allowed_request_fields(alert_type)
    results = []
    for pos in chosen:
        raw_row = raw_df.iloc[pos]
        raw_fields = {}
        for c in fields:
            v = raw_row[c]
            if pd.isna(v):
                v = None
            elif hasattr(v, "item"):
                v = v.item()
            elif hasattr(v, "isoformat"):
                v = v.isoformat()
            raw_fields[c] = v

        record_id = normalized_df.iloc[pos]["record_id"]
        alert_id = f"demo-{record_id}"
        scored = score_alert(alert_type, alert_id, raw_fields)

        id_map = IDENTITY_FIELDS[sheet_name]
        scored["identity"] = {
            "name": raw_fields.get(id_map["name"]),
            "dob": raw_fields.get(id_map["dob"]) if id_map["dob"] else None,
            "id": raw_fields.get(id_map["id"]),
            "country": raw_fields.get(id_map["country"]),
        }
        results.append(scored)

    results.sort(key=lambda r: r["novelty"]["global"], reverse=True)
    return results
