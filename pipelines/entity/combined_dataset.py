"""Combines CustomerViolation + TransactionNameViolation into one shared
entity dataset.

Master plan section 6.1 (feasibility report) / section 2 (master plan):
"Customer Name Alerts and Transaction Name Alerts share the entity-
intelligence problem... A shared representation layer can be used while
retaining alert-type context." Phase 3 already gave both sheets an
identical feature-source column schema (see `features/entity_features.py`
NAME_COLUMNS / CONTEXT_CATEGORICAL_COLUMNS / NATIONALITY_COLUMNS -- the
CustomerViolation and TransactionNameViolation entries are already
byte-identical lists) specifically so this combination is possible without
inventing a mapping.

IMPORTANT documented limitation: `customer_id` stays sheet-namespaced as
produced by Phase 2 (e.g. "customerviolation:customer:1234" vs
"transactionnameviolation:customer:1234"), even though both sheets use the
same source column name `UIN`. The client has not confirmed these two UIN
spaces are the same identifier space, so this module does NOT assume they
match. Consequence: a real customer who appears in both sheets will have
their history undercounted (Phase 3's `customer_prior_alert_count` and
Phase 5's model will see them as two separate customer histories). This is
the conservative, safe choice -- an unverified cross-sheet join would risk
a worse mistake (merging two different people's history because they
happen to share an integer). Revisit once the client confirms the UIN
semantics.
"""
from __future__ import annotations

import pandas as pd

# Columns present, with identical meaning, in both sheets -- used to
# validate the combination is legitimate (fails loudly if a future schema
# change breaks the shared-representation assumption).
SHARED_COLUMNS = [
    "customer_id", "record_id",
    "Alerted Party Name (Normalized)", "Alerted Party DOB (Parsed)",
    "Alerted Party DOB_missing", "Alerted Party Nationality (Normalized)",
    "Hit Details (Name) (Normalized)", "Hit Details (DOB) (Parsed)",
    "Hit Details (DOB) (Year)", "Hit Details (DOB) (MultiValue)",
    "Hit Details (DOB) (Unresolved)", "Hit Details (DOB)_missing",
    "Hit Details (Nationality) (Normalized)", "Hit Details (Nationality)_missing",
    "Matched Screening % (Parsed)", "Sanctions Screening List Name",
    "Alerted Party", "Branch Name", "Alert Generated Date & Time (Parsed)",
    "Alert Type", "Customer Type",
]


def build_combined_entity_dataset(
    customer_violation_df: pd.DataFrame,
    transaction_name_violation_df: pd.DataFrame,
) -> pd.DataFrame:
    missing_cv = [c for c in SHARED_COLUMNS if c not in customer_violation_df.columns]
    missing_tnv = [c for c in SHARED_COLUMNS if c not in transaction_name_violation_df.columns]
    if missing_cv or missing_tnv:
        raise ValueError(
            f"Shared-representation assumption broken -- missing columns. "
            f"CustomerViolation missing: {missing_cv}, "
            f"TransactionNameViolation missing: {missing_tnv}"
        )

    cv = customer_violation_df[SHARED_COLUMNS].copy()
    cv["alert_source_sheet"] = "CustomerViolation"

    tnv = transaction_name_violation_df[SHARED_COLUMNS].copy()
    tnv["alert_source_sheet"] = "TransactionNameViolation"

    combined = pd.concat([cv, tnv], axis=0, ignore_index=False)
    # record_id is unique per source row already (sha1 of sheet name + raw
    # row content, see pipelines/ingestion/identifiers.py) -- re-index so
    # downstream code has a clean, collision-free positional index without
    # losing that traceable id
    combined = combined.reset_index(drop=True)
    return combined
