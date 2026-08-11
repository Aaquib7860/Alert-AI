"""Feature-availability timestamp rule.

Master plan section 7: "Create a feature-availability timestamp rule so
temporal experiments only use information available at scoring time."

The "scoring time" for an alert is the moment it was generated/scanned --
NOT the moment it was closed or reviewed. Any field whose timestamp is
after the alert's scoring-time cutoff must not be used when building
features for that alert in a temporal (time-forward) experiment.

This module does not correct the timestamp anomalies found in Phase 1
(e.g. TransactionNameViolation alert-generated-before-transaction in 80.2%
of rows) -- those remain flagged, unexplained data-quality findings per the
master plan rule "Do not silently fix; validate business timestamp
semantics before temporal modeling." This module only enforces which
*columns* are allowed to feed a feature at scoring time, not the accuracy
of the timestamps themselves.
"""
from __future__ import annotations

import pandas as pd

# The column that defines "scoring time" for each sheet -- i.e. the moment
# the alert entered the system and a live model would have had to score it.
SCORING_TIME_COLUMN: dict[str, str] = {
    "CustomerViolation": "Alert Generated Date & Time",
    "TransactionNameViolation": "Alert Generated Date & Time",
    "Rule": "Scan Date",
}

# Columns whose values only exist strictly after scoring time -- i.e. the
# leakage registry from Phase 1, restated here in temporal-availability
# terms. Any column not listed here or in SCORING_TIME_COLUMN is assumed
# available at scoring time unless proven otherwise.
POST_SCORING_COLUMNS: dict[str, list[str]] = {
    "CustomerViolation": [
        "Maker Name", "Maker Comment Date", "Maker Comment",
        "Alert Closure Date & Time", "Alert Status",
    ],
    "TransactionNameViolation": [
        "Maker Name", "Maker Comment Date", "Maker Comment", "Alert Status",
    ],
    "Rule": ["Comment", "Actiondate", "Action Taken By", "Status"],
}


def assert_no_post_scoring_columns(columns: list[str], sheet_name: str) -> None:
    """Raises if any column in `columns` is a known post-scoring-time
    (leakage) field for this sheet. Intended to be called on the final
    feature column list right before model training/inference, as a hard
    stop rather than a soft warning.
    """
    forbidden = set(POST_SCORING_COLUMNS.get(sheet_name, []))
    violations = [c for c in columns if c in forbidden]
    if violations:
        raise ValueError(
            f"Post-scoring-time (leakage) columns present in feature set for "
            f"'{sheet_name}': {violations}. These are only known after human "
            f"review and must never be live model input."
        )


def filter_available_before(
    df: pd.DataFrame,
    event_time_col: str,
    cutoff_col: str,
) -> pd.DataFrame:
    """Returns only rows where `event_time_col` is at or before
    `cutoff_col` (both must be datetime columns). Used to build
    "prior transactions only" behavioural aggregates for a given customer
    in a time-forward validation experiment -- rows with a null event time
    or cutoff are excluded (cannot prove availability, so excluded rather
    than assumed available).
    """
    for col in (event_time_col, cutoff_col):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame")

    mask = df[event_time_col].notna() & df[cutoff_col].notna() & (
        df[event_time_col] <= df[cutoff_col]
    )
    return df[mask]
