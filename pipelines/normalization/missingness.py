"""Explicit missingness indicators for model-relevant fields.

Master plan section 7: "Create explicit missingness indicators for
model-relevant fields." A model must be able to distinguish "this DOB is
unknown" from any imputed/default value -- so missingness becomes its own
feature rather than being silently filled in.

Fields that are 100% missing in the current sample (POB, in both name-alert
sheets) are deliberately excluded here -- master plan Appendix B rule:
"Never rely on a field that is 100% missing in the current sample." An
indicator for a field with zero non-missing rows would be a constant, not a
feature, and its presence in the pipeline invites someone to wire the raw
field back in later without re-checking.
"""
from __future__ import annotations

import pandas as pd

# Fields worth an explicit missingness flag, per sheet -- chosen because
# they are partially (not 0% or 100%) missing per the Phase 1 audit, i.e.
# missingness itself carries information here.
MISSINGNESS_INDICATOR_FIELDS: dict[str, list[str]] = {
    "CustomerViolation": [
        "Alerted Party DOB",
        "Hit Details (DOB)",
        "Hit Details (Nationality)",
    ],
    "TransactionNameViolation": [
        "Alerted Party DOB",
        "Hit Details (DOB)",
        "Hit Details (Nationality)",
    ],
    "Rule": [
        "Beneficiary Name",
        "Beneficiary Id Number",
        "Beneficiary Relationship",
        "Currency Name",
        "Purpose",
        "Purpose Code",
    ],
}


def add_missingness_indicators(df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    """Returns a copy of df with f"{field}_missing" boolean columns added
    for every field registered for this sheet in MISSINGNESS_INDICATOR_FIELDS.
    """
    out = df.copy()
    for field in MISSINGNESS_INDICATOR_FIELDS.get(sheet_name, []):
        if field not in out.columns:
            raise KeyError(
                f"Expected field '{field}' for missingness indicator on "
                f"sheet '{sheet_name}' not found in DataFrame."
            )
        out[f"{field}_missing"] = out[field].isna()
    return out
